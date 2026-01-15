from verl import DataProto

from verl.workers.utils.padding import left_right_2_no_padding
from verl.utils import tensordict_utils as tu
from verl.utils.py_functional import rename_dict
from verl.trainer.ppo.ray_trainer import compute_response_mask, reduce_metrics, RayPPOTrainer
from tqdm import tqdm
import uuid
import numpy as np
import torch
from pprint import pprint
from verl.utils.debug import marked_timer
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
import megatron.core.parallel_state as mpu
from AutoTuner.utils.gpu_info import resolve_n_gpus

class LocalTrainer:
    def __init__(self, config, actor, train_dataloader):
        """
        Docstring for __init__
        
        :param self: this class itself, all the functions related to working group should be set to local
        :param config: just pass the config from main() is enough
        :param actor: pass the constructed actor from outside
        :param train_dataloader: pass the constructed train_dataloader from outside
        
        You may add more params in the future, but please add it carefully and cautiously!
        """
        self.async_rollout_mode = True
        
        # pass some params to the fields of the obj
        self.actor = actor
        self.config = config
        self.train_dataloader = train_dataloader
        
        # patch some functions as tool functions if we do not need to change it
        self._get_gen_batch = RayPPOTrainer._get_gen_batch
    
    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )

        actor_output = self.actor.update_actor(batch_td)
        
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        # modify key name
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        return actor_output

    def _get_dp_size(self) -> int:
        """
        Returns:
            The data parallel size (number of DP ranks).
        """
        return mpu.get_data_parallel_world_size()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        
        # TODO: we may have to check and FIX this function
        # since we don't use working group, we have to find a way to fix it locally
        dp_size = self._get_dp_size()
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
    
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        # self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        # if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        #     val_metrics = self._validate()
        #     assert val_metrics, f"{val_metrics=}"
        #     pprint(f"Initial validation metrics: {val_metrics}")
        #     logger.log(data=val_metrics, step=self.global_steps)
        #     if self.config.trainer.get("val_only", False):
        #         return

        # if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
        #     rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
        #     rollout_skip.wrap_generate_sequences()

        # add tqdm
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        # prev_step_profile = False
        # curr_step_profile = (
        #     self.global_steps in self.config.global_profiler.steps
        #     if self.config.global_profiler.steps is not None
        #     else False
        # )
        # next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                # if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                #     self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                # with marked_timer("start_profile", timing_raw):
                #     self._start_profiling(
                #         not prev_step_profile and curr_step_profile
                #         if self.config.global_profiler.profile_continuous_steps
                #         else curr_step_profile
                #     )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                
                with marked_timer("step", timing_raw):
                    # generate a batch
                    # with marked_timer("gen", timing_raw, color="red"):
                    #     if not self.async_rollout_mode:
                    #         gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                    #     else:
                    #         gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                    #     timing_raw.update(gen_batch_output.meta_info["timing"])
                    #     gen_batch_output.meta_info.pop("timing", None)

                    # if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    #     if self.reward_fn is None:
                    #         raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                    #     with marked_timer("gen_max", timing_raw, color="purple"):
                    #         gen_baseline_batch = deepcopy(gen_batch)
                    #         gen_baseline_batch.meta_info["do_sample"] = False
                    #         if not self.async_rollout_mode:
                    #             gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                    #         else:
                    #             gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                    #         batch = batch.union(gen_baseline_output)
                    #         # compute reward model score on batch
                    #         rm_scores = None
                    #         if self.use_rm and "rm_scores" not in batch.batch.keys():
                    #             if not self.use_reward_loop:
                    #                 rm_scores = self.rm_wg.compute_rm_score(batch)
                    #             else:
                    #                 assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                    #                 rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                    #             batch = batch.union(rm_scores)

                    #         # Compute or extract reward for REMAX baseline
                    #         reward_baseline_tensor = self._compute_or_extract_reward(
                    #             batch, reward_fn=self.reward_fn, sum_reward=True
                    #         )

                    #         keys_to_pop = set(gen_baseline_output.batch.keys())
                    #         if rm_scores is not None:
                    #             keys_to_pop.update(rm_scores.batch.keys())
                    #         batch.pop(batch_keys=list(keys_to_pop))

                    #         batch.batch["reward_baselines"] = reward_baseline_tensor

                    #         del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    
                    # TODO: we should take a look at this, it might be useful, so I carry the function
                    # but this function has something to fix, ctrl and click the func name to see
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # with marked_timer("reward", timing_raw, color="yellow"):
                    #     # compute reward model score
                    #     if self.use_rm and "rm_scores" not in batch.batch.keys():
                    #         if not self.use_reward_loop:
                    #             reward_tensor = self.rm_wg.compute_rm_score(batch)
                    #         else:
                    #             assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                    #             reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                    #         batch = batch.union(reward_tensor)

                    #     # Compute or extract reward for training
                    #     if self.config.reward_model.launch_reward_fn_async:
                    #         future_reward = compute_reward_async.remote(
                    #             data=batch, config=self.config, tokenizer=self.tokenizer
                    #         )
                    #     else:
                    #         reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                    #             batch, reward_fn=self.reward_fn, return_dict=False
                    #         )

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    # rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    # bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    # if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                    #     from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                    #     apply_bypass_mode(
                    #         batch=batch,
                    #         rollout_corr_config=rollout_corr_config,
                    #         policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                    #     )
                    # else:  # Recompute old_log_probs
                    #     with marked_timer("old_log_prob", timing_raw, color="blue"):
                    #         old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                    #         entropys = old_log_prob.batch["entropys"]
                    #         response_masks = batch.batch["response_mask"]
                    #         actor_config = self.config.actor_rollout_ref.actor
                    #         entropy_agg = agg_loss(
                    #             loss_mat=entropys,
                    #             loss_mask=response_masks,
                    #             loss_agg_mode=actor_config.loss_agg_mode,
                    #             loss_scale_factor=actor_config.loss_scale_factor,
                    #         )
                    #         old_log_prob_metrics = {
                    #             "actor/entropy": entropy_agg.detach().item(),
                    #             "perf/mfu/actor_infer": old_log_prob_mfu,
                    #         }
                    #         metrics.update(old_log_prob_metrics)
                    #         old_log_prob.batch.pop("entropys")
                    #         batch = batch.union(old_log_prob)
                    #         if "rollout_log_probs" in batch.batch.keys():
                    #             from verl.utils.debug.metrics import calculate_debug_metrics

                    #             metrics.update(calculate_debug_metrics(batch))

                    # assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    # if self.use_reference_policy:
                    #     # compute reference log_prob
                    #     with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                    #         ref_log_prob = self._compute_ref_log_prob(batch)
                    #         batch = batch.union(ref_log_prob)

                    # compute values
                    # if self.use_critic:
                    #     with marked_timer("values", timing_raw, color="cyan"):
                    #         values = self._compute_values(batch)
                    #         batch = batch.union(values)

                    # with marked_timer("adv", timing_raw, color="brown"):
                    #     # we combine with rule-based rm
                    #     reward_extra_infos_dict: dict[str, list]
                    #     if self.config.reward_model.launch_reward_fn_async:
                    #         reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    #     batch.batch["token_level_scores"] = reward_tensor

                    #     if reward_extra_infos_dict:
                    #         batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    #     # compute rewards. apply_kl_penalty if available
                    #     if self.config.algorithm.use_kl_in_reward:
                    #         batch, kl_metrics = apply_kl_penalty(
                    #             batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    #         )
                    #         metrics.update(kl_metrics)
                    #     else:
                    #         batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    #     # Compute rollout correction: IS weights, rejection sampling, and metrics
                    #     # Only runs in decoupled mode (computes once per batch using stable π_old)
                    #     # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                    #     if (
                    #         rollout_corr_config is not None
                    #         and "rollout_log_probs" in batch.batch
                    #         and not bypass_recomputing_logprobs  # Only in decoupled mode
                    #     ):
                    #         from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                    #         # Compute IS weights, apply rejection sampling, compute metrics
                    #         batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                    #         # IS and off-policy metrics already have rollout_corr/ prefix
                    #         metrics.update(is_metrics)

                    #     # compute advantages, executed on the driver process
                    #     norm_adv_by_std_in_grpo = self.config.algorithm.get(
                    #         "norm_adv_by_std_in_grpo", True
                    #     )  # GRPO adv normalization factor

                    #     batch = compute_advantage(
                    #         batch,
                    #         adv_estimator=self.config.algorithm.adv_estimator,
                    #         gamma=self.config.algorithm.gamma,
                    #         lam=self.config.algorithm.lam,
                    #         num_repeat=self.config.actor_rollout_ref.rollout.n,
                    #         norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    #         config=self.config.algorithm,
                    #     )

                    # update critic
                    # if self.use_critic:
                    #     with marked_timer("update_critic", timing_raw, color="pink"):
                    #         critic_output = self._update_critic(batch)
                    #     critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    #     metrics.update(critic_output_metrics)

                    # implement critic warmup
                    # if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                    with marked_timer("update_actor", timing_raw, color="red"):
                        actor_output = self._update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    # rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    # if rollout_data_dir:
                    #     self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                # if (
                #     self.val_reward_fn is not None
                #     and self.config.trainer.test_freq > 0
                #     and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                # ):
                #     with marked_timer("testing", timing_raw, color="green"):
                #         val_metrics: dict = self._validate()
                #         if is_last_step:
                #             last_val_metrics = val_metrics
                #     metrics.update(val_metrics)

                # # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                # esi_close_to_expiration = should_save_ckpt_esi(
                #     max_steps_duration=self.max_steps_duration,
                #     redundant_time=self.config.trainer.esi_redundant_time,
                # )
                # # Check if the conditions for saving a checkpoint are met.
                # # The conditions include a mandatory condition (1) and
                # # one of the following optional conditions (2/3/4):
                # # 1. The save frequency is set to a positive value.
                # # 2. It's the last training step.
                # # 3. The current step number is a multiple of the save frequency.
                # # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                # if self.config.trainer.save_freq > 0 and (
                #     is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                # ):
                #     if esi_close_to_expiration:
                #         print("Force saving checkpoint: ESI instance expiration approaching.")
                #     with marked_timer("save_checkpoint", timing_raw, color="green"):
                #         self._save_checkpoint()

                # with marked_timer("stop_profile", timing_raw):
                #     next_step_profile = (
                #         self.global_steps + 1 in self.config.global_profiler.steps
                #         if self.config.global_profiler.steps is not None
                #         else False
                #     )
                #     self._stop_profiling(
                #         curr_step_profile and not next_step_profile
                #         if self.config.global_profiler.profile_continuous_steps
                #         else curr_step_profile
                #     )
                #     prev_step_profile = curr_step_profile
                #     curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                # TODO: we may have to check and fully make use of this, so I did not remove this
                # but, for compute_data_metrics, we may have to check it for there might be some problems
                # metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                
                # real TODO: fix this "get_n_gpus"
                n_gpus = resolve_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                # if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                #     self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                # if (
                #     hasattr(self.config.actor_rollout_ref.actor, "profiler")
                #     and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                # ):
                #     self.actor_rollout_wg.dump_memory_snapshot(
                #         tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                #     )

                if is_last_step:
                    # if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    #     self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
                

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                # if hasattr(self.train_dataset, "on_batch_end"):
                #     # The dataset may be changed after each training batch
                #     self.train_dataset.on_batch_end(batch=batch)