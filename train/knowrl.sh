model_path=nvidia/OpenMath-Nemotron-1.5B
project_name=KnowRL # for wandb
experiment_name=KnowRL-Nemotron-1.5B # for wandb

train_files=train/train_data/cbrs.parquet
test_file_aime24=train/val/aime24/aime24.parquet 
test_file_aime25=train/val/aime25/aime25.parquet
test_file_brumo_2025=train/val/brumo_2025/brumo_2025.parquet
test_file_hmmt_25_2=train/val/hmmt_25_2/hmmt_25_2.parquet
test_files=["$test_file_aime24","$test_file_aime25","$test_file_brumo_2025","$test_file_hmmt_25_2"]

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_temperature=0.7
val_top_p=0.9

loss_agg_mode="token-mean" 

max_prompt_length=8192
max_response_length=32768
train_prompt_bsz=256
gen_prompt_bsz=512

actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
use_dynamic_bsz=True
infer_micro_batch_size=null # null for use_dynamic_bsz=True
train_micro_batch_size=null # null for use_dynamic_bsz=True
offload=False

enable_filter_groups=True
filter_groups_metric=seq_reward
max_num_gen_batches=10

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env=verl/trainer/runtime_env.yaml \
    --no-wait \
    -- \
    python3 -m recipe.dapo.main_dapo \
        algorithm.adv_estimator=grpo \
        data.train_files=$train_files \
        data.val_files=$test_files \
        data.train_batch_size=${train_prompt_bsz} \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.gen_batch_size=${gen_prompt_bsz} \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_micro_batch_size} \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
        actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.actor.clip_ratio_high=0.26 \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
        actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
        actor_rollout_ref.model.path=$model_path \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_micro_batch_size} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
        actor_rollout_ref.rollout.temperature=${temperature} \
        actor_rollout_ref.rollout.top_p=${top_p} \
        actor_rollout_ref.rollout.top_k="${top_k}" \
        actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
        actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
        actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=32 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_micro_batch_size} \
        actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
        algorithm.use_kl_in_reward=False \
        algorithm.filter_groups.enable=${enable_filter_groups} \
        algorithm.filter_groups.metric=${filter_groups_metric} \
        algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name=$project_name \
        trainer.experiment_name=$experiment_name \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=4 \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        trainer.val_before_train=True \
        trainer.default_local_dir=checkpoints/${project_name}/${experiment_name} \
        trainer.validation_data_dir=validation_data/${project_name}/${experiment_name} \
        trainer.rollout_data_dir=rollout_data/${project_name}/${experiment_name} \
        trainer.total_epochs=150 $@