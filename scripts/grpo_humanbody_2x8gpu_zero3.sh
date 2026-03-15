experiment_name=grpo_human_body_2x8

# 2-machine, 8 GPUs each (16 total)
# Run on machine 0: bash grpo_humanbody_2x8gpu_zero3.sh
# Run on machine 1: MACHINE_RANK=1 MASTER_ADDR=<machine0_ip> bash grpo_humanbody_2x8gpu_zero3.sh

MACHINE_RANK=${MACHINE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}

accelerate launch \
    --config_file configs/deepspeed/zero3_2x8gpu_offload.yaml \
    --machine_rank $MACHINE_RANK \
    --main_process_ip $MASTER_ADDR \
    -m reward_model_train.cli.grpo_train \
    --dataset_name "/m2v_intern/wangqunzhong/research/asset/kwai_data/dataset" \
    --output_dir log/model_checkpoints/$experiment_name \
    --remove_unused_columns False \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 4 \
    --num_iterations 8 \
    --num_generations 384 \
    --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --report_to "wandb" \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 100 \
    --num_train_epochs 5 \
    --use_peft True \
    --lora_task_type "CAUSAL_LM" \
    --lora_r 64 \
    --lora_target_modules "q_proj" "v_proj" \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --log_completions True \
    --data_pipeline "qwen2.5-humanbody-grpo" \
    --data_select_ratio 0.1 \
    --cache_dir "/m2v_intern/wangqunzhong/research/asset/huggingface/model/Qwen/Qwen2.5-VL-7B-Chat" \
    --torch_dtype "bfloat16" \
    --data_source "video" \
    --do_train True \
    --bf16 True \
    --max_completion_length 1024 \
    --fps 8. \
    --max_prompt_length 2048 \
    --run_name $experiment_name
