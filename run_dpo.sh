python DPO_trainer.py \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --model_name_or_path 'meta-llama/Llama-2-7b-hf' \
    --load_in_4bit \
    --use_peft \
    --learning_rate 1e-3 \
    --report_to 'wandb' \
    --run_name 'llama_dpo_gpt4_high_to_low_beta_0.3' \
    --max_length 1024 \
    --max_prompt_length 768 \
    --num_train_epochs 5 \
    --max_steps -1 \
    --evaluation_strategy 'epoch' \
    --eval_steps -1 \
    --logging_strategy 'steps' \
    --log_steps 20 \
    --logging_first_step \
    --save_strategy 'epoch' \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --metric_for_best_model 'eval_metrics_policy_UMLS_cuis_f' \
    --alignment_function 'dpo' \
    --output_dir './results/DPO_model/gpt4_edits_high_to_low/DPO-LLaMA(1|1|0.3)' \
    --alpha1 1.0 \
    --alpha2 1.0 \
    --beta 0.3 \
    --synthetic_data_type 'gpt4_edits_high_to_low' \

# python DPO_trainer.py \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --model_name_or_path 'gpt2' \
#     --learning_rate 1e-4 \
#     --report_to 'wandb' \
#     --run_name 'gpt2_dpo_gpt4_high_to_low_beta_0.3' \
#     --max_length 1024 \
#     --max_prompt_length 768 \
#     --num_train_epochs 5 \
#     --max_steps -1 \
#     --evaluation_strategy 'epoch' \
#     --eval_steps -1 \
#     --logging_strategy 'steps' \
#     --log_steps 20 \
#     --logging_first_step \
#     --save_strategy 'epoch' \
#     --save_total_limit 3 \
#     --load_best_model_at_end \
#     --metric_for_best_model 'eval_metrics_policy_UMLS_cuis_f' \
#     --alignment_function 'dpo' \
#     --output_dir './results/DPO_model/gpt4_edits_high_to_low/DPO-gpt2(1|1|0.3)' \
#     --alpha1 1.0 \
#     --alpha2 1.0 \
#     --beta 0.3 \
#     --synthetic_data_type 'gpt4_edits_high_to_low' \
