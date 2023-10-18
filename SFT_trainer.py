from transformers import HfArgumentParser
from trainer import ScriptArguments, load_dataset, trainer

parser = HfArgumentParser(ScriptArguments)

#for SFT
script_args = parser.parse_args_into_dataclasses(
        args=[
                '--per_device_train_batch_size', '4',
                '--per_device_eval_batch_size', '16',
                '--gradient_accumulation_steps', '2',
                '--model_name_or_path', 'gpt2',
                # '--model_name_or_path', 'huggy llama/llama-7b',
                # '--model_name_or_path', 'meta-llama/Llama-2-7b-hf',
                '--load_in_4bit',
                '--use_peft',
                # '--learning_rate', '1e-3',
                '--learning_rate', '1e-4',
                # '--report_to', 'wandb',
                '--run_name', 'SFT-avs-gpt2',
                '--max_length', '1024',
                '--max_prompt_length', '768',
                '--num_train_epochs', '20',
                '--max_steps', '-1',
                '--evaluation_strategy', 'epoch',
                '--eval_steps', '-1',
                # '--eval_first_step',
                '--logging_strategy', 'steps',
                '--log_steps', '10',
                '--logging_first_step',
                '--save_strategy', 'epoch',
                '--save_steps', '-1',
                '--save_total_limit', '3',
                '--load_best_model_at_end',
                '--metric_for_best_model', 'metrics_policy_rouge1',
                '--alignment_function', 'sft',
                '--output_dir', './results/avs/SFT_model/gpt2',
                # '--output_dir', './results/SFT_model/llama2_7b',
            ]
        )[0]

# 2. Load training dataset
# train_dataset = load_dataset("train", sanity_check=script_args.sanity_check)
train_dataset = load_dataset("sub_eval", sanity_check=script_args.sanity_check, alignment_function=script_args.alignment_function)

# 3. Load evaluation dataset
eval_dataset = load_dataset("sub_eval", sanity_check=script_args.sanity_check, alignment_function=script_args.alignment_function)

dpo_trainer = trainer(script_args, train_dataset, eval_dataset)