import copy
import os
import sys

import torch
from transformers import HfArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import set_seed
from dataclasses import asdict
from transformers.integrations.deepspeed import HfDeepSpeedConfig
import wandb
from transformers import DataCollatorForLanguageModeling

python_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(python_path)
from log import print
from arguments import ModelArguments, DataArguments, MyTrainingArguments
from mydatasets_continued_pretraining import ContinuedPretrainingDataset
from lomo_trainer import LOMOTrainer
from utils import DataCollatorForCauselLM, DataCollatorForCausalLM

from huggingface_hub import login
login(os.environ["HUGGINGFACE_TOKEN"])

def compute_metrics(all_pred, eval_dataset, eval_prefix=None):
    """Simple perplexity computation for continued pretraining evaluation"""
    # For continued pretraining, we typically just monitor loss/perplexity
    # This is a placeholder - actual metrics would be computed in the trainer
    return {'perplexity': 0.0}


def train_continued_pretraining():
    # ========== 1. logs and args ==========
    torch.set_default_dtype(torch.float16)
    parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
    if sys.argv[-1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    model_name = model_args.model_name_or_path.split('/')[-1]
    tag_name = '_'.join([data_args.dataset_name, model_name, training_args.tag] if training_args.tag else [data_args.dataset_name, model_name])
    hparam_name = 'continued_pretraining'
    if training_args.optim != 'sgd':
        hparam_name += '_' + training_args.optim
    if training_args.learning_rate != 5e-4:
        hparam_name += '_lr' + str(training_args.learning_rate)
    if training_args.per_device_train_batch_size != 8:
        hparam_name += '_bs' + str(training_args.per_device_train_batch_size)
    if training_args.lr_scheduler_type != 'linear':
        hparam_name += '_' + training_args.lr_scheduler_type
    if training_args.warmup != 0:
        hparam_name += '_warmup' + str(training_args.warmup)
    if training_args.clip_grad_norm and training_args.clip_grad_norm > 0:
        hparam_name += '_clipnorm' + str(training_args.clip_grad_norm)
    if training_args.clip_grad_value and training_args.clip_grad_value > 0:
        hparam_name += '_clipgrad' + str(training_args.clip_grad_value)
    if training_args.clip_loss_value and training_args.clip_loss_value > 0:
        hparam_name += '_cliploss' + str(training_args.clip_loss_value)
    
    # training_args.output_dir = os.path.join('outputs', tag_name, hparam_name)
    base_output_dir = training_args.output_dir if training_args.output_dir else 'outputs'
    experiment_path = os.path.join(tag_name, hparam_name)
    training_args.output_dir = os.path.join(base_output_dir, experiment_path)
    
    print(f"📁 Output directory: {training_args.output_dir}")
    
    # Ensure output directory exists
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Set WANDB mode to avoid interactive prompts
    if training_args.wandb_mode == 'disabled':
        os.environ['WANDB_MODE'] = 'disabled'
    elif training_args.wandb_mode == 'offline':
        os.environ['WANDB_MODE'] = 'offline'
    elif training_args.tag == 'debug':
        os.environ['WANDB_MODE'] = 'offline'
    else:
        os.environ['WANDB_MODE'] = 'online'
    
    # Set WANDB_SILENT to avoid login prompts
    os.environ['WANDB_SILENT'] = 'true'
    
    if training_args.local_rank in [-1, 0] and training_args.wandb_mode != 'disabled':
        wandb_config = copy.deepcopy(asdict(training_args))
        wandb_config.update(asdict(model_args))
        wandb_config.update(asdict(data_args))
        
        run_name = training_args.wandb_run_name
        if run_name is None:
            run_name = tag_name if hparam_name == 'continued_pretraining' else '_'.join([tag_name, hparam_name.replace('continued_pretraining_', '')])
        
        # Login with API key if available
        if 'WANDB_API_KEY' in os.environ and training_args.wandb_mode == 'online':
            wandb.login(key=os.environ['WANDB_API_KEY'])
       
        wandb.init(
            project=training_args.wandb_project if hasattr(training_args, 'wandb_project') else "lomo-continued-pretraining",
            entity=training_args.wandb_entity if hasattr(training_args, 'wandb_entity') else 'lomo_exp',
            name=run_name,
            config=wandb_config
        )

    # ========== 2. Load pretrained model and tokenizer. ==========
    ds_config = training_args.deepspeed
    dschf = HfDeepSpeedConfig(ds_config)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.gradient_checkpointing = training_args.gradient_checkpointing
    
    if training_args.resume_from_checkpoint is not None:
        print(f'Load checkpoint from {training_args.resume_from_checkpoint}.')
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path if training_args.resume_from_checkpoint is None else training_args.resume_from_checkpoint,
        local_files_only=model_args.local_file,
        config=config,
        cache_dir=model_args.cache_dir

    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        padding_side='left',
        cache_dir=model_args.cache_dir

    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ========== 3. Preprocessing the datasets. ==========
    # Validate dataset configuration
    if not data_args.dataset_path:
        raise ValueError("dataset_path is required in data_args")
    
    train_dataset = ContinuedPretrainingDataset(data_args, tokenizer, split='train')
    
    # DEBUG: Check dataset format
    print("🔍 DEBUG: Checking dataset format...")
    sample = train_dataset[0]
    print(f"Dataset sample keys: {sample.keys()}")
    for key, value in sample.items():
        print(f"  {key}: type={type(value)}, len={len(value) if hasattr(value, '__len__') else 'N/A'}")
        if key == 'input_ids' and hasattr(value, '__getitem__'):
            print(f"    First 5 values: {value[:5]}")
    print("✅ Dataset format check completed")
    
    eval_dataset = None
    if training_args.do_eval:
        try:
            eval_dataset = ContinuedPretrainingDataset(data_args, tokenizer, split='validation')
        except:
            print("No evaluation split found, using training split for evaluation")
            eval_dataset = train_dataset

    # ========== 4. Initialize our Trainer. ==========
    # data_collator = DataCollatorForCauselLM(tokenizer, max_length=data_args.data_max_length, padding_side='left')
    # data_collator = DataCollatorForCausalLM(tokenizer)
    
    # Custom data collator to filter out text fields
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # ⚠️ Causal LM
    )
    
    trainer = LOMOTrainer(
        model=model,
        training_args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # ========== 5. Training. ==========
    if training_args.do_train:
        print("***** Starting Continued Pretraining *****")
        print(f"  Dataset: {data_args.dataset_name}")
        print(f"  Model: {model_args.model_name_or_path}")
        print(f"  Train on inputs: {data_args.train_on_inputs}")
        print(f"  Max length: {data_args.data_max_length}")
        print(f"  Learning rate: {training_args.learning_rate}")
        print(f"  Batch size per device: {training_args.per_device_train_batch_size}")
        print(f"  Number of epochs: {training_args.num_train_epochs}")
        
        trainer.train()

    # ========== 6. Evaluation. ==========
    if training_args.do_eval and eval_dataset is not None:
        trainer.eval(trainer.global_step, 0, trainer.eval_dataset, trainer.eval_dataloader, 'eval')

    print("***** Training Completed *****")
    print(f"Model saved to: {training_args.output_dir}")


if __name__ == "__main__":
    train_continued_pretraining()