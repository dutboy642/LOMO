from dataclasses import dataclass, field
from typing import Optional
from transformers import Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="llama-7B")
    cache_dir: Optional[str] = field(default='../llama/checkpoint')
    # llama_dir: Optional[str] = field(default='/remote-home/klv/exps/MossOn3090/llama')
    local_file: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_dir: str = field(default='data')
    dataset_name: str = field(default='openbookqa')
    dataset_path: Optional[str] = field(default=None, metadata={"help": "Path to dataset (local file or HuggingFace repo)"})
    text_column: str = field(default="text", metadata={"help": "Column name containing text data"})
    refresh: bool = field(default=False, metadata={"help": "Whether to refresh the data."})

    data_tag: str = field(default='src')
    prompt_type: str = field(default='natural', metadata={"help": "The type of prompt, including [natural, brown]."})
    train_on_inputs: bool = field(default=False, metadata={"help": "Whether to train on input."})
    data_max_length: int = field(default=1024)
    few_shot_size: int = field(default=-1)
    in_context_learning: bool = field(default=False, metadata={"help": "Whether to use in-context learning."})
    sample_size: int = field(default=-1, metadata={"help": "Number of samples to use from dataset. -1 means use all data."})


@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    tag: str = field(default=None, metadata={"help": "Tag for the experiment."})

    # Wandb configuration
    wandb_mode: str = field(default='offline', metadata={"help": "Wandb mode: online, offline, or disabled"})
    wandb_project: str = field(default='lomo', metadata={"help": "Wandb project name"})
    wandb_entity: str = field(default='lomo_exp', metadata={"help": "Wandb entity name"})
    wandb_run_name: str = field(default=None, metadata={"help": "Wandb run name. If None, auto-generated"})

    predict_with_generate: bool = field(default=False, metadata={"help": "Whether to use generate for prediction."})

    clip_grad_norm: float = field(default=None, metadata={
        "help": "Maximum gradient normalized value (for gradient clipping)."})  # recommend 1.0
    clip_grad_value: float = field(default=None, metadata={"help": "Maximum gradient value (for gradient clipping)."})
    clip_loss_value: float = field(default=None,
                                   metadata={"help": "Maximum loss value (for token loss clipping)."})  # recommend 5.0
    warmup: float = field(default=0.0,
                          metadata={"help": "The number of warmup steps (int) or the warmup ratio (float)."})
    evaluation_strategy: str = field(default='epoch', metadata={})
    max_length: int = field(default=20, metadata={"help": "The maximum length of the sequence to be generated."})
    max_new_tokens: int = field(default=None, metadata={
        "help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."})
    do_sample: bool = field(default=False,
                            metadata={"help": "Whether or not to use sampling ; use greedy decoding otherwise."})
    temperature: float = field(default=1.0,
                               metadata={"help": "The value used to modulate the next token probabilities."})
    top_k: int = field(default=50, metadata={
        "help": "If set to int > 0, only the top k tokens with the highest probability will be considered for generation."})
    top_p: float = field(default=1.0, metadata={
        "help": "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation."})
    typical_p: float = field(default=1.0, metadata={
        "help": "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation."})
    repetition_penalty: float = field(default=1.0, metadata={
        "help": "The parameter for repetition penalty. 1.0 means no penalty. See this paper for more details: https://arxiv.org/pdf/1909.05858.pdf"})

    length_normalization: bool = field(default=True, metadata={"help": "Whether to normalize the loss by the length of the input."})
    unconditional_normalization: bool = field(default=False, metadata={"help": "Whether to normalize the loss by the length of the input."})

    hf_learning_rate: float = field(default=5e-4, metadata={"help": "The learning rate for the HF optimizer."})
    hf_weight_decay: float = field(default=0.0, metadata={"help": "The weight decay for the HF optimizer."})
    hf_lr_scheduler_type: str = field(default='linear', metadata={"help": "The lr scheduler type for the HF optimizer."})
    hf_warmup: int = field(default=0, metadata={"help": "The warmup steps for the HF optimizer."})

    # lora hyperparams
    peft_type: str = field(default=None, metadata={
        "help": "The type of PEFT, including [lora, prefix-tuning, prompt-tuning, p-tuning]."})
    lora_r: int = field(default=8, metadata={"help": "Lora attention dimension."})
    lora_alpha: int = field(default=16, metadata={"help": "The alpha parameter for Lora scaling."})
    lora_dropout: float = field(default=0.05, metadata={"help": "The dropout probability for Lora layers."})
    lora_only: bool = field(default=False, metadata={"help": "Whether to use LoRA without LOMO"})
