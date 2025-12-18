import os
import copy
import random
from tqdm import tqdm
from typing import Callable, Any

from datasets import load_dataset
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset

from log import print

IGNORE_INDEX = -100
REPRODUCIBILITY_SEED = 0


@dataclass
class DatasetInfo:
    path: str
    name: str = None
    exemplar_split: str = 'train'
    eval_split: str = 'validation'
    test_split: str = 'test'
    sample_size: int = -1
    text_column: str = 'text'  # Column name containing the text data


class ContinuedPretrainingDataset(Dataset):
    """Dataset for continued pretraining on raw text data"""
    
    def __init__(self, data_args, tokenizer, dataset_info, split):
        super().__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.split = split
        
        # Use sample_size from data_args if specified, otherwise use dataset_info
        if hasattr(data_args, 'sample_size') and data_args.sample_size != -1:
            self.sample_size = data_args.sample_size
            print(f"🎯 Using sample_size from config: {self.sample_size}")
        else:
            self.sample_size = dataset_info.sample_size
            print(f"📊 Using default sample_size: {self.sample_size}")
            
        self.text_column = dataset_info.text_column

        save_dir = os.path.join(data_args.data_dir, data_args.dataset_name, data_args.data_tag)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        save_file = os.path.join(save_dir, f'{split}.pt')
        if data_args.refresh or not os.path.exists(save_file):
            if dataset_info.path.endswith('.jsonl') or dataset_info.path.endswith('.json'):
                # Load from local JSON/JSONL file
                full_dataset = load_dataset('json', data_files=dataset_info.path, split='train')
                dataset = self._split_local_dataset(full_dataset, split)
            elif dataset_info.path.endswith('.txt'):
                # Load from text file
                with open(dataset_info.path, 'r', encoding='utf-8') as f:
                    texts = f.read().strip().split('\n\n')  # Split by double newline
                full_dataset = [{'text': text.strip()} for text in texts if text.strip()]
                dataset = self._split_local_dataset(full_dataset, split)
            else:
                # Load from HuggingFace datasets
                dataset = load_dataset(dataset_info.path, name=dataset_info.name, split=split)
            
            self.data = self.process_continued_pretraining(dataset, save_file)
        else:
            print('Loading data from', save_file)
            self.data = torch.load(save_file)
        
        print('Data size:', len(self.data))
        print('Data format:', self.data[0].keys() if self.data else "Empty dataset")
        if self.data:
            print('Max length:', max([len(d['input_ids']) for d in self.data]))

    def _split_local_dataset(self, full_dataset, split, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """Chia local dataset thành train/validation/test splits"""
        if isinstance(full_dataset, list):
            total_size = len(full_dataset)
        else:
            total_size = len(full_dataset)
            
        # Tính toán size cho mỗi split
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        # Shuffle data với seed cố định
        random.seed(REPRODUCIBILITY_SEED)
        indices = list(range(total_size))
        random.shuffle(indices)
        
        # Chia indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        print(f"📊 Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        # Trả về data theo split được yêu cầu
        if split == 'train':
            if isinstance(full_dataset, list):
                return [full_dataset[i] for i in train_indices]
            else:
                return full_dataset.select(train_indices)
        elif split == 'validation':
            if isinstance(full_dataset, list):
                return [full_dataset[i] for i in val_indices]
            else:
                return full_dataset.select(val_indices)
        elif split == 'test':
            if isinstance(full_dataset, list):
                return [full_dataset[i] for i in test_indices]
            else:
                return full_dataset.select(test_indices)
        else:
            raise ValueError(f"Unknown split: {split}")

    def process_continued_pretraining(self, dataset, save_file):
        """Process raw text data for continued pretraining"""
        data = []
        
        # Sample dataset early if sample_size is specified
        if self.sample_size > 0 and len(dataset) > self.sample_size:
            print(f'🎯 Sampling {self.sample_size} examples from {len(dataset)} total examples')
            random.seed(REPRODUCIBILITY_SEED)
            # Convert to list if it's not already, then sample
            if hasattr(dataset, 'select'):
                # For HuggingFace datasets
                indices = random.sample(range(len(dataset)), self.sample_size)
                dataset = dataset.select(indices)
            else:
                # For list datasets
                dataset = random.sample(list(dataset), self.sample_size)
            print(f'✅ Sampled dataset size: {len(dataset)}')
        else:
            print(f'📊 Processing full dataset with {len(dataset)} examples')
        
        for instance in tqdm(dataset, desc="Processing dataset"):
            if isinstance(instance, dict):
                text = instance[self.text_column]
            else:
                text = instance  # If it's already a string
            
            if not text or len(text.strip()) == 0:
                continue
                
            # Tokenize the entire text
            tokenized = self.tokenizer.encode(
                text.strip(), 
                truncation=True, 
                max_length=self.data_args.data_max_length
            )
            
            # Add EOS token
            if tokenized[-1] != self.tokenizer.eos_token_id:
                tokenized = tokenized + [self.tokenizer.eos_token_id]
            
            # For continued pretraining, input_ids and labels are the same
            # The model learns to predict the next token for the entire sequence
            input_ids = tokenized
            labels = copy.deepcopy(input_ids)
            
            # Optionally mask the first token (since there's no previous context)
            if not self.data_args.train_on_inputs and len(labels) > 1:
                labels[0] = IGNORE_INDEX
            
            data.append({
                'input_ids': input_ids,
                'labels': labels,
                'text': text.strip()
            })

        print(f'✅ Processed {len(data)} examples successfully')
        torch.save(data, save_file)
        print('Saving data to', save_file)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def get_continued_pretraining_dataset_info(dataset_name):
    """Get dataset configuration for continued pretraining"""
    
    if dataset_name == 'custom_text':
        return DatasetInfo(
            path="data/custom_text.txt",  # Path to your text file
            exemplar_split="train",
            eval_split="validation",
            sample_size=-1,
            text_column='text'
        )
    elif dataset_name == 'custom_jsonl':
        return DatasetInfo(
            path="data/custom_data.jsonl",  # Path to your JSONL file
            exemplar_split="train", 
            eval_split="validation",
            sample_size=-1,
            text_column='content'  # Column name in your JSONL
        )
    elif dataset_name == 'vietnamese_corpus':
        return DatasetInfo(
            path="data/vietnamese_corpus.txt",
            exemplar_split="train",
            eval_split="validation", 
            sample_size=-1,
            text_column='text'
        )
    elif dataset_name == 'wikipedia_ja':
        return DatasetInfo(
            path="data/wikipedia_ja.json",
            exemplar_split="train",
            eval_split="validation",
            sample_size=10000,
            text_column='text'
        )
    elif dataset_name == 'wikipedia_ja_100_samples':
        return DatasetInfo(
            path="data/wikipedia_ja_100_samples.json",
            exemplar_split="train",
            eval_split="validation",
            sample_size=100,
            text_column='text'
        )
    elif dataset_name == 'wikipedia_vi':
        return DatasetInfo(
            path="data/wikipedia_vi.json",
            exemplar_split="train",
            eval_split="validation",
            sample_size=-1,
            text_column='text'
        )
    elif dataset_name == 'wikipedia_en':
        return DatasetInfo(
            path="data/wikipedia_en.json",
            exemplar_split="train",
            eval_split="validation",
            sample_size=-1,
            text_column='text'
        )
    elif dataset_name == 'wikitext':
        return DatasetInfo(
            path="wikitext",
            name="wikitext-103-raw-v1",
            exemplar_split="train",
            eval_split="validation",
            sample_size=-1,
            text_column='text'
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")


if __name__ == '__main__':
    from transformers import HfArgumentParser, AutoTokenizer
    from arguments import ModelArguments, DataArguments

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    
    # Example usage
    model_args.model_name_or_path = 'huggyllama/llama-7b'
    data_args.dataset_name = 'custom_text'
    data_args.refresh = True
    data_args.data_tag = 'continued_pretraining'
    data_args.data_max_length = 1024
    data_args.train_on_inputs = True  # For continued pretraining, usually True

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        padding_side='left'
    )
    tokenizer.pad_token_id = 0

    dataset_info = get_continued_pretraining_dataset_info(data_args.dataset_name)
    train_dataset = ContinuedPretrainingDataset(data_args, tokenizer, dataset_info, split='train')