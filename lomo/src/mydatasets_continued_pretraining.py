import os
import copy
import random
from tqdm import tqdm
from typing import Callable, Any
import json

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    print("Warning: huggingface_hub not available, using local files only")

import numpy as np
import torch
from torch.utils.data import Dataset

from log import print

IGNORE_INDEX = -100
REPRODUCIBILITY_SEED = 0


class ContinuedPretrainingDataset(Dataset):
    """Dataset for continued pretraining on raw text data"""
    
    def __init__(self, data_args, tokenizer, split):
        super().__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.split = split
        
        # Get config from data_args
        self.dataset_path = data_args.dataset_path
        self.text_column = data_args.text_column
        self.sample_size = data_args.sample_size
        
        print(f"📊 Dataset config: path={self.dataset_path}, text_column={self.text_column}, sample_size={self.sample_size}")

        save_dir = os.path.join(data_args.data_dir, data_args.dataset_name, data_args.data_tag)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        save_file = os.path.join(save_dir, f'{split}.pt')
        if data_args.refresh or not os.path.exists(save_file):
            dataset = self._load_dataset_from_path(self.dataset_path, split)
            self.data = self.process_continued_pretraining(dataset, save_file)
        else:
            print('Loading data from', save_file)
            self.data = torch.load(save_file)
        
        print('Data size:', len(self.data))
        print('Data format:', self.data[0].keys() if self.data else "Empty dataset")
        if self.data:
            print('Max length:', max([len(d['input_ids']) for d in self.data]))

    def _load_dataset_from_path(self, dataset_path, split):
        """Load dataset from various sources using huggingface_hub or local files"""
        
        if os.path.isfile(dataset_path):
            # Local file
            print(f"📁 Loading local file: {dataset_path}")
            full_dataset = self._load_local_file(dataset_path)
            return self._split_local_dataset(full_dataset, split)
        
        elif '/' in dataset_path and not dataset_path.startswith('/'):
            # HuggingFace repository
            print(f"🤗 Loading from HuggingFace: {dataset_path}")
            if HAS_HF_HUB:
                full_dataset = self._load_huggingface_dataset(dataset_path)
                return self._split_local_dataset(full_dataset, split)
            else:
                raise ImportError("huggingface_hub is required for loading HuggingFace datasets")
        
        else:
            raise ValueError(f"Invalid dataset path: {dataset_path}")
    
    def _load_local_file(self, file_path):
        """Load data from local file"""
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    return [data]
        
        elif file_path.endswith('.jsonl'):
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line.strip()))
            return data
        
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = f.read().strip().split('\n\n')  # Split by double newline
            return [{self.text_column: text.strip()} for text in texts if text.strip()]
        
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def _load_huggingface_dataset(self, repo_id):
        """Load dataset from HuggingFace using huggingface_hub"""
        try:
            # Try to download the repository
            repo_path = snapshot_download(repo_id, repo_type="dataset")
            
            # Look for common dataset files
            for filename in ['train.json', 'data.json', 'dataset.json']:
                file_path = os.path.join(repo_path, filename)
                if os.path.exists(file_path):
                    print(f"📄 Found dataset file: {filename}")
                    return self._load_local_file(file_path)
            
            # Look for jsonl files
            for filename in ['train.jsonl', 'data.jsonl', 'dataset.jsonl']:
                file_path = os.path.join(repo_path, filename)
                if os.path.exists(file_path):
                    print(f"📄 Found dataset file: {filename}")
                    return self._load_local_file(file_path)
            
            raise FileNotFoundError(f"No suitable dataset file found in {repo_id}")
        
        except Exception as e:
            print(f"❌ Error loading from HuggingFace: {e}")
            raise

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

    # def process_continued_pretraining(self, dataset, save_file):
    #     """Process raw text data for continued pretraining"""
    #     data = []
        
    #     # Sample dataset early if sample_size is specified
    #     if self.sample_size > 0 and len(dataset) > self.sample_size:
    #         print(f'🎯 Sampling {self.sample_size} examples from {len(dataset)} total examples')
    #         random.seed(REPRODUCIBILITY_SEED)
    #         # Convert to list if it's not already, then sample
    #         if hasattr(dataset, 'select'):
    #             # For HuggingFace datasets
    #             indices = random.sample(range(len(dataset)), self.sample_size)
    #             dataset = dataset.select(indices)
    #         else:
    #             # For list datasets
    #             dataset = random.sample(list(dataset), self.sample_size)
    #         print(f'✅ Sampled dataset size: {len(dataset)}')
    #     else:
    #         print(f'📊 Processing full dataset with {len(dataset)} examples')
        
    #     for instance in tqdm(dataset, desc="Processing dataset"):
    #         if isinstance(instance, dict):
    #             text = instance[self.text_column]
    #         else:
    #             text = instance  # If it's already a string
            
    #         if not text or len(text.strip()) == 0:
    #             continue
    #             # DEBUG: Check what we're returning
    #         print(f"  text_column: {self.text_column}")
    #         print(f"  example[text_column] type: {type(instance[self.text_column])}")
    #         print(f"  example[text_column][:100]: {instance[self.text_column][:100]}")
            
    #         # Tokenize the entire text
    #         tokenized = self.tokenizer.encode(
    #             text.strip(), 
    #             truncation=True, 
    #             max_length=self.data_args.data_max_length
    #         )
    #         # print(f"  tokenized keys: {tokenized.keys()}")
    #         # print(f"  input_ids type: {type(tokenized['input_ids'])}")
    #         # print(f"  input_ids length: {len(tokenized['input_ids'])}")
    #         # print(f"  input_ids[:10]: {tokenized['input_ids'][:10]}")
        
    #         # Add EOS token
    #         if tokenized[-1] != self.tokenizer.eos_token_id:
    #             tokenized = tokenized + [self.tokenizer.eos_token_id]
            
    #         # For continued pretraining, input_ids and labels are the same
    #         # The model learns to predict the next token for the entire sequence
    #         input_ids = tokenized
    #         labels = copy.deepcopy(input_ids)
            
    #         # Optionally mask the first token (since there's no previous context)
    #         if not self.data_args.train_on_inputs and len(labels) > 1:
    #             labels[0] = IGNORE_INDEX
            
    #         data.append({
    #             'input_ids': input_ids,
    #             'labels': labels,
    #             'text': text.strip()
    #         })

    #     print(f'✅ Processed {len(data)} examples successfully')
    #     torch.save(data, save_file)
    #     print('Saving data to', save_file)
    #     return data


    def process_continued_pretraining(self, dataset, save_file):
        data = []

        if self.sample_size > 0 and len(dataset) > self.sample_size:
            random.seed(REPRODUCIBILITY_SEED)
            if hasattr(dataset, "select"):
                indices = random.sample(range(len(dataset)), self.sample_size)
                dataset = dataset.select(indices)
            else:
                dataset = random.sample(list(dataset), self.sample_size)

        for instance in tqdm(dataset, desc="Processing dataset"):
            if isinstance(instance, dict):
                text = instance[self.text_column]
            else:
                text = instance

            if not text or len(text.strip()) == 0:
                continue

            # ⚠️ DÙNG tokenizer(), KHÔNG DÙNG encode()
            tokenized = self.tokenizer(
                text.strip(),
                truncation=True,
                max_length=self.data_args.data_max_length,
                padding=False,          # để collator pad
                return_attention_mask=False
            )

            input_ids = tokenized["input_ids"]

            # Add EOS nếu cần
            if input_ids[-1] != self.tokenizer.eos_token_id:
                input_ids = input_ids + [self.tokenizer.eos_token_id]

            data.append({
                "input_ids": input_ids
            })

        torch.save(data, save_file)
        print(f"✅ Saved {len(data)} examples to {save_file}")
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


# Dataset configuration is now handled via YAML - no hardcoding needed!

# if __name__ == '__main__':
#     from transformers import HfArgumentParser, AutoTokenizer
#     from arguments import ModelArguments, DataArguments

#     parser = HfArgumentParser((ModelArguments, DataArguments))
#     model_args, data_args = parser.parse_args_into_dataclasses()
    
#     # Example usage - configure via args instead of hardcoded dataset_info
#     model_args.model_name_or_path = 'huggyllama/llama-7b'
#     data_args.dataset_path = 'data/wikipedia_ja_100_samples.json'
#     data_args.text_column = 'text'
#     data_args.sample_size = 100
#     data_args.refresh = True
#     data_args.data_tag = 'continued_pretraining'
#     data_args.data_max_length = 1024
#     data_args.train_on_inputs = True

#     tokenizer = AutoTokenizer.from_pretrained(
#         model_args.model_name_or_path,
#         use_fast=False,
#         padding_side='left'
#     )
#     tokenizer.pad_token_id = 0

#     train_dataset = ContinuedPretrainingDataset(data_args, tokenizer, split='train')