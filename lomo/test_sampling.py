#!/usr/bin/env python3
"""Test script to verify sampling functionality"""

import sys
import os
sys.path.append('src')

from transformers import HfArgumentParser, AutoTokenizer
from arguments import ModelArguments, DataArguments, MyTrainingArguments
from mydatasets_continued_pretraining import get_continued_pretraining_dataset_info, ContinuedPretrainingDataset

def test_sampling():
    print("🧪 Testing dataset sampling functionality")
    print("=" * 50)
    
    # Setup arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
    
    # Mock arguments for testing
    model_args = ModelArguments(model_name_or_path='gpt2')  # Use smaller model for testing
    data_args = DataArguments(
        dataset_name='wikipedia_ja',
        sample_size=100,  # Test with small sample
        data_max_length=512,
        refresh=True,  # Force refresh to test
        data_tag='test_sampling'
    )
    
    print(f"📊 Configuration:")
    print(f"   Dataset: {data_args.dataset_name}")
    print(f"   Sample size: {data_args.sample_size}")
    print(f"   Max length: {data_args.data_max_length}")
    print(f"   Refresh: {data_args.refresh}")
    print()
    
    try:
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Get dataset info
        dataset_info = get_continued_pretraining_dataset_info(data_args.dataset_name)
        print(f"🔍 Dataset info:")
        print(f"   Path: {dataset_info.path}")
        print(f"   Text column: {dataset_info.text_column}")
        print(f"   Default sample size: {dataset_info.sample_size}")
        print()
        
        # Create dataset
        print("🚀 Creating dataset...")
        train_dataset = ContinuedPretrainingDataset(
            data_args, tokenizer, dataset_info, split='train'
        )
        
        print(f"✅ Dataset created successfully!")
        print(f"   Final dataset size: {len(train_dataset)}")
        print(f"   Expected size: {data_args.sample_size}")
        
        if len(train_dataset) == data_args.sample_size:
            print("✅ Sampling working correctly!")
        else:
            print(f"❌ Sampling not working. Expected {data_args.sample_size}, got {len(train_dataset)}")
        
        # Show a sample
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"\n📄 Sample data:")
            print(f"   Input length: {len(sample['input_ids'])}")
            print(f"   Text preview: {sample['text'][:100]}...")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sampling()