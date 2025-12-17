#!/usr/bin/env python3
"""Simple test for sampling logic"""

import json
import random

# Simulate reading a large dataset
def test_sampling_logic():
    print("🧪 Testing sampling logic")
    
    # Create mock dataset
    mock_dataset = [{"text": f"Sample text {i}"} for i in range(100000)]
    print(f"📊 Mock dataset size: {len(mock_dataset)}")
    
    # Test sampling
    sample_size = 10000
    
    if sample_size > 0 and len(mock_dataset) > sample_size:
        print(f"🎯 Sampling {sample_size} examples from {len(mock_dataset)} total")
        random.seed(42)
        sampled = random.sample(mock_dataset, sample_size)
        print(f"✅ Sampled size: {len(sampled)}")
        
        if len(sampled) == sample_size:
            print("✅ Sampling logic works correctly!")
        else:
            print("❌ Sampling logic failed!")
    
    # Test with real wikipedia data if exists
    try:
        with open("data/wikipedia_ja.json", "r", encoding="utf-8") as f:
            real_data = json.load(f)
        
        print(f"\n📚 Real Wikipedia dataset size: {len(real_data)}")
        
        if sample_size > 0 and len(real_data) > sample_size:
            print(f"🎯 Would sample {sample_size} from {len(real_data)} articles")
            sampled_real = random.sample(real_data, sample_size)
            print(f"✅ Real sampling would produce: {len(sampled_real)} examples")
        
    except FileNotFoundError:
        print("📄 No real Wikipedia data found")

if __name__ == "__main__":
    test_sampling_logic()