#!/bin/bash

set -x

# Set default GPU if not specified
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="0"
    echo "⚠️  CUDA_VISIBLE_DEVICES not set, defaulting to GPU 0"
fi

# Validate CUDA_VISIBLE_DEVICES format
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "❌ Invalid CUDA_VISIBLE_DEVICES format: $CUDA_VISIBLE_DEVICES"
    echo "   Expected format: 0 or 0,1 or 0,1,2,3"
    export CUDA_VISIBLE_DEVICES="0"
    echo "   Using default: GPU 0"
fi

# Generate random port for DeepSpeed
port=$(shuf -i25000-30000 -n1)

echo "==================================="
echo "LOMO Continued Pretraining Script"
echo "==================================="
echo "Port: $port"
echo "Config: config/args_continued_pretraining.yaml"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "==================================="

# Create data directory if it doesn't exist
mkdir -p data

# Check if dataset manager exists and use it
if [ -f "src/dataset_manager.py" ]; then
    echo "📋 Checking dataset status..."
    python src/dataset_manager.py --list
    echo ""
    
    # Extract dataset name from config
    DATASET_NAME=$(grep "dataset_name:" config/args_continued_pretraining.yaml | sed "s/.*: *'//" | sed "s/'.*//")
    echo "🎯 Target dataset: $DATASET_NAME"
    
    # Auto-download if dataset doesn't exist
    if ! python -c "
import sys
sys.path.append('src')
from dataset_manager import DatasetManager
manager = DatasetManager()
info = manager.check_dataset_exists('$DATASET_NAME')
exit(0 if info['exists'] else 1)
"; then
        echo "📥 Dataset not found. Auto-downloading..."
        python src/dataset_manager.py --download "$DATASET_NAME"
        
        if [ $? -ne 0 ]; then
            echo "❌ Failed to download dataset. Creating sample instead..."
            if [[ "$DATASET_NAME" == *"custom"* ]]; then
                python src/dataset_manager.py --create-sample custom_text
                # Update config to use sample
                sed -i "s/dataset_name: .*/dataset_name: 'custom_text'/" config/args_continued_pretraining.yaml
            else
                echo "❌ Cannot create sample for $DATASET_NAME. Please prepare dataset manually."
                exit 1
            fi
        fi
    else
        echo "✅ Dataset already exists!"
    fi
else
    echo "⚠️  Dataset manager not found. Creating basic sample..."
    # Create a basic sample if dataset manager is not available
    if [ ! -f "data/custom_text.txt" ]; then
        cat > data/custom_text.txt << 'EOF'
Đây là một văn bản mẫu để thực hiện continued pretraining.
Continued pretraining giúp mô hình học thêm kiến thức về một lĩnh vực cụ thể.

Quá trình này khác với fine-tuning ở chỗ:
- Không cần format question/answer
- Train trên raw text để học ngôn ngữ và kiến thức domain
- Thường dùng learning rate thấp hơn

Ví dụ về continued pretraining:
- Train mô hình tiếng Anh trên corpus tiếng Việt
- Train mô hình general trên dữ liệu y khoa
- Train mô hình trên dữ liệu pháp lý

Mỗi đoạn text được tokenize và model học predict token tiếp theo.
Điều này giúp model hiểu ngữ cảnh và ngữ pháp của domain mới.

Với LOMO optimizer, ta có thể train full parameters với memory thấp.
Đây là ưu điểm lớn so với các optimizer truyền thống như AdamW.

Kết quả cuối cùng là một model được adapt cho domain cụ thể.
Model này sẽ generate text tốt hơn cho domain đó.
EOF
        echo "📄 Sample text file created at data/custom_text.txt"
        sed -i "s/dataset_name: .*/dataset_name: 'custom_text'/" config/args_continued_pretraining.yaml
    fi
fi

echo ""
echo "🚀 Starting continued pretraining..."
echo "📊 Final dataset status:"
if [ -f "src/dataset_manager.py" ]; then
    python src/dataset_manager.py --list
fi

# Run continued pretraining
echo "🚀 Starting DeepSpeed training..."
echo "Command: deepspeed --master_port $port --include localhost:$CUDA_VISIBLE_DEVICES src/train_lomo_continued_pretraining.py config/args_continued_pretraining.yaml"
echo ""

# Check number of GPUs and adjust include parameter
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT=${#GPU_ARRAY[@]}

if [ $GPU_COUNT -eq 1 ]; then
    # Single GPU training
    deepspeed --master_port "$port" \
        --include localhost:${CUDA_VISIBLE_DEVICES} \
        src/train_lomo_continued_pretraining.py \
        config/args_continued_pretraining.yaml
else
    # Multi-GPU training
    deepspeed --master_port "$port" \
        --include localhost:${CUDA_VISIBLE_DEVICES} \
        src/train_lomo_continued_pretraining.py \
        config/args_continued_pretraining.yaml
fi

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "==================================="
    echo "✅ Continued pretraining completed successfully!"
    echo "📁 Check outputs/ directory for results"
    echo "==================================="
else
    echo "==================================="
    echo "❌ Training failed with exit code: $exit_code"
    echo "📝 Check logs above for error details"
    echo "==================================="
    exit $exit_code
fi