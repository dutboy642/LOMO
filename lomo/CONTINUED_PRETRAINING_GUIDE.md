# Hướng dẫn Continued Pretraining với LOMO

## 1. Khái niệm Continued Pretraining

**Continued Pretraining** (hay Domain Adaptation) khác hoàn toàn với instruction tuning:

- **Instruction Tuning**: Train theo format input→output, Q&A
- **Continued Pretraining**: Train trên raw text, học ngôn ngữ và domain knowledge

## 2. Cấu trúc Data cho Continued Pretraining

### Format 1: Text file (.txt)
```
data/custom_text.txt

Đây là đoạn văn bản đầu tiên.
Model sẽ học predict token tiếp theo trong đoạn này.

Đây là đoạn văn bản thứ hai.
Mỗi đoạn được phân cách bởi dòng trống.

Continued pretraining giúp model học:
- Ngữ pháp và cú pháp của ngôn ngữ mới
- Kiến thức domain cụ thể
- Style viết của lĩnh vực
```

### Format 2: JSONL file (.jsonl)
```json
{"text": "Đoạn văn bản thứ nhất cho continued pretraining..."}
{"text": "Đoạn văn bản thứ hai với nội dung domain cụ thể..."}
{"text": "Mỗi dòng là một JSON object với field 'text'..."}
```

### Format 3: JSON file (.json)
```json
[
    {"content": "Nội dung văn bản 1..."},
    {"content": "Nội dung văn bản 2..."},
    {"content": "Field name có thể custom..."}
]
```

## 3. Cấu hình Dataset Custom

Chỉnh sửa file `src/mydatasets_continued_pretraining.py`:

```python
def get_continued_pretraining_dataset_info(dataset_name):
    if dataset_name == 'your_dataset_name':
        return DatasetInfo(
            path="data/your_data.txt",  # Path tới file data
            exemplar_split="train",
            eval_split="validation", 
            sample_size=-1,  # -1 = dùng toàn bộ data
            text_column='text'  # Tên column chứa text
        )
```

## 4. Các loại Dataset được hỗ trợ

### A. Vietnamese Corpus
```python
elif dataset_name == 'vietnamese_corpus':
    return DatasetInfo(
        path="data/vietnamese_corpus.txt",
        text_column='text'
    )
```

### B. Medical Domain
```python
elif dataset_name == 'medical_texts':
    return DatasetInfo(
        path="data/medical_corpus.jsonl",
        text_column='content'
    )
```

### C. Legal Domain  
```python
elif dataset_name == 'legal_documents':
    return DatasetInfo(
        path="data/legal_corpus.json",
        text_column='document_text'
    )
```

## 5. Chạy Continued Pretraining

### Bước 1: Chuẩn bị data
```bash
mkdir -p data
# Copy file data của bạn vào thư mục data/
cp your_corpus.txt data/custom_text.txt
```

### Bước 2: Chỉnh sửa config
Chỉnh sửa `config/args_continued_pretraining.yaml`:

```yaml
# Data
dataset_name: 'custom_text'  # Tên dataset trong mydatasets_continued_pretraining.py
data_max_length: 2048  # Độ dài sequence (giảm nếu OOM)
train_on_inputs: true  # Luôn true cho continued pretraining

# Training  
learning_rate: 1e-5  # LR thấp cho continued pretraining
num_train_epochs: 3  # Ít epoch hơn
per_device_train_batch_size: 4  # Giảm nếu OOM
```

### Bước 3: Chạy training
```bash
# Chạy với GPU cụ thể
CUDA_VISIBLE_DEVICES=0 bash run_continued_pretraining.sh

# Chạy với nhiều GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_continued_pretraining.sh
```

## 6. Monitoring và Evaluation

### Wandb Tracking
```yaml
report_to: 'wandb'
logging_steps: 100
eval_steps: 500
```

### Metrics quan trọng:
- **Loss**: Giảm dần theo steps
- **Perplexity**: Càng thấp càng tốt
- **Learning Rate**: Theo scheduler

## 7. Tips Optimization

### Memory Management
```yaml
per_device_train_batch_size: 2  # Giảm nếu OOM
gradient_accumulation_steps: 8  # Tăng để maintain effective batch size
gradient_checkpointing: true    # Luôn bật
data_max_length: 1024          # Giảm sequence length
```

### Learning Rate
```yaml
# Domain adaptation (tiếng Việt, chuyên ngành)
learning_rate: 1e-5

# Language adaptation (ngôn ngữ hoàn toàn mới)  
learning_rate: 5e-6

# Style adaptation
learning_rate: 2e-5
```

### Training Duration
```yaml
# Light adaptation
num_train_epochs: 1-2

# Medium adaptation  
num_train_epochs: 3-5

# Heavy adaptation
num_train_epochs: 5-10
```

## 8. Ví dụ thực tế

### A. Vietnamese Adaptation
```bash
# 1. Chuẩn bị corpus tiếng Việt
cat vietnamese_wiki.txt vietnamese_news.txt > data/vietnamese_corpus.txt

# 2. Config
dataset_name: 'vietnamese_corpus'
learning_rate: 1e-5
num_train_epochs: 3

# 3. Run
bash run_continued_pretraining.sh
```

### B. Medical Domain
```bash
# 1. Chuẩn bị medical texts
python prepare_medical_corpus.py --output data/medical_corpus.jsonl

# 2. Config  
dataset_name: 'medical_texts'
learning_rate: 5e-6
num_train_epochs: 5

# 3. Run
bash run_continued_pretraining.sh
```

## 9. So sánh với Fine-tuning

| Aspect | Continued Pretraining | Fine-tuning |
|--------|----------------------|-------------|
| **Data Format** | Raw text | Input-Output pairs |
| **Learning Rate** | Rất thấp (1e-5) | Cao hơn (1e-4, 3e-4) |
| **Epochs** | Ít (1-5) | Nhiều hơn (10-100) |
| **Mục tiêu** | Học domain/ngôn ngữ | Học task cụ thể |
| **train_on_inputs** | `true` | `false` |

## 10. Troubleshooting

### OOM (Out of Memory)
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16  
data_max_length: 512
```

### Loss không giảm
- Kiểm tra learning rate (có thể quá cao)
- Kiểm tra data quality
- Kiểm tra `train_on_inputs: true`

### Training chậm
```yaml
dataloader_num_workers: 4
group_by_length: true
gradient_checkpointing: false  # Nếu memory đủ
```