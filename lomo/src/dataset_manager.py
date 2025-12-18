#!/usr/bin/env python3
"""
Comprehensive dataset manager for LOMO continued pretraining
Supports automatic download and verification of various datasets
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import requests
from tqdm import tqdm

class DatasetManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def check_dataset_exists(self, dataset_name: str) -> Dict[str, any]:
        """Check if dataset exists and return info"""
        dataset_map = {
            'wikipedia_ja': 'wikipedia_ja.json',
            'wikipedia_ja_100_samples': 'wikipedia_ja_100_samples.json',
            'wikipedia_vi': 'wikipedia_vi.json', 
            'wikipedia_en': 'wikipedia_en.json',
            'custom_text': 'custom_text.txt',
            'custom_jsonl': 'custom_data.jsonl',
            'vietnamese_corpus': 'vietnamese_corpus.txt'
        }
        
        if dataset_name not in dataset_map:
            return {'exists': False, 'error': f'Unknown dataset: {dataset_name}'}
        
        file_path = self.data_dir / dataset_map[dataset_name]
        
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            return {
                'exists': True,
                'path': str(file_path),
                'size_mb': round(size_mb, 1),
                'records': self._count_records(file_path)
            }
        else:
            return {'exists': False, 'path': str(file_path)}
    
    def _count_records(self, file_path: Path) -> Optional[int]:
        """Count number of records in dataset"""
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return len(data) if isinstance(data, list) else None
            elif file_path.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return sum(1 for line in f if line.strip())
            elif file_path.suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Count paragraphs (separated by double newlines)
                    return len([p for p in content.split('\n\n') if p.strip()])
        except Exception:
            return None
        return None
    
    def download_wikipedia(self, language: str, force: bool = False) -> bool:
        """Download Wikipedia dataset using the download script"""
        from download_wiki_dataset import download_wikipedia_dataset
        
        try:
            output_path = download_wikipedia_dataset(
                language=language,
                output_dir=str(self.data_dir),
                force_download=force
            )
            return os.path.exists(output_path)
        except Exception as e:
            print(f"❌ Error downloading Wikipedia {language}: {e}")
            return False
    
    def download_sample_vietnamese_corpus(self) -> bool:
        """Download a sample Vietnamese corpus"""
        url = "https://raw.githubusercontent.com/undertheseanlp/corpus/master/sample_data/vietnamese_text_sample.txt"
        output_path = self.data_dir / "vietnamese_corpus.txt"
        
        try:
            print("📥 Downloading sample Vietnamese corpus...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(output_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✅ Downloaded to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error downloading Vietnamese corpus: {e}")
            return False
    
    def create_sample_dataset(self, dataset_type: str) -> bool:
        """Create a sample dataset for testing"""
        if dataset_type == "custom_text":
            content = """Đây là một tập dữ liệu mẫu cho continued pretraining.
Continued pretraining là quá trình tiếp tục huấn luyện một mô hình ngôn ngữ lớn trên dữ liệu domain cụ thể.

Khác với fine-tuning, continued pretraining không cần format câu hỏi-trả lời.
Thay vào đó, mô hình học predict token tiếp theo từ raw text.

Điều này giúp mô hình học được:
- Từ vựng mới của domain
- Cấu trúc ngữ pháp đặc biệt
- Kiến thức chuyên môn
- Style viết của lĩnh vực

Ví dụ về continued pretraining:
- Train mô hình tiếng Anh trên corpus tiếng Việt
- Train mô hình general trên dữ liệu y khoa
- Train mô hình chat trên dữ liệu pháp lý

LOMO optimizer giúp train full parameters với memory thấp.
Đây là breakthrough quan trọng cho việc adapt LLMs.

Kết quả là mô hình được fine-tune cho domain cụ thể.
Performance sẽ tốt hơn đáng kể trên task domain đó."""
            
            output_path = self.data_dir / "custom_text.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Created sample text dataset: {output_path}")
            return True
        
        elif dataset_type == "custom_jsonl":
            samples = [
                {"text": "Mẫu câu tiếng Việt đầu tiên cho continued pretraining."},
                {"text": "Continued pretraining giúp mô hình học domain knowledge."},
                {"text": "LOMO optimizer tiết kiệm memory đáng kể khi training."},
                {"text": "Full parameter training với LOMO rất hiệu quả."},
                {"text": "Dữ liệu raw text không cần preprocessing phức tạp."}
            ]
            
            output_path = self.data_dir / "custom_data.jsonl"
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            print(f"✅ Created sample JSONL dataset: {output_path}")
            return True
        
        return False
    
    def list_datasets(self) -> List[Dict]:
        """List all available datasets with their status"""
        datasets = [
            'wikipedia_ja','wikipedia_ja_100_samples','wikipedia_ja', 'wikipedia_vi', 'wikipedia_en',
            'custom_text', 'custom_jsonl', 'vietnamese_corpus'
        ]
        
        result = []
        for dataset in datasets:
            info = self.check_dataset_exists(dataset)
            result.append({'name': dataset, **info})
        
        return result
    
    def auto_setup(self, dataset_name: str, **kwargs) -> bool:
        """Automatically setup a dataset"""
        info = self.check_dataset_exists(dataset_name)
        
        if info['exists'] and not kwargs.get('force', False):
            print(f"✅ Dataset '{dataset_name}' already exists")
            return True
        
        if dataset_name.startswith('wikipedia_'):
            language = dataset_name.split('_')[1]
            return self.download_wikipedia(language, kwargs.get('force', False))
        
        elif dataset_name == 'vietnamese_corpus':
            return self.download_sample_vietnamese_corpus()
        
        elif dataset_name in ['custom_text', 'custom_jsonl']:
            return self.create_sample_dataset(dataset_name)
        
        else:
            print(f"❌ Unknown dataset: {dataset_name}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Dataset manager for LOMO continued pretraining")
    parser.add_argument('--list', action='store_true', help='List all datasets and their status')
    parser.add_argument('--download', type=str, help='Download specific dataset')
    parser.add_argument('--language', type=str, default='ja', help='Language for Wikipedia dataset')
    parser.add_argument('--force', action='store_true', help='Force re-download even if exists')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--create-sample', type=str, choices=['custom_text', 'custom_jsonl'], 
                       help='Create sample dataset')
    
    args = parser.parse_args()
    
    manager = DatasetManager(args.data_dir)
    
    if args.list:
        print("📋 Dataset Status")
        print("=" * 50)
        datasets = manager.list_datasets()
        for dataset in datasets:
            status = "✅" if dataset['exists'] else "❌"
            print(f"{status} {dataset['name']:<20}", end="")
            if dataset['exists']:
                print(f" Size: {dataset['size_mb']}MB, Records: {dataset.get('records', 'N/A')}")
            else:
                print(" Not found")
    
    elif args.download:
        print(f"📥 Setting up dataset: {args.download}")
        success = manager.auto_setup(args.download, force=args.force, language=args.language)
        if success:
            print(f"✅ Dataset '{args.download}' is ready!")
        else:
            print(f"❌ Failed to setup dataset '{args.download}'")
    
    elif args.create_sample:
        print(f"🔨 Creating sample dataset: {args.create_sample}")
        success = manager.create_sample_dataset(args.create_sample)
        if success:
            print(f"✅ Sample dataset '{args.create_sample}' created!")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()