from datasets import load_dataset
import json
import os
from pathlib import Path

def download_wikipedia_dataset(language="ja", date="20231101", output_dir="data", force_download=False):
    """
    Download Wikipedia dataset for specified language if not already exists
    
    Args:
        language (str): Language code (e.g., 'ja', 'vi', 'en')
        date (str): Wikipedia dump date (e.g., '20231101')
        output_dir (str): Output directory
        force_download (bool): Force re-download even if file exists
    """
    # Create output directory if not exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"wikipedia_{language}.json")
    
    # Check if dataset already exists
    if os.path.exists(output_path) and not force_download:
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"✅ Dataset already exists: {output_path}")
        print(f"📁 File size: {file_size:.1f} MB")
        print("⏭️  Skipping download. Use force_download=True to re-download.")
        return output_path
    
    print(f"🌐 Downloading Wikipedia dataset: {language} ({date})")
    print(f"📁 Output path: {output_path}")
    
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset("wikimedia/wikipedia", f"{date}.{language}", split="train")
        print(f"📊 Dataset size: {len(dataset)} articles")
        
        # Save to JSON
        print("💾 Saving to JSON format...")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset.to_list(), f, ensure_ascii=False, indent=2)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"✅ Successfully saved to: {output_path}")
        print(f"📁 File size: {file_size:.1f} MB")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        # Clean up partial file if exists
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

def download_multiple_languages(languages=["ja", "vi", "en"], date="20231101", output_dir="data"):
    """Download Wikipedia datasets for multiple languages"""
    print(f"🌍 Downloading Wikipedia datasets for {len(languages)} languages...")
    
    results = {}
    for lang in languages:
        print(f"\n--- Processing {lang} ---")
        try:
            path = download_wikipedia_dataset(lang, date, output_dir)
            results[lang] = {"status": "success", "path": path}
        except Exception as e:
            print(f"❌ Failed to download {lang}: {e}")
            results[lang] = {"status": "failed", "error": str(e)}
    
    # Summary
    print("\n" + "="*50)
    print("📋 DOWNLOAD SUMMARY")
    print("="*50)
    successful = [lang for lang, result in results.items() if result["status"] == "success"]
    failed = [lang for lang, result in results.items() if result["status"] == "failed"]
    
    print(f"✅ Successful: {successful} ({len(successful)}/{len(languages)})")
    if failed:
        print(f"❌ Failed: {failed}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Wikipedia datasets")
    parser.add_argument("--language", "-l", default="ja", help="Language code (default: ja)")
    parser.add_argument("--date", "-d", default="20231101", help="Wikipedia dump date (default: 20231101)")
    parser.add_argument("--output-dir", "-o", default="data", help="Output directory (default: data)")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-download even if file exists")
    parser.add_argument("--multiple", "-m", nargs="+", help="Download multiple languages (e.g., --multiple ja vi en)")
    
    args = parser.parse_args()
    
    if args.multiple:
        download_multiple_languages(args.multiple, args.date, args.output_dir)
    else:
        download_wikipedia_dataset(args.language, args.date, args.output_dir, args.force)
