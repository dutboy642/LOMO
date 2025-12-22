import json
import random

import json
import random
import os

def sample_json_file(
    input_path: str,
    output_dir: str,
    num_samples: int = 100,
    seed: int = 42,
):
    random.seed(seed)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "JSON root phải là list"

    n = min(num_samples, len(data))
    sampled = random.sample(data, n)

    os.makedirs(output_dir, exist_ok=True)

    base, ext = os.path.splitext(os.path.basename(input_path))
    output_path = os.path.join(output_dir, f"{base}_{n}_samples{ext}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(f"Saved {n} samples → {output_path}")

# Ví dụ dùng
sample_json_file(
    "/home/thanhnguyen/code/Build_LLM/LOMO/lomo/data/wikipedia_ja.json",
    "/home/thanhnguyen/code/Build_LLM/LOMO/lomo/data",
    num_samples=10000
)
