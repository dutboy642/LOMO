from modal import (
    Image,
    App, 
    Volume,
    Secret
)


NUM_GPUS = 1
MASTER_PORT = "10000"
CUDA_VISIBLE_DEVICES = f'localhost:{",".join([str(i) for i in list(range(NUM_GPUS))])}'

app = App("LOMO")

image = (
    # Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    Image.from_registry("nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04", add_python="3.10")
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(
        "transformers",
        "torch",
        "tensorboard",
        "tqdm",
        # "deepspeed",
        "rich>=13.3.5",
        "accelerate>=0.20.3",
        "datasets",
        "huggingface_hub",
        "peft",
        "wandb",
        "numpy<2.0.0",
        "deepspeed==0.10.0",
    )
    .env({
        "PORT": MASTER_PORT,
        # "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES
    })
    .add_local_file("src/arguments.py", "/root/src/arguments.py")
    .add_local_file("src/dataset_manager.py", "/root/src/dataset_manager.py")
    .add_local_file("src/download_wiki_dataset.py", "/root/src/download_wiki_dataset.py")
    .add_local_file("src/mydatasets_continued_pretraining.py", "/root/src/mydatasets_continued_pretraining.py")
    .add_local_file("src/lomo.py", "/root/src/lomo.py")
    .add_local_file("src/lomo_trainer.py", "/root/src/lomo_trainer.py")
    .add_local_file("src/train_lomo_continued_pretraining.py", "/root/src/train_lomo_continued_pretraining.py")
    .add_local_file("src/prompts.py", "/root/src/prompts.py")
    .add_local_dir("log", "/root/log")
    .add_local_dir("config", "/root/config")
    .add_local_file("src/utils.py", "/root/src/utils.py")
    .add_local_file("data/wikipedia_ja_100_samples.json", "/root/data/wikipedia_ja_100_samples.json")
    .add_local_file("run_continued_pretraining.sh", "/root/run_continued_pretraining.sh")

)

volumes = {
    "/mnt/lomo": Volume.from_name(name="LOMO", create_if_missing=True)
}

secrets = [
    Secret.from_name("huggingface-token")
]


@app.function(
    image=image, 
    volumes=volumes, 
    secrets=secrets, 
    # gpu=f"A100-40GB:{NUM_GPUS}", 
    gpu=f"A10:{NUM_GPUS}", 
    timeout=3600*12
)
def main():
    import os
    os.system(
        "bash "
        "run_continued_pretraining.sh"
    )
