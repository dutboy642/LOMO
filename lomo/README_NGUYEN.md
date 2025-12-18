Đổi dataset
đổi trong file _modal đoạn .add_local_file
đổi trong check_dataset_exists , list_datasets của dataset_manager.py
đổi  dataset_name trong file yaml
đổi get_continued_pretraining_dataset_info của mydataset

# Yêu cầu rebuild images
MODAL_FORCE_BUILD=1 modal deploy LOMO/lomo/modal_deploy_image.py

# Có thể chạy file setup_dataset để tạo / kiểm tra data
chmod +x setup_dataset.sh

## Interactive mode
./setup_dataset.sh

## Command line
./setup_dataset.sh -d 'data/my_corpus.json' -t 'content' -s 5000