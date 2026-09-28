# Get data
# Copy a certain amount of valid MFTRCAD samples according to valid_label_samples.txt

import os

print(os.getcwd())

from utils.get_data_mftr import (
    clear_dataset_folders,
    copy_dataset_from_valid_list,
    check_dataset_integrity,
)

n = 1000

# Please download the dataset from:
# https://www.kaggle.com/datasets/xmy2000/mftrcad/data
src_root = "C:\\Users\\go36sal\\Downloads\\archive"

# Destination folder
dst_root = "data/mftrcad"

# This file should contain valid sample names, one per line:
# 20240116_231044_0_result
# 20240116_231044_3_result
# ...
valid_txt_path = "output/mftrcad/valid_label_samples.txt"

clear_dataset_folders(dst_root)

copy_dataset_from_valid_list(
    n=n,
    src_root=src_root,
    dst_root=dst_root,
    valid_txt_path=valid_txt_path,
)

check_dataset_integrity(dst_root, n)