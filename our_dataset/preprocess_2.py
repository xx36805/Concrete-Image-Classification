import os
import random
import shutil


def move_random_files(src_dir, dst_dir, remaining_dir, ratio=0.2):
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(remaining_dir, exist_ok=True)

    files_list = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    move_to_num = int(len(files_list) * ratio)

    select_files_list = random.sample(files_list, move_to_num)
    remaining_files_list = list(set(files_list) - set(select_files_list))

    for file in select_files_list:
        shutil.copy(os.path.join(src_dir, file), os.path.join(dst_dir, file))

    for file in remaining_files_list:
        shutil.copy(os.path.join(src_dir, file), os.path.join(remaining_dir, file))



src_directory = "original"
dst_directory = "condef/test"
remaining_directory = "condef/train"

for folder in os.listdir(src_directory):
    folder_path = os.path.join(src_directory, folder)
    if os.path.isdir(folder_path):
        move_random_files(folder_path, os.path.join(dst_directory, folder), os.path.join(remaining_directory, folder))
