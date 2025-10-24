import os
import torch
import argparse
from PIL import Image
import sys
from tqdm import tqdm
import shutil
import yaml
import random

sys.path.append('..')
from utils import util

# 设置随机种子以保证可重复性
random.seed(42)
torch.manual_seed(42)

with open('../config.yml', 'r') as f:
    config = yaml.safe_load(f)

data_path = os.path.abspath(config['data_path1'])
# 根据您的路径调整
origin_path = os.path.join(data_path, 'mini-imagenet')
train_path = os.path.join(origin_path, 'train')
val_path = os.path.join(origin_path, 'val')  # 如果有val目录
test_path = os.path.join(origin_path, 'test')  # 如果有test目录

raw_path = os.path.join(data_path, 'miniImagenet_fewshot_raw')
cropped_path = os.path.join(data_path, 'miniImagenet_fewshot_cropped')

util.mkdir(raw_path)
util.mkdir(cropped_path)


# 获取所有类别
def get_categories_from_folder(folder_path):
    """从文件夹获取类别列表"""
    if os.path.exists(folder_path):
        categories = [d for d in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, d))]
        return sorted(categories)
    return []


# 获取训练、验证、测试集的类别
train_categories = get_categories_from_folder(train_path)
val_categories = get_categories_from_folder(val_path) if os.path.exists(val_path) else []
test_categories = get_categories_from_folder(test_path) if os.path.exists(test_path) else []

# 如果没有单独的val和test目录，则从训练集中划分
if not val_categories and not test_categories:
    print("No separate val/test folders found, splitting from train categories...")
    random.shuffle(train_categories)

    # 按照64-16-20的比例划分（miniImageNet标准划分）
    num_train = 64
    num_val = 16
    num_test = 20

    val_categories = train_categories[num_train:num_train + num_val]
    test_categories = train_categories[num_train + num_val:num_train + num_val + num_test]
    train_categories = train_categories[:num_train]

print(f"Train categories: {len(train_categories)}")
print(f"Val categories: {len(val_categories)}")
print(f"Test categories: {len(test_categories)}")

# 创建类别ID映射
cat_name2id = {}
cat_id2name = {}

for idx, cat_name in enumerate(train_categories + val_categories + test_categories):
    if cat_name not in cat_name2id:
        cat_id = len(cat_name2id)
        cat_name2id[cat_name] = cat_id
        cat_id2name[cat_id] = cat_name


# 构建类别到图像的映射
def build_cat2img_mapping(categories, base_path):
    """构建类别到图像的映射"""
    cat2img = {}
    for cat_name in categories:
        cat_id = cat_name2id[cat_name]
        cat_dir = os.path.join(base_path, cat_name)

        if os.path.exists(cat_dir):
            image_files = [f for f in os.listdir(cat_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            cat2img[cat_id] = []

            for img_file in image_files:
                # 使用文件名（不含扩展名）作为图像ID
                img_id = os.path.splitext(img_file)[0]
                cat2img[cat_id].append({
                    'file_name': img_file,
                    'path': os.path.join(cat_dir, img_file)
                })

    return cat2img


# 构建各数据集的映射
train_cat2img = build_cat2img_mapping(train_categories, train_path)
val_cat2img = build_cat2img_mapping(val_categories, val_path if val_path else train_path)
test_cat2img = build_cat2img_mapping(test_categories, test_path if test_path else train_path)

# 创建目录结构
split = ['train', 'val', 'test']
split_categories = [train_categories, val_categories, test_categories]
split_cat2img = [train_cat2img, val_cat2img, test_cat2img]

print('Organizing miniImagenet_fewshot_raw ...')
for i in range(3):
    split_name = split[i]
    split_path = os.path.join(raw_path, split_name)
    util.mkdir(split_path)

    categories = split_categories[i]
    cat2img = split_cat2img[i]

    for cat_name in categories:
        util.mkdir(os.path.join(split_path, cat_name))

    for cat_name in tqdm(categories, desc=f'Processing {split_name} raw'):
        cat_id = cat_name2id[cat_name]
        if cat_id in cat2img:
            for img_info in cat2img[cat_id]:
                origin_img = img_info['path']
                target_img = os.path.join(split_path, cat_name, img_info['file_name'])

                if os.path.exists(origin_img):
                    shutil.copy(origin_img, target_img)
                else:
                    print(f"Warning: {origin_img} does not exist")

print('Getting pre-resized 84x84 images for validation and test')
util.get_pre_folder(image_folder=raw_path, transform_type=0)

print('Organizing miniImagenet_fewshot_cropped ...')
# 对于miniImageNet，如果没有边界框信息，cropped版本将与raw版本相同
# 只是将格式从jpg转换为png

for i in range(3):
    split_name = split[i]
    split_path = os.path.join(cropped_path, split_name)
    util.mkdir(split_path)

    categories = split_categories[i]
    cat2img = split_cat2img[i]

    for cat_name in categories:
        util.mkdir(os.path.join(split_path, cat_name))

    for cat_name in tqdm(categories, desc=f'Processing {split_name} cropped'):
        cat_id = cat_name2id[cat_name]
        if cat_id in cat2img:
            for img_info in cat2img[cat_id]:
                origin_img = img_info['path']
                # 转换为png格式
                target_img_name = os.path.splitext(img_info['file_name'])[0] + '.png'
                target_img = os.path.join(split_path, cat_name, target_img_name)

                if os.path.exists(origin_img):
                    # 打开图像并保存为png格式
                    image = Image.open(origin_img)
                    image.save(target_img)
                else:
                    print(f"Warning: {origin_img} does not exist")

print('Getting pre-resized 84x84 images for validation and test')
util.get_pre_folder(image_folder=cropped_path, transform_type=1)

print('miniImageNet few-shot dataset preparation completed!')
print(f"Raw data saved to: {raw_path}")
print(f"Cropped data saved to: {cropped_path}")

# 打印统计信息
print("\nDataset Statistics:")
for i in range(3):
    split_name = split[i]
    categories = split_categories[i]
    cat2img = split_cat2img[i]

    total_images = 0
    for cat_id in cat2img:
        total_images += len(cat2img[cat_id])

    print(f"{split_name}: {len(categories)} categories, {total_images} images")