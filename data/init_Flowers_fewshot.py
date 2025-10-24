import os
import torch
import argparse
from PIL import Image
import sys
from tqdm import tqdm
import shutil
import yaml
import scipy.io as sio
import numpy as np

sys.path.append('..')
from utils import util

with open('../config.yml', 'r') as f:
    config = yaml.safe_load(f)

data_path = os.path.abspath(config['data_path2'])
# 根据您的路径调整
origin_path = os.path.join(data_path, 'text_flower')  # 包含jpg文件夹和setid.mat
jpg_path = os.path.join(origin_path, 'jpg')
raw_path = os.path.join(data_path, 'Flowers_fewshot_raw')
cropped_path = os.path.join(data_path, 'Flowers_fewshot_cropped')

util.mkdir(raw_path)
util.mkdir(cropped_path)

# 读取setid.mat文件获取数据集划分
mat_data = sio.loadmat(os.path.join(origin_path, 'setid.mat'))
train_ids = mat_data['trnid'][0]
val_ids = mat_data['valid'][0]
test_ids = mat_data['tstid'][0]

# 读取imagelabels.mat获取图像标签
labels_data = sio.loadmat(os.path.join(origin_path, 'imagelabels.mat'))
image_labels = labels_data['labels'][0]

# 创建图像ID到文件名的映射
image_files = [f for f in os.listdir(jpg_path) if f.endswith('.jpg')]
image_files.sort()  # 确保顺序一致

# 创建图像ID到类别标签的映射
id2label = {}
for i, img_file in enumerate(image_files):
    # 图像编号从1开始，对应labels中的索引
    img_id = int(img_file.split('_')[1].split('.')[0])
    id2label[img_id] = image_labels[img_id - 1]  # -1因为MATLAB索引从1开始

# 创建类别到图像的映射
cat2img = {}
for img_id, label in id2label.items():
    if label not in cat2img:
        cat2img[label] = []
    cat2img[label].append(img_id)

# 获取所有类别
all_categories = sorted(list(cat2img.keys()))
print(f"Total categories: {len(all_categories)}")

# 划分训练、验证、测试集
# 按照CUB和Aircraft的划分方式，按类别ID的奇偶性划分
train_cat = []
val_cat = []
test_cat = []

for cat_id in all_categories:
    if cat_id % 2 == 0:
        train_cat.append(cat_id)
    elif cat_id % 4 == 1:
        val_cat.append(cat_id)
    elif cat_id % 4 == 3:
        test_cat.append(cat_id)

print(f"Train categories: {len(train_cat)}")
print(f"Val categories: {len(val_cat)}")
print(f"Test categories: {len(test_cat)}")

# 创建目录结构
split = ['train', 'val', 'test']
split_cat = [train_cat, val_cat, test_cat]

print('Organizing Flowers_fewshot_raw ...')
for i in range(3):
    split_path = os.path.join(raw_path, split[i])
    util.mkdir(split_path)

    for cat_id in split_cat[i]:
        cat_dir = f"class_{cat_id:05d}"  # 格式化为5位数，前面补零
        util.mkdir(os.path.join(split_path, cat_dir))

    for cat_id in tqdm(split_cat[i]):
        cat_dir = f"class_{cat_id:05d}"
        for img_id in cat2img[cat_id]:
            img_file = f"image_{img_id:05d}.jpg"
            origin_img = os.path.join(jpg_path, img_file)
            target_img = os.path.join(split_path, cat_dir, img_file)

            if os.path.exists(origin_img):
                shutil.copy(origin_img, target_img)
            else:
                print(f"Warning: {origin_img} does not exist")

print('Getting pre-resized 84x84 images for validation and test')
util.get_pre_folder(image_folder=raw_path, transform_type=0)

print('Organizing Flowers_fewshot_cropped ...')
# 对于Flowers数据集，如果没有边界框信息，cropped版本将与raw版本相同
# 只是将格式从jpg转换为png

for i in range(3):
    split_path = os.path.join(cropped_path, split[i])
    util.mkdir(split_path)

    for cat_id in split_cat[i]:
        cat_dir = f"class_{cat_id:05d}"
        util.mkdir(os.path.join(split_path, cat_dir))

    for cat_id in tqdm(split_cat[i]):
        cat_dir = f"class_{cat_id:05d}"
        for img_id in cat2img[cat_id]:
            img_file = f"image_{img_id:05d}.jpg"
            origin_img = os.path.join(jpg_path, img_file)
            target_img = os.path.join(split_path, cat_dir, f"image_{img_id:05d}.png")

            if os.path.exists(origin_img):
                # 打开图像并保存为png格式
                image = Image.open(origin_img)
                image.save(target_img)
            else:
                print(f"Warning: {origin_img} does not exist")

print('Getting pre-resized 84x84 images for validation and test')
util.get_pre_folder(image_folder=cropped_path, transform_type=1)

print('Flowers few-shot dataset preparation completed!')