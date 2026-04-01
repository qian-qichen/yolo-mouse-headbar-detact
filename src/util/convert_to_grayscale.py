#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
递归查找指定目录下的所有图片，并将其原地转化为灰度图像
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm


def convert_images_to_grayscale(root_dir, image_extensions=None):
    """
    递归查找指定目录下的所有图片，并将其原地转化为灰度图像
    
    Args:
        root_dir (str): 根目录路径
        image_extensions (list): 支持的图片扩展名列表，默认为 ['.jpg', '.jpeg', '.png', '.bmp']
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # 转换为小写以便比较
    image_extensions = [ext.lower() for ext in image_extensions]
    
    # 获取所有图片文件
    image_files = []
    root_path = Path(root_dir)
    
    for ext in image_extensions:
        image_files.extend(root_path.rglob(f"*{ext}"))
        image_files.extend(root_path.rglob(f"*{ext.upper()}"))
    
    # 去重并排序
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"在目录 {root_dir} 下未找到任何图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件，开始转换为灰度图像...")
    
    # 使用tqdm显示进度
    converted_count = 0
    for img_path in tqdm(image_files, desc="转换进度"):
        try:
            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"警告: 无法读取文件 {img_path}，跳过...")
                continue
            
            # 如果已经是灰度图，跳过
            if len(img.shape) == 2:
                continue
            
            # 转换为灰度图
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 原地保存（覆盖原文件）
            cv2.imwrite(str(img_path), gray_img)
            converted_count += 1
            
        except Exception as e:
            print(f"错误: 处理文件 {img_path} 时发生异常: {e}")
            continue
    
    print(f"转换完成！共处理 {len(image_files)} 个文件，成功转换 {converted_count} 个彩色图片为灰度图。")


if __name__ == "__main__":
    # 设置要处理的目录
    target_dir = "../../dataset/ballGray/images"
    
    # 转换为绝对路径
    abs_target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), target_dir))
    
    if not os.path.exists(abs_target_dir):
        print(f"错误: 目录 {abs_target_dir} 不存在")
        exit(1)
    
    print(f"开始处理目录: {abs_target_dir}")
    convert_images_to_grayscale(abs_target_dir)