import os
import random
import shutil
import argparse

def random_select_images(source_folder, demo_folder, num_images):
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"源文件夹 {source_folder} 不存在！")
        return
    
    # 创建目标文件夹
    if not os.path.exists(demo_folder):
        os.makedirs(demo_folder)
    
    # 获取所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(source_folder) if f.lower().endswith(image_extensions)]
    
    # 检查图片数量是否足够
    if len(images) < num_images:
        print(f"源文件夹中的图片数量不足！共有 {len(images)} 张图片，无法选取 {num_images} 张。")
        return
    
    # 随机选择指定数量的图片
    selected_images = random.sample(images, num_images)
    
    # 复制图片到目标文件夹
    for image in selected_images:
        src_path = os.path.join(source_folder, image)
        dst_path = os.path.join(demo_folder, image)
        shutil.copy(src_path, dst_path)
    
    print(f"已成功随机选取 {num_images} 张图片并复制到 {demo_folder} 文件夹中！")



def main():
    parser = argparse.ArgumentParser(description="随机选择图片并复制到目标文件夹")
    parser.add_argument("source_folder", type=str, help="源图片文件夹路径")
    parser.add_argument("--demo_folder", type=str, default=None, help="目标文件夹路径（默认为 source_folder + '_demo'）")
    parser.add_argument("--num_images", type=int, default=200, help="随机选择的图片数量（默认为 200）")

    
    args = parser.parse_args()
    if args.demo_folder is None:
        args.demo_folder = args.source_folder + '_demo'
    
    random_select_images(args.source_folder, args.demo_folder, args.num_images)

    
if __name__ == "__main__":
    main()