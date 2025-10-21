import cv2
import os
import argparse
from tqdm import tqdm

def invert_image_opencv(image_path, save_path):
    image = cv2.imread(image_path)
    # 对每个通道进行反色处理
    assert image is not None, f'fail to read image {image_path}'
    # inverted_image = 255 - image
    inverted_image = image

    if len(inverted_image.shape) == 3 and inverted_image.shape[2] == 3:
        img_HSV = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2HSV)
        # img_HSV[:, :, 2] = cv2.equalizeHist(img_HSV[:, :, 2])
        img_HSV[:, :, 2] = 255 - img_HSV[:, :, 2]
        inverted_image = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2BGR)
    else:
        inverted_image = cv2.equalizeHist(inverted_image)
    cv2.imwrite(save_path, inverted_image)

def batch_invert_images_opencv(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(('png', 'jpg', 'jpeg')):
            # print(f"Processing {filename} with OpenCV...")
            invert_image_opencv(os.path.join(input_folder, filename),
                                os.path.join(output_folder, filename))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Invert colors of images in a folder using OpenCV.")
    parser.add_argument("input_folder", help="Path to the input folder containing images.")
    parser.add_argument("output_folder", help="Path to the output folder to save inverted images.")
    args = parser.parse_args()
    batch_invert_images_opencv(args.input_folder, args.output_folder)