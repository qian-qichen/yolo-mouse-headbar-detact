from PIL import Image
import os
from pathlib import Path
from ultralytics.models.yolo import YOLO
import argparse
batch_size=4
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
def yoloResultGetAndPlot(model_path,source_img_dir,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = YOLO(model_path)
    image_extensions = ('.jpg', '.jpeg', '.png')
    # import pdb;pdb.set_trace()
    images = [Path(os.path.join(source_img_dir, f)) for f in os.listdir(source_img_dir) if f.lower().endswith(image_extensions)]
    # images_batches = 
    # results = model.predict(images, conf = 0.1) 

    # 分批推理
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i+batch_size]
        results = model.predict(batch_imgs,
                               iou=0.2,
                               imgsz=(1008, 1008),
                               conf=0.05,
                               )
        for result in results:
            name = Path(result.path).name
            saving_path = os.path.join(output_dir, name)
            result.save(filename=saving_path, kpt_radius=2, line_width=1, color_mode= 'instance')
            # print(result.keypoints)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="show yolo result")
    parser.add_argument('-m',"--model_path",default='runs/glassBoard-r/weights/best.pt',type=str)
    parser.add_argument('-s',"--source_img_dir",default='dataset/glassR/images/test', type=str)
    parser.add_argument('-o', "--output_dir", default='show',type=str)
    # parser.add_argument("-cut","--cutOutputImage", action="store_true",default=False)
    args = parser.parse_args()
    yoloResultGetAndPlot(model_path=args.model_path, source_img_dir=args.source_img_dir, output_dir=args.output_dir)



