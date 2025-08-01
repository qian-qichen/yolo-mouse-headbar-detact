from PIL import Image
import os
from pathlib import Path
from ultralytics import YOLO
import argparse
def yoloResultGetAndPlot(model_path,source_img_dir,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = YOLO(model_path)
    image_extensions = ('.jpg', '.jpeg', '.png')
    # import pdb;pdb.set_trace()
    images = [Path(os.path.join(source_img_dir, f)) for f in os.listdir(source_img_dir) if f.lower().endswith(image_extensions)]
    
    # results = model.predict(images, conf = 0.1) 
    results = model.predict(images,
                            iou=0.5,
                            imgsz=(2304, 2048),
                            conf = 0.5,
                            ) 
    # print(results)
    for result in results:
        name = Path(result.path).name
        saving_path = os.path.join(output_dir, name)
        # import pdb;pdb.set_trace()
        # Save results to disk
        result.save(filename=saving_path,kpt_radius=2,line_width=1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="show yolo result")
    parser.add_argument('-m',"--model_path",type=str)
    parser.add_argument('-s',"--source_img_dir", type=str)
    parser.add_argument('-o', "--output_dir", type=str)
    args = parser.parse_args()
    yoloResultGetAndPlot(model_path=args.model_path, source_img_dir=args.source_img_dir, output_dir=args.output_dir)



