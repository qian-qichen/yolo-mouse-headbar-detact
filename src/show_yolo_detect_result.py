from PIL import Image
import os
from pathlib import Path
from ultralytics.models.yolo import YOLO
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
batch_size=4
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
def yoloResultGetAndPlot(model_path,source_img_dir,output_dir, max_frames=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = YOLO(model_path)

    if Path(source_img_dir).is_dir():
        print(f"Source path {source_img_dir} is a directory.")
        image_extensions = ('.jpg', '.jpeg', '.png')
        # import pdb;pdb.set_trace()
        images = [Path(os.path.join(source_img_dir, f)) for f in os.listdir(source_img_dir) if f.lower().endswith(image_extensions)]
        
        # 限制处理的图像数量
        if max_frames is not None and max_frames > 0:
            images = images[:max_frames]
            print(f"Processing only first {len(images)} images due to max_frames limit")
        
        # 分批推理
        for i in tqdm(range(0, len(images), batch_size)):
            batch_imgs = images[i:i+batch_size]
            results = model.predict(batch_imgs,
                                   iou=0.2,
                                   imgsz=1920,  # 使用与训练相同的输入尺寸
                                   conf=0.05,
                                   )
            for result in results:
                name = Path(result.path).name
                saving_path = os.path.join(output_dir, name)
                result.save(filename=saving_path, kpt_radius=2, line_width=1, color_mode= 'instance')
                # print(result.keypoints)
    elif Path(source_img_dir).is_file():
        print(f"Source path {source_img_dir} is a file.")
        if source_img_dir.lower().endswith(video_extensions):
            print(f"Source path {source_img_dir} is a video file.")
            test_cap = cv2.VideoCapture(source_img_dir)
            ret, frame = test_cap.read()
            if not ret:
                raise RuntimeError("Failed to read the first frame from video.")
            height, width = frame.shape[:2]
            frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video resolution: {width}x{height}, Total frames: {frame_count}")
            test_cap.release()
            out_video_path = os.path.join(output_dir,f"marked_{os.path.basename(source_img_dir)}")
            # results = model.track(source=source_img_dir, batch=batch_size, iou=0.2, imgsz=1920, conf=0.05, stream=True,verbose=False,tracker="botsort.yaml")  # 使用与训练相同的输入尺寸
            results = model.predict(
                source=source_img_dir,
                iou=0.05,
                imgsz=1920,
                conf=0.05,
                stream=True,
                verbose=False
            )
            writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
            
            # 确定要处理的帧数
            total_frames_to_process = min(frame_count, max_frames) if max_frames is not None and max_frames > 0 else frame_count
            pbar = tqdm(total=total_frames_to_process, desc="video infering")
            
            frame_idx = 0
            for result in results:
                if max_frames is not None and frame_idx >= max_frames:
                    break
                frame = result.plot()
                writer.write(frame)
                pbar.update()
                frame_idx += 1
            writer.release()
            pbar.close()
        else:
            raise ValueError(f"Unsupported file type: {source_img_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="show yolo result")
    parser.add_argument('-m',"--model_path",default='runs/headbar2stage/weights/best.pt',type=str)  # 修正为正确的模型路径
    parser.add_argument('-s',"--source_img_dir",default='dataset/headbar/images/test', type=str)  # 修正为正确的数据集路径
    parser.add_argument('-o', "--output_dir", default='show',type=str)
    parser.add_argument('-n', "--max_frames", default=None, type=int, help="Maximum number of frames/images to process")
    # parser.add_argument("-cut","--cutOutputImage", action="store_true",default=False)
    args = parser.parse_args()
    yoloResultGetAndPlot(model_path=args.model_path, source_img_dir=args.source_img_dir, output_dir=args.output_dir, max_frames=args.max_frames)