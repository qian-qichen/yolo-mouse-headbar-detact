from PIL import Image
import os
from pathlib import Path
from ultralytics.models.yolo import YOLO
import argparse
import json
from tqdm import tqdm
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
VIDEOS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def yoloinfer(model_path, source_path, output_path, batch_size, iou, imgsz, conf,best_only=False):
        model_path = Path(model_path)
        if not model_path.exists() or not model_path.is_file():
            raise ValueError(f"Model path {model_path} does not exist or is not a file.")
        output_path = Path(output_path)
        # if output_dir.
        # output_path.mkdir(parents=True, exist_ok=True)

        source_path = Path(source_path)
        if source_path.is_dir():
            print(f"Source path {source_path} is a directory.")
            raise NotImplementedError
        elif source_path.is_file():
            if source_path.suffix.lower() in VIDEOS:
                print(f"Source path {source_path} is a video file.")
                if output_path.is_dir():
                    output_path = os.path.join(output_path,source_path.stem+'_idHelper.json') 
                model = YOLO(model_path)
                results = model.track(source=source_path, batch=batch_size, iou=iou, imgsz=imgsz, conf=conf, stream=True,verbose=False,tracker="botsort.yaml")
                # imgsz should be (height, width)
                failed = []
                VideoLenth = get_video_frame_count(source_path)
                with open(output_path, 'w', encoding='utf-8') as f:
                    for i, result in tqdm(enumerate(results),total=VideoLenth):
                        try:
                            detections = json.loads(result.to_json())
                        except Exception:
                            # fallback: 将原始字符串写入
                            f.write(result.to_json())
                            f.write('\n')
                            failed.append(i)
                            continue
                        if not detections:
                            f.write('[]\n')
                            failed.append(i)
                            continue
                        if best_only and detections:
                            best = max(detections, key=lambda x: x.get('confidence', 0.0))
                            f.write(json.dumps([best], ensure_ascii=False))
                        else:
                            f.write(json.dumps(detections, ensure_ascii=False))
                        f.write('\n')
                print(f"frame index that detection failed:\n {failed}")
            else:
                raise ValueError(f"Source path {source_path} is a file but not a recognized video format.")
        else:
            raise ValueError(f"Source path {source_path} does not exist.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="show yolo result")
    parser.add_argument('-m',"--model_path",type=str)
    parser.add_argument('-s',"--source_path", type=str)
    parser.add_argument('-o', "--output_path", default='show',type=str)
    parser.add_argument('-b', "--batch_size", default=24, type=int)
    parser.add_argument('-i', "--iou", default=0.2, type=float)
    parser.add_argument('-isz', "--imgsz", nargs=2, default=[1024, 1024], type=int)
    parser.add_argument('-c', "--conf", default=0.8, type=float)
    parser.add_argument('-bo','--best_only',action='store_true')
    # parser.add_argument("-cut","--cutOutputImage", action="store_true",default=False)
    args = parser.parse_args()
    yoloinfer(model_path=args.model_path, source_path=args.source_path, output_path=args.output_path, batch_size=args.batch_size, iou=args.iou, imgsz=tuple(args.imgsz), conf=args.conf, best_only=args.best_only)




