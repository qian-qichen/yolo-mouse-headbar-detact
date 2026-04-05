from PIL import Image
import os
from pathlib import Path
from ultralytics.models.yolo import YOLO
import argparse
import json
from tqdm import tqdm
import cv2
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
VIDEOS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def yoloinfer(model_path, source_path, output_path, batch_size, iou, imgsz, conf,best_only=False):
        """Run YOLO inference on a video source and save results to a JSON-like file. Each line in the output file corresponds to a frame in the video and contains a JSON array of detected objects for that frame."""
        model_path = Path(model_path)
        if not model_path.exists() or not model_path.is_file():
            raise ValueError(f"Model path {model_path} does not exist or is not a file.")
        output_path = Path(output_path)
        # if output_dir.
        # output_path.mkdir(parents=True, exist_ok=True)
        if not isinstance(output_path, Path):  
            source_path = Path(source_path)
        if source_path.is_dir():
            print(f"Source path {source_path} is a directory.")
        elif source_path.is_file():
            if source_path.suffix.lower() in VIDEOS:
                print(f"Source path {source_path} is a video file.")
                if output_path.is_dir():
                    output_path = os.path.join(output_path,source_path.stem+'_yolo.json') 
                model = YOLO(model_path)
                # results = model.track(source=source_path, batch=batch_size, iou=iou, imgsz=imgsz, conf=conf, stream=True,verbose=False,tracker="botsort.yaml")
                results = model.predict(source=source_path, batch=batch_size, iou=iou, imgsz=imgsz, conf=conf, stream=True,verbose=False)
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
                            # keep the best detection per class
                            grouped = {}
                            for det in detections:
                                # support various key names for class and confidence
                                cls = det.get('class') if isinstance(det, dict) else None
                                if cls is None:
                                    cls = det.get('name') if isinstance(det, dict) else None
                                    if cls is None:
                                        cls = det.get('class_name') if isinstance(det, dict) else None
                                # fallback to numeric class id as string
                                if cls is None:
                                    cls = str(det.get('class', '')) if isinstance(det, dict) else None
                                if cls is None or cls == '':
                                    continue
                                conf = det.get('confidence', det.get('conf', det.get('score', 0.0))) if isinstance(det, dict) else 0.0
                                if conf is None:
                                    conf = 0.0
                                existing = grouped.get(cls)
                                if existing is None or conf > (existing.get('confidence', existing.get('conf', existing.get('score', 0.0))) if isinstance(existing, dict) else 0.0):
                                    grouped[cls] = det
                            bests = list(grouped.values()) if grouped else [max(detections, key=lambda x: x.get('confidence', x.get('conf', x.get('score', 0.0)) if isinstance(x, dict) else 0.0))]
                            f.write(json.dumps(bests, ensure_ascii=False))
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
    parser.add_argument('-s',"--source", type=str)
    parser.add_argument('-o', "--output", default='show',type=str)
    parser.add_argument('-b', "--batch_size", default=24, type=int)
    parser.add_argument('-i', "--iou", default=0.2, type=float)
    parser.add_argument('-isz', "--imgsz", nargs=2, default=[1920, 1920], type=int)
    parser.add_argument('-c', "--conf", default=0.1, type=float)
    parser.add_argument('-bo','--best_only',action='store_true')
    # parser.add_argument("-cut","--cutOutputImage", action="store_true",default=False)
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise ValueError(f"Source path {source} does not exist.")
    elif source.is_dir():
        print(f"Source path {source} is a directory.")
        videos_path = [i for i in source.glob('*') if i.is_file() and i.suffix.lower() in VIDEOS]
        for video_path in videos_path:
            yoloinfer(model_path=args.model_path, source_path=video_path, output_path=args.output, batch_size=args.batch_size, iou=args.iou, imgsz=tuple(args.imgsz), conf=args.conf, best_only=args.best_only)
    elif source.is_file():
        yoloinfer(model_path=args.model_path, source_path=source, output_path=args.output, batch_size=args.batch_size, iou=args.iou, imgsz=tuple(args.imgsz), conf=args.conf, best_only=args.best_only)




