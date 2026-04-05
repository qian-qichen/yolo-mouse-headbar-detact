import json
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

def draw_yolo_boxes_on_video(yolo_json_path, video_path, output_path, conf_thresh=0.3, box_color=(0,255,0), thickness=2):
    # 读取YOLO检测结果
    file = open(yolo_json_path, 'r')

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')

    # 视频参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for idx in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        yolo_result = json.loads(file.readline())
        # 读取对应帧的检测结果
        dets = yolo_result
        if not dets:
            out.write(frame)
            continue
        if isinstance(dets, dict):
            dets = [dets]
        for det in dets:
            if not det or not isinstance(det, dict):
                continue
            if 'box' not in det:
                continue
            conf = det.get('confidence', 1.0)
            if conf < conf_thresh:
                continue
            box = det['box']
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
            label = det.get('name', str(det.get('class', '')))
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        out.write(frame)
    cap.release()
    out.release()
    file.close()


def main():
    parser = argparse.ArgumentParser(description='Draw YOLO 2D detection results on video frames.')
    parser.add_argument('-j', '--yolo_json', required=True, help='Path to YOLO output JSON file')
    parser.add_argument('-v', '--video', required=True, help='Path to input video')
    parser.add_argument('-o', '--output', default=None, help='Path to output video (optional)')
    parser.add_argument('-c', '--conf', type=float, default=0.3, help='Confidence threshold')
    args = parser.parse_args()
    draw_yolo_boxes_on_video(args.yolo_json, args.video, args.output, args.conf)

if __name__ == '__main__':
    main()
