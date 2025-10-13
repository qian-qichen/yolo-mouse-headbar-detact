from src.markDetection import MarkerImproved_yoloDetector,HarrisPara, HarrisDilate, OtherPara, DetectionPara, YOLOInferPAra
from src.lift2Dto3D import Lifter, load_camera_para_from_json
from ultralytics import YOLO
import os
from pathlib import Path
import argparse
import numpy as np
import cv2
import json
from tqdm import tqdm

DETECTION_PARA = DetectionPara(
                                ts=0.8,
                                detectPara_Harris=HarrisPara(
                                    blockSize=9,
                                    ksize=3,
                                    k=0.06
                                ),
                                detectPara_other_dict=OtherPara(
                                    sub_pixcel_window_size=10,
                                    harris_dilate=HarrisDilate(
                                        kernel_size=3,
                                        interations=1
                                    )
                                )
)
YOLO_INFER = YOLOInferPAra(iou=0.5, imgsz=(3840, 2176), conf=0.50)
image_extensions = ('.jpg', '.jpeg', '.png')
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
def main1():

    model_path = "runs/marker/weights/last.pt"
    # source_path = 'dataset/marker/images/train'
    source_path = 'data/dualCamExample/202509121.mov'
    output_dir = 'show'
    batch_size = 10
    model = YOLO(model_path)
    detector = MarkerImproved_yoloDetector(model,yolo_infer_para=YOLO_INFER,detection_para=DETECTION_PARA)
    if os.path.isfile(source_path):# video file
        out_video_path = os.path.join(output_dir,f"marked_{os.path.basename(source_path)}")
        marker_detection_per_frame, yolo_results_per_frame = detector.videoinferShow(video=source_path,show_new_video=out_video_path)
        saving_path = os.path.join(output_dir,f"{os.path.basename(source_path).split('.')[0]}.json")
        with open(saving_path,'w') as file:
            json.dump({
                'marker_detection_per_frame':[i.tolist() for i in marker_detection_per_frame],
                'yolo_results_per_frame': yolo_results_per_frame,
            },fp=file)

    else:# picture dir
        images = [os.path.join(source_path, f) for f in os.listdir(source_path) if f.lower().endswith(image_extensions)]
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i+batch_size]
            finedtections,yolo_results = detector.ImgArraysBatchInfer(batch_imgs)
            for finedtection, result in zip(finedtections,yolo_results):
                name = Path(result.path).name
                saving_path = os.path.join(output_dir, name)
                show = result.plot(kpt_radius=2, line_width=1)
                show = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
                # print(finedtection)
                for det in finedtection:
                    x, y = int(det[0]), int(det[1])
                    cv2.circle(show, (x, y), 4, (0, 255, 0), 2)
                cv2.imwrite(saving_path, show)
                print(saving_path)

def main2():
    model_path = "runs/marker/weights/last.pt"
    source_path = 'data/dualCamExample/video'
    video_source_path = "data/dualCamExample/video"
    output_dir = 'show'
    POINTS_NUM = 2
    batch_size = 10
    model = YOLO(model_path)
    detector = MarkerImproved_yoloDetector(model,detection_para=DETECTION_PARA)
    # 查找source_path下所有json文件
    json_files = [f for f in os.listdir(source_path) if f.lower().endswith('.json')]
    cams_para = {}
    videos_path = {}
    for json_file in json_files:
        base_name = os.path.splitext(json_file)[0]
        json_path = os.path.join(source_path, json_file)
        camera_para = load_camera_para_from_json(json_path)
        video_file = None
        for ext in video_extensions:
            candidate = os.path.join(video_source_path, base_name + ext)
            if os.path.isfile(candidate):
                video_file = candidate
                break
        if video_file is None:
            print(f"No video found for {json_file}")
            continue
        cams_para[base_name] = camera_para
        videos_path[base_name] = video_file
    cams_para = {k: v for k, v in sorted(cams_para.items(), key=lambda x: x[0])}
    cam_names = list(cams_para.keys())
    lifter = Lifter(cams=cams_para)

    cap = cv2.VideoCapture(videos_path[cam_names[0]])
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    iterer = [detector.videoInferGenerater(videos_path[name],batch_size=1) for name in cam_names]
    lifter_outs = []
    import pdb
    for i,outs in tqdm(enumerate(zip(*iterer)),total=frame_count):
        try:
            points = np.stack([out[0][0][:,:2] for out in outs])
        except Exception as e:
            lifter_outs.append(None)
            continue

        lifting_out = lifter.undistortedliftingLine(points)
        lifter_outs.append({"v":lifting_out[0].tolist(),'p':lifting_out[1].tolist()})
    with open(os.path.join(video_source_path,'lifting_out.json'),'w') as f:
        json.dump(lifter_outs,f)
    # for cam_name,video_path in videos_path.items():
    #     marker_detection_per_frame, yolo_results_per_frame = detector.videoInfer(video=video_path)
    #     d2_results[cam_name] = {
    #         'marker_detection_per_frame':marker_detection_per_frame,
    #         'yolo_results_per_frame':yolo_results_per_frame,
    #     }



if __name__ == "__main__":
    main2()
