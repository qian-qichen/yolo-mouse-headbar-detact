from src.markDetection import MarkerImproved_yoloDetector,HarrisPara, HarrisDilate, OtherPara, DetectionPara
from ultralytics import YOLO
import os
from pathlib import Path
import argparse
import numpy as np
import cv2
import json

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

def main():

    model_path = "runs/marker/weights/last.pt"
    # source_path = 'dataset/marker/images/train'
    source_path = 'data/headBarEample.mp4'
    output_dir = 'show'
    image_extensions = ('.jpg', '.jpeg', '.png')
    batch_size = 10
    model = YOLO(model_path)
    detector = MarkerImproved_yoloDetector(model,detection_para=DETECTION_PARA)
    if os.path.isfile(source_path):# video file
        out_video_path = os.path.join(output_dir,f"marked_{os.path.basename(source_path)}")
        marker_detection_per_frame, yolo_results_per_frame = detector.videoInfer(video=source_path,show_new_video=out_video_path)
        saving_path = os.path.join(output_dir,f"{os.path.basename(source_path).split('.')[0]}.json")
        with open(saving_path,'w') as file:
            json.dump({
                'marker_detection_per_frame':np.asarray(marker_detection_per_frame).tolist(), 
                'yolo_results_per_frame':np.asarray(yolo_results_per_frame).tolist()
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

if __name__ == "__main__":
    main()
