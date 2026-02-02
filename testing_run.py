from src.markDetection import MarkerImproved_yoloDetector,HarrisPara, HarrisDilate, OtherPara, DetectionPara, YOLOInferPAra
from src.lift2Dto3D import Lifter, load_camera_para_from_json
from ultralytics.models.yolo import YOLO
import os
from pathlib import Path
import argparse
import numpy as np
import cv2
import json
from tqdm import tqdm
from src.util import VIDEO_EXTENSIONS, IMG_EXTENSIONS
from src.util.yaml2dataclass import load_yaml_as_dataclass
from src.util.cilHelp import load_cli_args
os.environ['_VISIBLE_DEVICES'] = '5'


# YOLO_INFER = YOLOInferPAra(iou=0.4, imgsz=(1024, 1024), conf=0.4,augment=False)
# DETECTION_PARA = DetectionPara(
#                                 ts=0.3,
#                                 detectPara_Harris=HarrisPara(
#                                     blockSize=11,
#                                     ksize=5,
#                                     k=0.04
#                                 ),
#                                 detectPara_other_dict=OtherPara(
#                                     sub_pixcel_window_size=9,
#                                     harris_dilate=HarrisDilate(
#                                         kernel_size=3,
#                                         interations=1
#                                     )
#                                 )
# )


def singleVideo_run(model_path:str, video_source_path:str, yoloInfer_path:str, detection_path:str, output_dir:str, batch_size:int=4, topK:int=2,show=False):


    YOLO_INFER = load_yaml_as_dataclass(yoloInfer_path, YOLOInferPAra)
    DETECTION_PARA = load_yaml_as_dataclass(detection_path, DetectionPara)
    model = YOLO(model_path)
    detector = MarkerImproved_yoloDetector(model,yolo_infer_para=YOLO_INFER,detection_para=DETECTION_PARA,video_batch_size=batch_size)
    if os.path.isfile(video_source_path):# video file
        out_video_path = os.path.join(output_dir,f"marked_{os.path.basename(video_source_path)}")
        if show:
            marker_detection_per_frame, yolo_results_per_frame = detector.videoInferShow(video=video_source_path,show_new_video=out_video_path,apply_mask=True,topK=topK)
        else:
            marker_detection_per_frame, yolo_results_per_frame = detector.videoInfer(video=video_source_path,apply_mask=True,topK=topK)
        saving_path = os.path.join(output_dir,f"{os.path.basename(video_source_path).split('.')[0]}_2dInfer.json")
        with open(saving_path,'w') as file:
            json.dump({
                'marker_detection_per_frame':[i.tolist() for i in marker_detection_per_frame],
                'yolo_results_per_frame': yolo_results_per_frame,
            },fp=file)

    # else:# picture dir
    #     images = [os.path.join(source_path, f) for f in os.listdir(source_path) if f.lower().endswith(image_extensions)]
    #     for i in tqdm(range(0, len(images), batch_size)):
    #         batch_imgs = images[i:i+batch_size]
    #         finedtections,yolo_results = detector.ImgArraysBatchInfer(batch_imgs)
    #         for finedtection, result in zip(finedtections,yolo_results):
    #             name = Path(result.path).name
    #             saving_path = os.path.join(output_dir, name)
    #             show = result.plot(kpt_radius=2, line_width=1)
    #             show = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
    #             # print(finedtection)
    #             for det in finedtection:
    #                 x, y = int(det[0]), int(det[1])
    #                 cv2.circle(show, (x, y), 4, (0, 255, 0), 2)
    #             cv2.imwrite(saving_path, show)
    #             print(saving_path)

ARGS = {
    "model_path": "model.pt",
    "video_source_path": 'data/example.mp4',
    "yoloInfer_path": 'config/demo_yoloinfer.yaml',
    "detection_path" : 'config/marker_detection_profile.yaml',
    "topK": 2,
    "output_dir": 'temp',
    "batch_size": 10,
    "cuda": 5,
    "show" : False
}
HELPS = {
    "model_path": "Path to the YOLO model file",
    "video_source_path": "Path to the input video file",
    "yoloInfer_path": "Path to the YOLO inference configuration YAML file",
    "detection_path": "Path to the marker detection configuration YAML file",
    "topK": "Number of top detections to consider",
    "output_dir": "Directory to save output files",
    "batch_size": "Batch size for processing",
    "show": "Whether to display the video with detections"
}

if __name__ == "__main__":
    args, args_dict = load_cli_args(ARGS,HELPS)
    cuda_id = args_dict.pop('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)
    singleVideo_run(**args_dict)

