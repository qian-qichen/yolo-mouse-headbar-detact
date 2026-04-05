from src.markDetection import MarkerImproved_yoloDetector,BallImproved_yoloDetector, HarrisPara, HarrisDilate, OtherPara, DetectionPara, YOLOInferPAra, BallDetectionPara
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
from src.util.helper import load_yaml_as_dataclass
from src.util.cilHelp import load_cli_args
os.environ['_VISIBLE_DEVICES'] = '5'


def _to_jsonable(value):
    """Recursively convert numpy values and nested containers to JSON-safe types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


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


def marker_singleVideo_run(model_path:str, video_source_path:str, yoloInfer_path:str, output_dir:str, detection_path:str, batch_size:int=4, topK:int=2,show=False):


    YOLO_INFER = load_yaml_as_dataclass(yoloInfer_path, YOLOInferPAra)
    DETECTION_PARA = load_yaml_as_dataclass(detection_path, DetectionPara) if detection_path else None
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
                'marker_detection_per_frame': _to_jsonable(marker_detection_per_frame),
                'yolo_results_per_frame': _to_jsonable(yolo_results_per_frame),
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


def ball_singleVideo_run(model_path: str, video_source_path: str, yoloInfer_path: str, output_dir: str, detection_path: str, batch_size: int = 4, topK: int = 2, show=False):
    YOLO_INFER = load_yaml_as_dataclass(yoloInfer_path, YOLOInferPAra)
    DETECTION_PARA = load_yaml_as_dataclass(detection_path, BallDetectionPara) if detection_path else None
    model = YOLO(model_path)
    h_range = {
        'left': ((32, 45),),
        'right':((0, 15),), 
    }
    detector = BallImproved_yoloDetector(model,H_ranges=h_range, yolo_infer_para=YOLO_INFER, ball_detect_para=DETECTION_PARA, video_batch_size=batch_size)
    # import pdb;pdb.set_trace()
    if os.path.isfile(video_source_path):  # video file
        out_video_path = os.path.join(output_dir, f"ball_marked_{os.path.basename(video_source_path)}")
        if show:
            ball_detection_per_frame, yolo_results_per_frame = detector.videoInferShow(video=video_source_path, show_new_video=out_video_path, apply_mask=True, topK=topK)
        else:
            ball_detection_per_frame, yolo_results_per_frame = detector.videoInfer(video=video_source_path, apply_mask=True, topK=topK)
        json_saving_path = os.path.join(os.path.dirname(video_source_path), f"{os.path.basename(video_source_path).split('.')[0]}_ball_2dInfer.json")
        with open(json_saving_path, 'w') as file:
            json.dump({
                'ball_detection_per_frame': _to_jsonable(ball_detection_per_frame),
                'yolo_results_per_frame': _to_jsonable(yolo_results_per_frame),
            }, fp=file)
    else:
        print(f'no such file {video_source_path}')


ARGS = {
    "model_path": "model.pt",
    "video_source_path": 'data/example.mp4',
    "yoloInfer_path": 'config/demo_yoloinfer.yaml',
    "detection_path" : 'config/marker_detection_profile.yaml',
    "topK": 1,
    "output_dir": 'temp',
    "batch_size": 10,
    "cuda": 5,
    "show" : False,
    "marker": "ball"
}

HELPS = {
    "model_path": "Path to the YOLO model file",
    "video_source_path": "Path to the input video file",
    "yoloInfer_path": "Path to the YOLO inference configuration YAML file",
    "detection_path": "Path to the marker detection configuration YAML file",
    "topK": "Number of top detections to consider",
    "output_dir": "Directory to save output files",
    "batch_size": "Batch size for processing",
    "output_dir": "Whether to display the video with detections",
    "marker": "which kind of marker to detect, currently support 'ball' and 'corner'"
}

if __name__ == "__main__":
    args, args_dict = load_cli_args(ARGS,HELPS)
    cuda_id = args_dict.pop('cuda')
    marker = args_dict.pop('marker')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)
    if marker == 'corner':
        marker_singleVideo_run(**args_dict)
    elif marker == 'ball':
        ball_singleVideo_run(**args_dict)
    else:
        raise ValueError(f"Unknown marker type: {marker}.")
