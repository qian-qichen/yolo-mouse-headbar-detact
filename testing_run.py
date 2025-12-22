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
os.environ['CUDA_VISIBLE_DEVICES'] = '5'


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




ARGS = {
    "model_path": "model.pt",
    "video_source_path": 'data/example.mp4',
    "yoloInfer_path": 'config/demo_yoloinfer.yaml',
    "detection_path" : 'config/marker_detection_profile.yaml',
    "topK": 2,
    "output_dir": 'temp',
    "batch_size": 10
}

def singleVideo_run(model_path:str, video_source_path:str, yoloInfer_path:str, detection_path:str, output_dir:str, batch_size:int=4, topK:int=2):


    YOLO_INFER = load_yaml_as_dataclass(yoloInfer_path, YOLOInferPAra)
    DETECTION_PARA = load_yaml_as_dataclass(detection_path, DetectionPara)
    model = YOLO(model_path)
    detector = MarkerImproved_yoloDetector(model,yolo_infer_para=YOLO_INFER,detection_para=DETECTION_PARA,video_batch_size=batch_size)
    if os.path.isfile(video_source_path):# video file
        out_video_path = os.path.join(output_dir,f"marked_{os.path.basename(video_source_path)}")
        marker_detection_per_frame, yolo_results_per_frame = detector.videoInferShow(video=video_source_path,show_new_video=out_video_path,apply_mask=True,topK=topK)
        saving_path = os.path.join(output_dir,f"{os.path.basename(video_source_path).split('.')[0]}.json")
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



def main():
    
    model_path = "runs/headbar/weights/best.pt"
    source_path = 'data/headbarDual/'
    # yoloInfer_path = 'data/headbarDual/side1127_yoloinfer.yaml'
    yoloInfer_path = 'data/headbarDual/top1127_yoloinfer.yaml'
    detection_path = 'temp/marker_detection_profile.yaml'
    DETECTION_PARA = load_yaml_as_dataclass(detection_path, DetectionPara)
    topK = 2
    output_dir = 'headbartemp'
    batch_size = 10
    output_dir = 'show'
    POINTS_NUM = 2
    batch_size = 10
    model = YOLO(model_path)
    detector = MarkerImproved_yoloDetector(model,detection_para=DETECTION_PARA)
    # 查找source_path下所有json文件
    video_files = [f for f in os.listdir(source_path) if f.lower().endswith(VIDEO_EXTENSIONS)]
    cams_para = {}
    videos_path_ = {}
    failed_count = 0
    for video_file in video_files:
        base_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(source_path, video_file)
        json_path = video_path.split('.')[0] + '.json'
        try:
            camera_para = load_camera_para_from_json(json_path)
        except Exception as e:
            print(e)
            continue
        cams_para[base_name] = camera_para
        videos_path_[base_name] = video_path
    cams_para = {k: v for k, v in sorted(cams_para.items(), key=lambda x: x[0])}
    cam_names = list(cams_para.keys())
    lifter = Lifter(cams=cams_para)
    videos_path = {name:videos_path_[name] for name in lifter.cam_order}

    cap = cv2.VideoCapture(videos_path[cam_names[0]])
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    lifter_outs = []
    for i,outs in tqdm(enumerate(detector.multiVideoInferGenetater(videos_path,batch_size=1,topK=2)),total=frame_count): # due to program limitation, batch_size must be 1 here
        try:
            # points = np.stack([out[0][0][:2,:2] for name,out in outs.items()])
            valid_points = {name:out['marker'][0][:2,:2] for name,out in outs.items()}
            points = np.stack(list(valid_points.values()))
            assert points.shape[1] == POINTS_NUM, f"detected points num {points.shape[1]} not equal to required {POINTS_NUM}"
            valid_camName = list(valid_points.keys())
        except Exception as e:
            print('failed at frame ',i,e)
            failed_count +=1
            lifter_outs.append(None)
            # import pdb;pdb.set_trace()
            continue
        # print(points.shape)
        lifting_out = lifter.undistortedliftingLine(points.copy(),valid_camName)
        middle_point = lifter.lifting(points.mean(axis=1)[np.newaxis,:,:])

        out = {"v":lifting_out[0].tolist(),'p':lifting_out[1].tolist(),'middle':middle_point.tolist()}
        pt2d = {name:points[i].tolist() for i,name in enumerate(cam_names)}
        out = {**out, **pt2d}
        lifter_outs.append(out)
    # iterer = [detector.videoInferGenerater(videos_path[name],batch_size=1) for name in cam_names]
    # 

    # for i,outs in tqdm(enumerate(zip(*iterer)),total=frame_count):

    #     try:
    #         points = np.stack([out[0][0][:2,:2] for out in outs])
    #     except Exception as e:
    #         lifter_outs.append(None)
    #         continue
    #     # print(points.shape)
    #     lifting_out = lifter.undistortedliftingLine(points.copy())
    #     middle_point = lifter.lifting(points.mean(axis=1)[np.newaxis,:,:])

    #     out = {"v":lifting_out[0].tolist(),'p':lifting_out[1].tolist(),'middle':middle_point.tolist()}
    #     pt2d = {name:points[i].tolist() for i,name in enumerate(cam_names)}
    #     out = {**out, **pt2d}
    #     lifter_outs.append(out)

    with open(os.path.join(source_path,'lifting_out.json'),'w') as f:
        json.dump(lifter_outs,f)
    # for cam_name,video_path in videos_path.items():
    #     marker_detection_per_frame, yolo_results_per_frame = detector.videoInfer(video=video_path)
    #     d2_results[cam_name] = {
    #         'marker_detection_per_frame':marker_detection_per_frame,
    #         'yolo_results_per_frame':yolo_results_per_frame,
    #     }
    print(f"total failed frames: {failed_count}/{frame_count}")



if __name__ == "__main__":
    args, args_dict = load_cli_args(ARGS)
    singleVideo_run(**args_dict)
    # main2()
