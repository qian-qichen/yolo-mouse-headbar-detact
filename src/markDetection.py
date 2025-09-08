import cv2
import numpy as np
from typing import Optional, List, Callable, Protocol
import argparse
from ultralytics import YOLO
from ultralytics.engine.results import Results
import numpy as np
import torch
from ultralytics.utils.plotting import save_one_box
import os
from dataclasses import dataclass, field
import tempfile

DEFAULT_Harris_PARA = {
    'blockSize':9,
    'ksize':3,
    'k':0.04,
}
DEFAULT_other_PARA = {
    'sub_pixcel_window_size':9,
    'harris_dilate':{
        'kernel_size':3,
        'interations':2
    }
}
def cornerDetect(img,ts=0.01,detectPara_Harris:dict=DEFAULT_Harris_PARA,detectPara_other_dict:dict=DEFAULT_other_PARA)->np.ndarray:
    '''
    args:
        - img: image as numpy ndarray, BGR
    returns: markers[N,3] {x,y,trust}
    '''
    sub_pixcel_window_size = detectPara_other_dict['sub_pixcel_window_size']
    harris_dilate_kernel_size = detectPara_other_dict['harris_dilate']['kernel_size']
    harris_dilate_interations = detectPara_other_dict['harris_dilate']['interations']

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # gray = cv2.erode(gray, np.ones((3, 3), np.uint8), iterations=1)
    harris_response = cv2.cornerHarris(gray,**detectPara_Harris)
    dst = cv2.dilate(harris_response,kernel=np.ones((harris_dilate_kernel_size, harris_dilate_kernel_size), np.uint8),iterations=harris_dilate_interations) # type:ignore
    # dst = harris_response
    ret, dst = cv2.threshold(dst,ts*dst.max(),255,0)
    dst = np.uint8(dst)
 
    # find 
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst) # type:ignore
    centroids = centroids.astype(np.float32)
 
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = np.ones((centroids.shape[0],3),dtype=np.float32)
    corners[:,0:2] = cv2.cornerSubPix(gray,centroids,(sub_pixcel_window_size,sub_pixcel_window_size),(-1,-1),criteria)

    cornerInt = corners[:,0:2].round().astype(int)
    # corners[:,2] = harris_response[np.clip(cornerInt[:,1], 0, harris_response.shape[0]-1), np.clip(cornerInt[:,0], 0, harris_response.shape[1]-1)]
    corners[:,2] = harris_response[np.clip(cornerInt[:,1], 0, harris_response.shape[0]-1), np.clip(cornerInt[:,0], 0, harris_response.shape[1]-1)]

    return corners

DEFAULT_INFER_PARA = {
    'iou':0.4,
    'imgsz':(3840, 2176),
    'conf':0.5,
}

DEFAULT_DETECTION_PARA = {
    'ts':0.2,
    'detectPara_Harris':DEFAULT_Harris_PARA,
    'detectPara_other_dict':DEFAULT_other_PARA
}

def find_nearest_points_index(root:np.ndarray,candidate:np.ndarray):
    '''
    args:
        - root: [N points, coordinate]
        - candidate: [N points, coordinate]
    '''
    dists = np.linalg.norm(root[:, None, :] - candidate[None, :, :], axis=2)
    return np.argmin(dists, axis=1)
    # return list(range(candidate.shape[0]))

class MarkerDetectorProtocol(Protocol):
    def __call__(self, img: np.ndarray, *, ts: float, detectPara_Harris:dict,detectPara_other_dict:dict) -> np.ndarray: ...

class MarkerImproved_yoloDetector:
    
    def __init__(self,yolo_model:YOLO|dict,yolo_infer_para:dict=DEFAULT_INFER_PARA,markerDetector:MarkerDetectorProtocol=cornerDetect, detection_para:dict=DEFAULT_DETECTION_PARA,cores:int=8):
        if isinstance(yolo_model, YOLO):
            self.yolo = yolo_model
        else:
            ck_path = yolo_model.pop('ck_path')
            self.yolo = YOLO(**yolo_model).load(ck_path)
        self.yolo_infer_para = yolo_infer_para
        
        self.markerDetector = markerDetector
        self.detection_para = detection_para

        self.video_batch_size = 10

        max_core = os.cpu_count()
        assert isinstance(max_core,int)
        self.cores = min(cores, max_core) # TODO: add multi-processing optimization on marker detection
        if self.cores < cores:
            print(f"only use {self.cores} for marker detection due to hardware limitation")
    def ImgArraysBatchInfer(self,imgs:np.ndarray|torch.Tensor|List[str]|List[os.PathLike]):
        '''
        args:
            - imgs: [H,W,C] numpy arrays (untested), [B,C,H,W] torch tensors (untested) or list of path to the img
        '''
        yolo_results = self.yolo.predict(imgs, **self.yolo_infer_para)
        marker_detection = []
        for i,result in enumerate(yolo_results):
            if result.boxes is None:
                print(f"not object detected in the {i}-th image {result.path if result.path else ''}")
                continue
            if result.keypoints is None:
                print(f"not keypoints detected in the {i}-th image {result.path if result.path else ''}")
                continue 
            keypoints:np.ndarray = result.keypoints.cpu().numpy().xy   # type: ignore
            boxes = result.boxes.cpu().xyxy
            num_obj = boxes.shape[0]
            fine_markers = []
            for i in range(num_obj):
                crop_img = save_one_box(xyxy=boxes[i],im=result.orig_img,BGR=True,gain=1,pad=0,save=False)
                base_point = np.asanyarray(boxes[i][:2].reshape(1,2))
                markers = self.markerDetector(crop_img, **self.detection_para)
                markers[:,:2] = markers[:,:2] + base_point
                corse_points = keypoints[i]
                fine_index = find_nearest_points_index(corse_points[:,:2], markers[:,:2])
                fine_markers.append(markers[fine_index])
            fine_markers = np.concatenate(fine_markers, axis=0) if fine_markers else np.empty((0, 3), dtype=np.float32)
            marker_detection.append(fine_markers)
        return marker_detection, yolo_results
            
    def videoInfer(self,video, show_new_video:Optional[str]=None): # TODO: 直接用yolo的video inference 会提高效率吗?  
        """
        Perform inference on a video file or stream.
        Args:
            video: Path to video file or integer for webcam.
        Returns:
            marker_detection_per_frame: List of marker detections per frame.
            bbox_point_per_frame: List of YOLO results per frame.
        """
        def extect_form_yolo_result(yolo_result:Results):
            return {
                'bbox':yolo_result.boxes.cpu().xyxy.tolist(), # type: ignore
                'point':yolo_result.keypoints.cpu().xy.tolist() # type: ignore
            }
        cap = cv2.VideoCapture(video)
        marker_detection_per_frame = []
        bbox_point_per_frame = []
        if show_new_video is None:
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                if len(frames) == self.video_batch_size:
                    marker_detections, yolo_results = self.ImgArraysBatchInfer(frames)
                    marker_detection_per_frame.extend(marker_detections)
                    bbox_point_per_frame.extend([extect_form_yolo_result(yr) for yr in yolo_results])
                    del frames
                    frames = []
            # 处理剩余不足一个batch的帧
            if frames:
                marker_detections, yolo_results = self.ImgArraysBatchInfer(frames)
                marker_detection_per_frame.extend(marker_detections)
                bbox_point_per_frame.extend([extect_form_yolo_result(yr) for yr in yolo_results])
            del frames
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(show_new_video, fourcc, fps, (width, height))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                if len(frames) == self.video_batch_size:
                    marker_detections, yolo_results = self.ImgArraysBatchInfer(frames)
                    marker_detection_per_frame.extend(marker_detections)
                    bbox_point = []
                    for marker_detection, result in zip(marker_detections,yolo_results):
                        show = result.plot(kpt_radius=2, line_width=1)
                        show = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
                        # print(finedtection)
                        for det in marker_detection:
                            x, y = int(det[0]), int(det[1])
                            cv2.circle(show, (x, y), 4, (0, 255, 0), 2)
                        out.write(show)
                        bbox_point.append(extect_form_yolo_result(result))
                    # bbox_point_per_frame.extend([extect_form_yolo_result(yr) for yr in yolo_results])
                    bbox_point_per_frame.extend(bbox_point)
                    del frames
                    frames = []
            # 处理剩余不足一个batch的帧
            if frames:
                marker_detections, yolo_results = self.ImgArraysBatchInfer(frames)
                marker_detection_per_frame.extend(marker_detections)
                bbox_point = []
                for marker_detection, result in zip(marker_detections,yolo_results):
                    show = result.plot(kpt_radius=2, line_width=1)
                    show = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
                    # print(finedtection)
                    for det in marker_detection:
                        x, y = int(det[0]), int(det[1])
                        cv2.circle(show, (x, y), 4, (0, 255, 0), 2)
                    out.write(show)
                    bbox_point.append(extect_form_yolo_result(result))
                # bbox_point_per_frame.extend([extect_form_yolo_result(yr) for yr in yolo_results])
                bbox_point_per_frame.extend(bbox_point)
            del frames
            out.release()
        cap.release()
        
        return marker_detection_per_frame, bbox_point_per_frame

    


# TODO 启动dataclass增加安全性    
@dataclass
class HarrisPara:
    blockSize: int = 9
    ksize: int = 3
    k: float = 0.04

@dataclass
class HarrisDilate:
    kernel_size: int = 3
    interations: int = 2

@dataclass
class OtherDict:
    sub_pixcel_window_size: int = 9
    harris_dilate: HarrisDilate = field(default_factory=HarrisDilate)

@dataclass
class DetectionPara:
    ts: float = 0.2
    detectPara_Harris: HarrisPara = field(default_factory=HarrisPara)
    detectPara_other_dict: OtherDict = field(default_factory=OtherDict)
DEFAULT_HARRIS_PARA_DATACLASS = HarrisPara(blockSize=9, ksize=3, k=0.04)
DEFAULT_HARRIS_DILATE_DATACLASS = HarrisDilate(kernel_size=3, interations=2)
DEFAULT_OTHER_PARA_DATACLASS = OtherDict(
        sub_pixcel_window_size=9,
        harris_dilate=DEFAULT_HARRIS_DILATE_DATACLASS
    )
DEFAULT_DETECTION_PARA_DATACLASS = DetectionPara(
        ts=0.2,
        detectPara_Harris=DEFAULT_HARRIS_PARA_DATACLASS,
        detectPara_other_dict=DEFAULT_OTHER_PARA_DATACLASS
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="when run directly, take a image and return detection result for parameter turning")
    parser.add_argument('-s','--source', type=str,help="source image path")
    parser.add_argument('-o','--output',type=str,default=None,required=False, help="save detection here if given")
    args = parser.parse_args()

    img = cv2.imread(args.source)
    if img is None:
        raise FileNotFoundError(f"Image not found: {args.source}")
    result = cornerDetect(img, **DEFAULT_DETECTION_PARA)
    print(result.shape)
    print(result)

    tmax,tmin = result[:,2].max(),result[:,2].min()
    trange = tmax -tmin
    for x, y, trust in result:
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), 2)
        cv2.putText(img, f"{trust:.2e}", (int(x)+10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if args.output is not None:
        print(args.output)
        
        cv2.imwrite(args.output, img)
    