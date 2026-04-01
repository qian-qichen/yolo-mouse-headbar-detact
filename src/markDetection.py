import cv2
import numpy as np
from typing import Optional, List, Callable, Protocol, Tuple, Dict
from copy import deepcopy
import argparse
from ultralytics.models.yolo import YOLO
from ultralytics.engine.results import Results
import numpy as np
import torch
from ultralytics.utils import ops 
import os
from dataclasses import dataclass, field, asdict
import tempfile
from pathlib import Path
import json
from tqdm import tqdm
from .util import VIDEO_EXTENSIONS, YOLO_KF_PROFIE
from .util.cilHelp import load_cli_args
from .util.helper import load_yaml_as_dataclass, to_jsonable
from scipy.sparse.csgraph import connected_components
import line_profiler
import logging
logger = logging.getLogger(__name__)

@dataclass
class YOLOInferPAra:
    iou: float = 0.7
    imgsz: Tuple[int,int] = (3840, 2176)
    conf: float = 0.5
    verbose: bool = False
    augment: bool = False
    stream: bool = False
    roi: tuple[int,int,int,int]|None = None  # leftx,topy,rightx,downy
@dataclass
class HarrisPara:
    blockSize: int = 9
    ksize: int = 3
    k: float = 0.04

@dataclass
class HarrisDilate:
    kernel_size: int = 3
    interations: int = 0

@dataclass
class OtherPara:
    sub_pixcel_window_size: int = 5
    harris_dilate: HarrisDilate = field(default_factory=HarrisDilate)

@dataclass
class DetectionPara:
    ts: float = 0.0
    detectPara_Harris: HarrisPara = field(default_factory=HarrisPara)
    detectPara_other_dict: OtherPara = field(default_factory=OtherPara)



@dataclass
class BallDetectionPara:
    r_min: int = 6
    r_max: int = 16
    circles_expected: int = 2 
    minDist: int = 25
    threshold_max: int = 100
    threshold_min: int = 1
    dp_max: float = 12.0
    r_step: int = 3
    dp_step: float = 0.25
    threshold_step: int = 2 
    r_range: int = 3
    canny_upThreshHold: int = 100
    gaussian_filter_kernel_size: int = 3
    strach_light: str | bool = False # 'linear' or False
    H_ranges: Tuple[Tuple[int, int], ...] = ((0, 180),)
    pick_target: bool = True
    filtering_method: str = 'hue_std' # In ['hue_std', 'canny_edge']  ## TODO plan to support ['hue_std', 'canny_edge', 'gauss_filter', 'gauss_filter_saperate', 'center_position']

DEFAULT_INFER_PARA = {
    'iou':0.4,
    'imgsz':(3840, 2176),
    'conf':0.5,
}
DEFAULT_YOLO_INFER_PARA = YOLOInferPAra(iou=0.4, imgsz=(3840, 2176), conf=0.5)
DEFAULT_HARRIS_PARA = HarrisPara(blockSize=9, ksize=3, k=0.04)
DEFAULT_HARRIS_DILATE = HarrisDilate(kernel_size=3, interations=2)
DEFAULT_OTHER_PARA = OtherPara(
        sub_pixcel_window_size=5,
        harris_dilate=DEFAULT_HARRIS_DILATE
    )
DEFAULT_DETECTION_PARA = DetectionPara(
        ts=0.15,
        detectPara_Harris=DEFAULT_HARRIS_PARA,
        detectPara_other_dict=DEFAULT_OTHER_PARA
    )

DEFAULT_BALL_DETECTION_PARA = BallDetectionPara()

def clip_image_with_minSize(xywh,im:np.ndarray,min_wh,):
    # xywh[2:] = torch.maximum(xywh[2:], min_wh)

    cx, cy, w, h = float(xywh[0]), float(xywh[1]), max(float(xywh[2]), min_wh[0]), max(float(xywh[3]),min_wh[1])
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    # import pdb;pdb.set_trace()
    H, W = im.shape[0], im.shape[1]
    # clip 到图像边界
    x1i = max(0, int(round(x1)))
    y1i = max(0, int(round(y1)))
    x2i = min(W, int(round(x2)))
    y2i = min(H, int(round(y2)))

    pad_w = max(0, int(w)-(x2i-x1i))
    pad_h = max(0, int(h)-(y2i-y1i))

    crop = im[ y1i : y2i,x1i :x2i,:]
    if pad_w > 0 or pad_h > 0:
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        crop = cv2.copyMakeBorder(crop,top=top,bottom=bottom,left=left,right=right,borderType=cv2.BORDER_REPLICATE,value=[0, 0, 0],dst=None)


    return crop,(x1i,y1i,x2i,y2i)



def cornerDetect(img,ts=0.01,detectPara_Harris:dict=DEFAULT_HARRIS_PARA,detectPara_other_dict:dict=DEFAULT_OTHER_PARA)->np.ndarray:
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
    # dst = harris_response
    hmax,hmin = harris_response.max(),harris_response.min()
    ret, dst = cv2.threshold(harris_response,ts*(hmax - hmin) + hmin,255,0)
    dst = np.uint8(dst)
    dst = cv2.dilate(dst,kernel=np.ones((harris_dilate_kernel_size, harris_dilate_kernel_size), np.uint8),iterations=harris_dilate_interations)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    centroids = centroids[1:].astype(np.float32)
    corners = np.ones((centroids.shape[0],3),dtype=np.float32)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    
    try:
        corners[:,0:2] = cv2.cornerSubPix(gray,centroids,(sub_pixcel_window_size,sub_pixcel_window_size),(-1,-1),criteria)
    except Exception as e:
        import pdb;pdb.set_trace()
    cornerInt = corners[:,0:2].round().astype(int)
    # corners[:,2] = harris_response[np.clip(cornerInt[:,1], 0, harris_response.shape[0]-1), np.clip(cornerInt[:,0], 0, harris_response.shape[1]-1)]
    corners[:,2] = harris_response[np.clip(cornerInt[:,1], 0, harris_response.shape[0]-1), np.clip(cornerInt[:,0], 0, harris_response.shape[1]-1)]

    return corners


def circle_para_distance(circle1, circle2, r_weight=1):
    """计算两对圆参数空间上的距离"""
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 +  r_weight * (r1 - r2) ** 2)

def circle_pair_para_distance(pair1, pair2, r_weight=1):
    """计算两对圆参数空间上的距离，返回两圆距离的平均"""
    c1_1, c1_2 = pair1
    c2_1, c2_2 = pair2
    d1 = circle_para_distance(c1_1, c2_1, r_weight) + circle_para_distance(c1_2, c2_2, r_weight)
    d2 = circle_para_distance(c1_1, c2_2, r_weight) + circle_para_distance(c1_2, c2_1, r_weight)
    return min(d1, d2)

def average_same_circle_pairs(data, distMat, threshold=1e-3):
    """
    merge same circle pairs as one
    """
    isLower = distMat < threshold
    n_components, labels = connected_components(isLower,directed=False)
    item_shape = data[0].shape
    new_data = np.empty((n_components, *item_shape))
    sim_count = np.empty(n_components)
    for i in range(n_components):
        new_data[i] = np.mean(data[labels == i], axis=0)
        sim_count[i] = np.sum(labels == i)
    return new_data, sim_count

def pick_detection(detected_circles, n=2, threshold=1e-3):
    assert n==2, "Only support n=2"    
    distances = np.zeros((len(detected_circles), len(detected_circles)))
    for i in range(len(detected_circles)):
        for j in range(i + 1, len(detected_circles)):
            distances[i, j] = circle_pair_para_distance(detected_circles[i], detected_circles[j])
            distances[j, i] = distances[i, j]

    merged_circles, similarity_counts = average_same_circle_pairs(detected_circles, distances, threshold=threshold)
    max_index = np.argmax(similarity_counts)
    most_similar_circle = merged_circles[max_index]
    return most_similar_circle




@line_profiler.profile
def hough_circle_AUTOsearch(img: np.ndarray,
                            r_min: int = 1,
                            r_max:int = 101,
                            circles_expected: int = 1,
                            minDist: int = 2,
                            threshold_max: int = 300, # some blog suggest less than 300
                            threshold_min: int = 1,
                            dp_max: float = 12,
                            r_step: int = 3,
                            dp_step: float = 0.25,
                            threshold_step: int = 2,
                            r_range: int = 3,
                            canny_upThreshHold: int = 150,
                            H_ranges: Tuple[Tuple[int, int], ...] = ((0, 180),),
                            pick_target: bool = True,
                            gaussian_filter_kernel_size: int = 3,
                            strach_light: str | bool = False,
                            filtering_method: str = 'hue_std',
                            # logger: logging.Logger = logger
                            ):
    """
    auto guess hough circle detection parameters dp, radius, threshold
    This function is tuned for single-circle detection in each cropped ROI.
    args:
    img: GBR image
    return:a list of((np.array of [x, y, r], dp, radius, threshold)
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (gaussian_filter_kernel_size, gaussian_filter_kernel_size), 0)
    if strach_light:
        assert isinstance(strach_light, str), "If strach_light is True, it should be a string indicating the method to strach light, e.g., 'linear'"
        if strach_light == 'linear':
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        else:
            raise NotImplementedError(f"Strach light method {strach_light} is not implemented yet")
    detected_circles = []
    detected_paras = []
    threshold_guess = threshold_max # start with threshold_max
    while threshold_guess > threshold_min:
        dp_guess = 1.0 # dp all start with 1.0
        while dp_guess < dp_max:
            r_guess = r_max
            while True:
                logger.debug(f"dp_guess: {dp_guess}, r_guess: {r_guess}, threshold_guess: {threshold_guess}")
                # print(f"dp_guess: {dp_guess}, r_guess: {r_guess}, threshold_guess: {threshold_guess}")
                circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT, dp=dp_guess,minDist=minDist,param1=canny_upThreshHold,param2=threshold_guess,minRadius=r_guess-r_range,maxRadius=r_guess+r_range) # TODO try cv2.HOUGH_GRADIENT_ALT
                if circles is not None:
                    # Normalize to shape (-1, 3) even when there is 1 circle
                    # circles_arr = np.array(circles, dtype=np.float32).reshape(-1, 3)
                    circles_arr = circles.reshape(-1, 3)
                    for circle in circles_arr:
                        detected_circles.append(circle)
                        detected_paras.append((dp_guess, r_guess, threshold_guess))
                    break
                r_guess = r_guess - r_step
                if r_guess < r_min:
                    break
            dp_guess = dp_guess + dp_step
        threshold_guess = threshold_guess - threshold_step
    detected_circles = np.vstack(detected_circles).reshape(-1, 3)
    if detected_circles.shape[0] == 0:
        return None, None, None, None
    else:
        if filtering_method == 'hue_std':    
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h_channel = hsv[..., 0]
            
            h_stds = []
            h_means = []
            for x, y, r in detected_circles:
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (int(x), int(y)), int(r), 1, thickness=-1)
                h_values = h_channel[mask == 1]
                if h_values.size == 0:
                    h_stds.append(np.inf)
                    h_means.append(np.inf)
                else:
                    h_stds.append(float(np.std(h_values)))
                    h_means.append(float(np.mean(h_values)))
            h_means = np.asarray(h_means).squeeze()
            passed = np.zeros_like(h_means, dtype=bool)
            for h_range in H_ranges:
                passed |= ((h_means >= h_range[0]) & (h_means <= h_range[1]))
            filtered_circles = detected_circles[passed]
            h_stds = np.asarray(h_stds)[passed]
            if filtered_circles.shape[0] == 0:
                return detected_circles, detected_paras, None, None
            if pick_target:
                # Pick one circle with smallest hue std inside the circle mask.
                min_idx = int(np.argmin(h_stds))
                picked_circle = np.asarray(filtered_circles[min_idx:min_idx + 1], dtype=np.float32)
                return detected_circles, detected_paras, filtered_circles, picked_circle
            else:
                return detected_circles, detected_paras, filtered_circles, None
        elif filtering_method == 'canny_edge':
            low_thresh = int(canny_upThreshHold / 2)
            edges = cv2.Canny(gray, low_thresh, canny_upThreshHold)
            edge_on_edge_count = []
            ys, xs = np.nonzero(edges)
            if ys.size == 0:
                edge_on_edge_count = [0] * len(detected_circles)
            else:
                xs = xs.astype(np.float32)
                ys = ys.astype(np.float32)
                for x, y, r in detected_circles:
                    d2 = (xs - x)**2 + (ys - y)**2
                    edge_on_edge_count.append(int(np.count_nonzero((d2 <= r*r) & (d2 >= (r-1)*(r-1)))))

            # edge_on_edge_count
            edge_on_edge_count = np.asarray(edge_on_edge_count)
            mean = np.mean(edge_on_edge_count)
            std = np.std(edge_on_edge_count)
            thresh = mean - std # TODO parameterize this threshold
            passed = edge_on_edge_count >= thresh
            filtered_circles = detected_circles[passed]
            if filtered_circles.shape[0] == 0:
                return detected_circles, detected_paras, None, None
            if pick_target:
                max_idx = int(np.argmax(edge_on_edge_count))
                picked_circle = np.asarray(filtered_circles[max_idx:max_idx + 1], dtype=np.float32)
                return detected_circles, detected_paras, filtered_circles, picked_circle
        else:
            raise NotImplementedError(f"Filtering method {filtering_method} is not supported")    
        



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
    def __call__(self, img: np.ndarray, **kwargs) -> np.ndarray: ...


def extect_form_yolo_result(yolo_result:Results):
    bbox = yolo_result.boxes.cpu().xyxy.tolist() if yolo_result.boxes is not None else []  # type: ignore
    point = yolo_result.keypoints.cpu().xy.tolist() if yolo_result.keypoints is not None else []  # type: ignore
    return {
        'bbox': bbox,
        'point': point,
    }

class improved_yoloDetector:
    def __init__(self,yolo_model:YOLO|dict,yolo_infer_para:YOLOInferPAra=DEFAULT_YOLO_INFER_PARA, cores:int=8,video_batch_size:int=4):
        """
        :param yolo_model: the YOLO model or model config dict to load the model
        :type yolo_model: YOLO | dict
        :param yolo_infer_para: dataclass of YOLO inference parameters
        :type yolo_infer_para: YOLOInferPAra
        :param cores: number of CPU cores to use for marker detection, multi-core acceleration not implemented yet
        :type cores: int
        :param video_batch_size: batch size for video inference
        :type video_batch_size: int
        """
        if isinstance(yolo_model, YOLO):
            self.yolo = yolo_model
        else:
            ck_path = yolo_model.pop('ck_path')
            self.yolo = YOLO(**yolo_model).load(ck_path).to('cuda')
        self.roi = yolo_infer_para.roi
        self.yolo_infer_para = asdict(yolo_infer_para)
        self.yolo_infer_para.pop('roi',None)
        
        self.video_batch_size = video_batch_size

        max_core = os.cpu_count()
        assert isinstance(max_core,int)
        self.cores = min(cores, max_core) # TODO: add multi-processing optimization on marker detection
        if self.cores < cores:
            print(f"only use {self.cores} for marker detection due to hardware limitation")

    def _set_mask(self, imgW,imgH,leftx,topy,rightx,downy):
        """
        set mask for yolo detection, intended to apply on the entire video so yolo focus on specific region (on original image size)
        
        :param imgW: width of the image
        :param imgH: height of the image
        :param leftx: left x coordinate of the ROI
        :param topy: top y coordinate of the ROI
        :param rightx: right x coordinate of the ROI
        :param downy: down y coordinate of the ROI
        """
        # normalize indices to integers and clamp to image bounds
        try:
            leftx = int(round(float(leftx))) if leftx is not None else 0
            topy = int(round(float(topy))) if topy is not None else 0
            rightx = int(round(float(rightx))) if rightx is not None else int(imgW)
            downy = int(round(float(downy))) if downy is not None else int(imgH)
        except Exception:
            raise ValueError(f"Invalid ROI coordinates: {leftx},{topy},{rightx},{downy}")

        leftx = max(0, min(int(imgW), leftx))
        rightx = max(0, min(int(imgW), rightx))
        topy = max(0, min(int(imgH), topy))
        downy = max(0, min(int(imgH), downy))

        if leftx >= rightx or topy >= downy:
            raise ValueError(f"Invalid ROI after clamping: {leftx},{topy},{rightx},{downy}")

        # normalize indices to integers and clamp to image bounds
        try:
            leftx = int(round(float(leftx))) if leftx is not None else 0
            topy = int(round(float(topy))) if topy is not None else 0
            rightx = int(round(float(rightx))) if rightx is not None else int(imgW)
            downy = int(round(float(downy))) if downy is not None else int(imgH)
        except Exception:
            raise ValueError(f"Invalid ROI coordinates: {leftx},{topy},{rightx},{downy}")

        leftx = max(0, min(int(imgW), leftx))
        rightx = max(0, min(int(imgW), rightx))
        topy = max(0, min(int(imgH), topy))
        downy = max(0, min(int(imgH), downy))

        if leftx >= rightx or topy >= downy:
            raise ValueError(f"Invalid ROI after clamping: {leftx},{topy},{rightx},{downy}")

        mask = np.zeros((imgH, imgW, 1), dtype=np.uint8)
        mask[topy:downy, leftx:rightx] = 1
        self.mask = mask
        return mask
   

class MarkerImproved_yoloDetector:
    """
    A marker detector that combines YOLO object detection with a fine marker detection algorithm.
    supports ROI mask in format [leftx,topy,rightx,downy] (on orginal image size)
    """
    def __init__(self,yolo_model:YOLO|dict,yolo_infer_para:YOLOInferPAra=DEFAULT_YOLO_INFER_PARA,markerDetector:MarkerDetectorProtocol=cornerDetect, detection_para:DetectionPara=DEFAULT_DETECTION_PARA, target_id_h_ranges: Optional[Dict[int, Tuple[int, int]]] = None, cores:int=8,video_batch_size:int=4):
        """
        :param yolo_model: the YOLO model or model config dict to load the model
        :type yolo_model: YOLO | dict
        :param yolo_infer_para: dataclass of YOLO inference parameters
        :type yolo_infer_para: YOLOInferPAra
        :param markerDetector: the fine marker detector function, obeys MarkerDetectorProtocol
        :type markerDetector: MarkerDetectorProtocol
        :param detection_para: dataclass of fine marker detection parameters
        :type detection_para: DetectionPara
        :param cores: number of CPU cores to use for marker detection, multi-core acceleration not implemented yet
        :type cores: int
        :param video_batch_size: batch size for video inference
        :type video_batch_size: int
        """
        if isinstance(yolo_model, YOLO):
            self.yolo = yolo_model
        else:
            ck_path = yolo_model.pop('ck_path')
            self.yolo = YOLO(**yolo_model).load(ck_path).to('cuda')
        self.roi = yolo_infer_para.roi
        self.yolo_infer_para = asdict(yolo_infer_para)
        self.yolo_infer_para.pop('roi',None)
        
        self.markerDetector = markerDetector
        
        self.detection_para = asdict(detection_para)
        sub_pixcel_window_size = detection_para.detectPara_other_dict.sub_pixcel_window_size
        self.min_wh = (2 * sub_pixcel_window_size + 5, 2 * sub_pixcel_window_size + 5)
        self.video_batch_size = video_batch_size

        max_core = os.cpu_count()
        assert isinstance(max_core,int)
        self.cores = min(cores, max_core) # TODO: add multi-processing optimization on marker detection
        if self.cores < cores:
            print(f"only use {self.cores} for marker detection due to hardware limitation")

    def _set_mask(self, imgW,imgH,leftx,topy,rightx,downy):
        """
        set mask for yolo detection, intended to apply on the entire video so yolo focus on specific region (on original image size)
        
        :param imgW: width of the image
        :param imgH: height of the image
        :param leftx: left x coordinate of the ROI
        :param topy: top y coordinate of the ROI
        :param rightx: right x coordinate of the ROI
        :param downy: down y coordinate of the ROI
        """
        mask = np.zeros((imgH, imgW, 1), dtype=np.uint8)
        mask[topy:downy, leftx:rightx] = 1
        self.mask = mask
        return mask

    def ImgArraysBatchInfer(self,imgs:np.ndarray|torch.Tensor|List[np.ndarray],topK:Optional[int]=None, apply_mask:bool=False):
        """
        Perform batch inference on a list of images or image arrays, returning fine marker detections and YOLO results.
        
        :param imgs: [B,H,W,C] numpy arrays (untested), [B,C,H,W] torch tensors (untested) or list of path to the img
        :type imgs: np.ndarray | torch.Tensor | List[np.ndarray]
        :param topK: Number of top confident (yolo infer confidence) detections to keep per image. If None, all detections are kept.
        :type topK: int | None
        :param apply_mask: whether to apply the ROI mask to the input images before inference. Default is False.
        :type apply_mask: bool
        """
        if apply_mask and not hasattr(self, 'mask'):
            raise ValueError("apply_mask=True but ROI mask is not initialized; call _set_mask first or use videoInfer/videoInferShow with apply_mask enabled.")

        # apply mask and check input type
        if apply_mask:
            if isinstance(imgs, list):
                imgs = [img*self.mask for img in imgs]
            elif isinstance(imgs, np.ndarray):
                if imgs.ndim == 4:
                    imgs = imgs * self.mask[None,...]
                elif imgs.ndim ==3:
                    imgs = imgs * self.mask
                else:
                    raise ValueError("imgs should be 3 or 4 dimensioned when numpy array")
            elif isinstance(imgs, torch.Tensor):
                if imgs.ndim == 4:
                    mask_tensor = torch.from_numpy(self.mask).to(imgs.device)
                    imgs = imgs * mask_tensor[None,...].permute(0,3,1,2)
                elif imgs.ndim ==3:
                    mask_tensor = torch.from_numpy(self.mask).to(imgs.device)
                    imgs = imgs * mask_tensor.permute(2,0,1)
                else:
                    raise ValueError("imgs should be 3 or 4 dimensioned when torch tensor")
            else:
                raise TypeError("imgs should be numpy array, torch tensor or list of ndarray to the img")
        else :
            if isinstance(imgs, list):
                pass
            elif isinstance(imgs, np.ndarray) or isinstance(imgs, torch.Tensor):
                if imgs.ndim not in [3,4]:
                    raise ValueError("imgs should be 3 or 4 dimensioned")
            else:
                raise TypeError("imgs should be numpy array, torch tensor or list of ndarray to the img")

        yolo_results = self.yolo.predict(imgs, **self.yolo_infer_para)
        # yolo_results = self.yolo.track(imgs,stream=True, persist=True,tracker=YOLO_KF_PROFIE, **self.yolo_infer_para)
        marker_detection = []
        for i,result in enumerate(yolo_results):
            if result.boxes is None:
                print(f"not object detected in the {i}-th image {result.path if result.path else ''}")
                continue
            if result.keypoints is None:
                print(f"not keypoints detected in the {i}-th image {result.path if result.path else ''}")
                continue
            # 获取置信度并排序，选择topK
            
            confs = result.boxes.conf.cpu().numpy()
            if topK is not None and len(confs) > topK:
                top_indices = np.argsort(confs)[-topK:][::-1]
                result.boxes = result.boxes[top_indices.copy()]
                result.keypoints = result.keypoints[top_indices.copy()]
                boxes = result.boxes.xywh.cpu().numpy()
                keypoints = result.keypoints.cpu().numpy().xy
            else:
                boxes = result.boxes.xywh.cpu().numpy()
                keypoints = result.keypoints.cpu().numpy().xy
            # keypoints:np.ndarray = result.keypoints.cpu().numpy().xy   # type: ignore
  
            num_obj = boxes.shape[0]
            fine_markers = []
            for i in range(num_obj):
                crop_img,left_top_right_down = clip_image_with_minSize(xywh=boxes[i],im=result.orig_img,min_wh=self.min_wh)
                # print(crop_img.shape)
                # base_point = np.asanyarray(boxes[i][:2].reshape(1,2).cpu())
                base_point = np.asarray(left_top_right_down[:2])
                markers = self.markerDetector(crop_img, **self.detection_para)

                markers[:,:2] = markers[:,:2] + base_point
                corse_points = keypoints[i]
                fine_index = find_nearest_points_index(corse_points[:,:2], markers[:,:2])
                fine_markers.append(markers[fine_index])
            fine_markers = np.concatenate(fine_markers, axis=0) if fine_markers else np.empty((0, 3), dtype=np.float32)
            marker_detection.append(fine_markers)
            if len(fine_markers) > topK:
                import pdb;pdb.set_trace()
        return marker_detection, yolo_results

    def videoInferShow(self,video, show_new_video,topK:Optional[int]=None,apply_mask:Optional[bool|List[int,int,int,int]]=False):
        """
        对输入视频进行分批推理并将可视化结果写入新视频，同时返回每帧的标记检测与从 YOLO 结果提取的边界/关键点信息。
        此方法按 self.video_batch_size 将视频帧分批送入 self.ImgArraysBatchInfer 进行推理，随后将推理得到的可视化结果（通过 result.plot(...)）与检测到的标记点绘制到输出视频中，并收集每帧的检测数据与从 YOLO 结果提取的信息。
        Args:
            video (str): 输入视频文件的路径。
            show_new_video (str): 输出可视化视频文件的路径（帧率与输入视频一致，分辨率与输入一致）。
        Returns:
            tuple: 包含两个元素的元组 (marker_detection_per_frame, bbox_point_per_frame)：
                marker_detection_per_frame (list): 按帧顺序的标记检测结果列表。每个元素对应一帧，通常为该帧检测到的标记列表；每个标记至少包含可索引的坐标信息 (例如 [x, y, ...])。
                bbox_point_per_frame (list): 按帧顺序的从 YOLO 结果提取的边界框/关键点信息列表。每个元素的具体结构由 extect_form_yolo_result(result) 决定。
        Raises:
            RuntimeError: 当无法打开输入视频或无法创建/写入输出视频文件时，可能导致后续操作失败并抛出运行时错误。
            ValueError: 当相关配置（例如 self.video_batch_size）非法时，函数可能无法按预期工作并抛出值错误。
        Notes:
            - 本方法在内存中一次缓存最多 self.video_batch_size 帧以进行批量推理；若 video_batch_size 过大或视频过长，可能导致较高的内存占用。
            - 函数会在处理完成后释放 VideoWriter 和 VideoCapture 资源（out.release(), cap.release()）。
            - 可视化过程中使用 result.plot(...) 获得的图像可能为 RGB，需要根据上下文决定是否转换为 BGR 写入视频（代码中对部分批次进行了颜色转换，注意一致性）。
            - marker_detection_per_frame 与 bbox_point_per_frame 的帧数量应与输入视频的帧数对应（除非视频无法读取或提前出错）。
        """

        cap = cv2.VideoCapture(video)
        marker_detection_per_frame = []
        bbox_point_per_frame = []
        fourcc = cv2.VideoWriter_fourcc(*'avc1') # type: ignore
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(show_new_video, fourcc, fps, (width, height))

        if isinstance(apply_mask, list) and len(apply_mask) ==4:
            leftx, topy, rightx, downy = apply_mask
            self._set_mask(width,height,leftx,topy,rightx,downy)
            apply_mask = True
        elif apply_mask is True and self.roi is not None:
            leftx, topy, rightx, downy = self.roi
            self._set_mask(width,height,leftx,topy,rightx,downy)


        frames = []
        def _iner_batch_infer_and_write(frames_batch, out_writer):
                marker_detections, yolo_results = self.ImgArraysBatchInfer(frames_batch, topK=topK, apply_mask=apply_mask)
                marker_detection_per_frame.extend(marker_detections)
                bbox_point = []
                for marker_detection, result in zip(marker_detections,yolo_results):
                    show = result.plot(kpt_radius=2, line_width=1)
                    for det in marker_detection:
                        x, y = int(det[0]), int(det[1])
                        cv2.circle(show, (x, y), 3, (0, 255, 0), 1) # color in BGR, now is pure green
                    out_writer.write(show)
                    bbox_point.append(extect_form_yolo_result(result))
                bbox_point_per_frame.extend(bbox_point)
                
        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                if len(frames) == self.video_batch_size:
                    _iner_batch_infer_and_write(frames, out)
                    del frames
                    frames = []
                    pbar.update(self.video_batch_size)
            if frames: # the rest frames
                _iner_batch_infer_and_write(frames, out)
                pbar.update(len(frames))
            del frames

        out.release()
        cap.release()
        
        return marker_detection_per_frame, bbox_point_per_frame     
    
    def videoInferGenerater(self,video, batch_size=None,topK=None,apply_mask:Optional[bool|List[int,int,int,int]]=False): # TODO: 直接用yolo的video inference 会提高效率吗?  
        """
        Perform inference on a video file or stream.
        Args:
            video: Path to video file or integer for webcam.
        Returns:
            marker_detection_per_frame: List of marker detections per frame.
            bbox_point_per_frame: List of YOLO results per frame.
        """
        if batch_size is None:
            batch_size = self.video_batch_size

        cap = cv2.VideoCapture(video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if isinstance(apply_mask, list) and len(apply_mask) ==4:
            leftx, topy, rightx, downy = apply_mask
            self._set_mask(width,height,leftx,topy,rightx,downy)
            apply_mask = True    
        elif apply_mask is True and self.roi is not None:
            leftx, topy, rightx, downy = self.roi
            self._set_mask(width,height,leftx,topy,rightx,downy)
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if len(frames) == batch_size:
                marker_detections, yolo_results = self.ImgArraysBatchInfer(frames,topK,apply_mask=apply_mask)
                del frames
                frames = []
                bbox_point_batch = [extect_form_yolo_result(yr) for yr in yolo_results]
                yield marker_detections, bbox_point_batch
        # 处理剩余不足一个batch的帧
        if frames:
            marker_detections, yolo_results = self.ImgArraysBatchInfer(frames,topK,apply_mask=apply_mask)
            bbox_point_batch = [extect_form_yolo_result(yr) for yr in yolo_results]
            del frames
            cap.release()
            yield marker_detections, bbox_point_batch
        cap.release()
    
    def multiVideoInferGenetater(self,videos:Dict[str,str],batch_size=None,topK=None,apply_mask:Optional[bool|List[int,int,int,int]]=False):
        """
        Perform inference on a video file or stream.
        Args:
        - video: dict like {name of the video:path to the file}
        Returns:
        - dict{name of the video: detection result}, the detection result refers to:
            - marker_detection_per_frame: List of marker detections per frame.
            - bbox_point_per_frame: List of YOLO results per frame.
        """
        generators = {name:self.videoInferGenerater(path,batch_size=batch_size,topK=topK,apply_mask=apply_mask) for name,path in videos.items()}
        finished = set()
        out = {}
        while len(finished) < len(generators):
            for name, gen in generators.items():
                if name in finished:
                    continue
                try:
                    marker, yolo_out = next(gen)
                    out[name] = {
                        'marker':marker,
                        'bbox_points_yolo':yolo_out,
                    }                    
                except StopIteration:
                    finished.add(name)
            yield out

    def videoInfer(self, video,topK=None,apply_mask:Optional[bool|List[int,int,int,int]]=False):
        """infer in one go"""
        # marker_detection_per_frame = []
        # bbox_point_per_frame = []
        # if batch_size is None:
        #     batch_size = self.video_batch_size
        
        # cap = cv2.VideoCapture(video)
        # video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # cap.release()
        # with tqdm(total=video_length, desc="Processing video") as pbar:    
        #     for marker_detections, bbox_point_batch in self.videoInferGenerater(video,batch_size=batch_size,topK=topK,apply_mask=apply_mask):
        #         marker_detection_per_frame.extend(marker_detections)
        #         pbar.update(batch_size)
        #     bbox_point_per_frame.extend(bbox_point_batch)
        #     pbar.update(len(bbox_point_batch))
        cap = cv2.VideoCapture(video)
        marker_detection_per_frame = []
        bbox_point_per_frame = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if isinstance(apply_mask, list) and len(apply_mask) ==4:
            leftx, topy, rightx, downy = apply_mask
            self._set_mask(width,height,leftx,topy,rightx,downy)
            apply_mask = True
        elif apply_mask is True and self.roi is not None:
            leftx, topy, rightx, downy = self.roi
            self._set_mask(width,height,leftx,topy,rightx,downy)

        frames = []
        def _iner_batch_infer(frames_batch):
                marker_detections, yolo_results = self.ImgArraysBatchInfer(frames_batch, topK=topK, apply_mask=apply_mask)
                marker_detection_per_frame.extend(marker_detections)
                bbox_point = []
                for marker_detection, result in zip(marker_detections,yolo_results):
                    bbox_point.append(extect_form_yolo_result(result))
                bbox_point_per_frame.extend(bbox_point)
                
        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                if len(frames) == self.video_batch_size:
                    _iner_batch_infer(frames)
                    del frames
                    frames = []
                    pbar.update(self.video_batch_size)
            if frames: # the rest frames
                _iner_batch_infer(frames)
                pbar.update(len(frames))
            del frames

        cap.release()
                
        return marker_detection_per_frame, bbox_point_per_frame

    # def videosInferFenerator(self,videos,batch_size=None):
    #     if batch_size is None:
    #         batch_size = self.video_batch_size

class BallImproved_yoloDetector:
    # now support box only yolo
    # TODO：完善这个推理工具

    def __init__(self, yolo_model,H_ranges:Dict=None, yolo_infer_para = DEFAULT_YOLO_INFER_PARA, ball_detect_para:BallDetectionPara = DEFAULT_BALL_DETECTION_PARA, min_wh=25, cores=8, video_batch_size=4):
        self.yoloDetector = improved_yoloDetector(yolo_model, yolo_infer_para, cores, video_batch_size)
        self.detection_para = asdict(ball_detect_para)
        self.min_wh = (min_wh, min_wh)
        if H_ranges is not None:
            self.h_ranges = H_ranges
        else:
            self.h_ranges = {}


    def ImgArraysBatchInfer(self,imgs:np.ndarray|torch.Tensor|List[np.ndarray], topK:Optional[int]=None, apply_mask:bool=False):
        """
        Perform batch inference on a list of images or image arrays, returning ball detections and YOLO results.
        :param imgs: [B,H,W,C] numpy arrays (untested), [B,C,H,W] torch tensors (untested) or list of path to the img
        :type imgs: np.ndarray | torch.Tensor | List[np.ndarray]
        :param topK: Number of top confident (yolo infer confidence) detections to keep per image. If None, all detections are kept.
        :type topK: int | None
        :param apply_mask: whether to apply the ROI mask to the input images before inference. Default is False.
        :type apply_mask: bool
        """
        if apply_mask and not hasattr(self.yoloDetector, 'mask'):
            raise ValueError("apply_mask=True but ROI mask is not initialized; call _set_mask first")

        # apply mask and check input type
        if apply_mask:
            if isinstance(imgs, list):
                imgs = [img * self.yoloDetector.mask for img in imgs]
            elif isinstance(imgs, np.ndarray):
                if imgs.ndim == 4:
                    imgs = imgs * self.yoloDetector.mask[None,...]
                elif imgs.ndim ==3:
                    imgs = imgs * self.yoloDetector.mask
                else:
                    raise ValueError("imgs should be 3 or 4 dimensioned when numpy array")
            elif isinstance(imgs, torch.Tensor):
                if imgs.ndim == 4:
                    mask_tensor = torch.from_numpy(self.yoloDetector.mask).to(imgs.device)
                    imgs = imgs * mask_tensor[None,...].permute(0,3,1,2)
                elif imgs.ndim ==3:
                    mask_tensor = torch.from_numpy(self.yoloDetector.mask).to(imgs.device)
                    imgs = imgs * mask_tensor.permute(2,0,1)
                else:
                    raise ValueError("imgs should be 3 or 4 dimensioned when torch tensor")
            else:
                raise TypeError("imgs should be numpy array, torch tensor or list of ndarray to the img")
        else :
            if isinstance(imgs, list):
                pass
            elif isinstance(imgs, np.ndarray) or isinstance(imgs, torch.Tensor):
                if imgs.ndim not in [3,4]:
                    raise ValueError("imgs should be 3 or 4 dimensioned")
            else:
                raise TypeError("imgs should be numpy array, torch tensor or list of ndarray to the img")

        yolo_results = self.yoloDetector.yolo.predict(imgs, **self.yoloDetector.yolo_infer_para)
        ball_detection = []
        for frame_i, result in enumerate(yolo_results):
            if result.boxes is None:
                print(f"not object detected in the {frame_i}-th image {result.path if result.path else ''}")
                ball_detection.append([])
                continue

            if topK is not None:
                uniq_labels = np.unique(result.boxes.cls.cpu().numpy())
                labels_index = {label: np.where(result.boxes.cls.cpu().numpy() == label)[0] for label in uniq_labels}
                labels_conf = {label: result.boxes.conf[indices].cpu().numpy() for label, indices in labels_index.items()}
                index_to_futher_infer = []
                for label, confs in labels_conf.items():
                    if len(confs) > topK:
                        top_inLabelRange_indices = np.argsort(confs)[-topK:][::-1]
                        index_to_futher_infer.extend(labels_index[label][top_inLabelRange_indices].tolist())
                    else:
                        index_to_futher_infer.extend(labels_index[label].tolist())
                result.boxes = result.boxes[index_to_futher_infer]
            frame_balls = []
            for obj_i in range(len(result.boxes)):
                bbox = result.boxes[obj_i]
                # import pdb;pdb.set_trace()
                crop_img, left_top_right_down = clip_image_with_minSize(
                    xywh=bbox.xywh.cpu().numpy().squeeze(),
                    im=result.orig_img,
                    min_wh=self.min_wh,
                )
                base_point = np.asarray(left_top_right_down[:2], dtype=np.float32)
                obj_name = result.names[int(result.boxes.cls[obj_i])] if result.boxes.cls is not None else None
                detection_para = deepcopy(self.detection_para)
                if obj_name is not None and obj_name in self.h_ranges:
                    detection_para['H_ranges'] = self.h_ranges[obj_name]

                _, _, _, ball = hough_circle_AUTOsearch(crop_img, **detection_para)
                if ball is None:
                    continue

                balls = np.asarray(ball, dtype=np.float32).reshape(-1, 3)
                balls[:, :2] = balls[:, :2] + base_point
                frame_balls.append({'balls': balls, 'class': obj_name})

            ball_detection.append(frame_balls)

        return ball_detection, yolo_results


    
    def videoInferShow(self, video, show_new_video, topK:Optional[int]=None, apply_mask:Optional[bool|List[int]]=False):
        """
        For BallImproved_yoloDetector: run inference on video and save visualization.

        Returns:
            (ball_detection_per_frame, bbox_point_per_frame)
        """
        cap = cv2.VideoCapture(video)
        ball_detection_per_frame = []
        bbox_point_per_frame = []
        fourcc = cv2.VideoWriter_fourcc(*'avc1') # type: ignore
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(show_new_video, fourcc, fps, (width, height))
        
        if isinstance(apply_mask, list) and len(apply_mask) == 4:
            leftx, topy, rightx, downy = apply_mask
            self.yoloDetector._set_mask(width, height, leftx, topy, rightx, downy)
            apply_mask = True
        elif apply_mask is True:
            if self.yoloDetector.roi is not None:
                leftx, topy, rightx, downy = self.yoloDetector.roi
                self.yoloDetector._set_mask(width, height, leftx, topy, rightx, downy)
            else:
                self.yoloDetector._set_mask(width, height, 0, 0, width, height)

        frames = []

        def _inner_batch_infer_and_write(frames_batch, out_writer):
            ball_detections, yolo_results = self.ImgArraysBatchInfer(frames_batch, topK=topK, apply_mask=apply_mask)
            ball_detection_per_frame.extend(ball_detections)
            bbox_points = []
            for frame_balls, result in zip(ball_detections, yolo_results):
                show = result.plot(kpt_radius=2, line_width=1)
                for obj in frame_balls:
                    if obj['balls'] is None:
                        continue
                    for ball in obj['balls']:
                        x, y, r = int(ball[0]), int(ball[1]), int(ball[2])
                        cv2.circle(show, (x, y), 1, (0, 255, 0), 1)
                        cv2.circle(show, (x, y), r, (0, 0, 255), 1)
                out_writer.write(show)
                bbox_points.append(extect_form_yolo_result(result))
            bbox_point_per_frame.extend(bbox_points)

        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                if len(frames) == self.yoloDetector.video_batch_size:
                    _inner_batch_infer_and_write(frames, out)
                    del frames
                    frames = []
                    pbar.update(self.yoloDetector.video_batch_size)
            if frames:
                _inner_batch_infer_and_write(frames, out)
                pbar.update(len(frames))
            del frames

        out.release()
        cap.release()

        return ball_detection_per_frame, bbox_point_per_frame

    def videoInfer(self, video, topK: Optional[int] = None, apply_mask: Optional[bool | List[int]] = False):
        """Infer on a video in one pass and return ball detections and YOLO bbox/keypoint info per frame."""
        cap = cv2.VideoCapture(video)
        ball_detection_per_frame = []
        bbox_point_per_frame = []
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        apply_mask_flag = False
        if isinstance(apply_mask, list) and len(apply_mask) == 4:
            leftx, topy, rightx, downy = apply_mask
            self.yoloDetector._set_mask(width, height, leftx, topy, rightx, downy)
            apply_mask_flag = True
        elif apply_mask is True and self.yoloDetector.roi is not None:
            leftx, topy, rightx, downy = self.yoloDetector.roi
            self.yoloDetector._set_mask(width, height, leftx, topy, rightx, downy)
            apply_mask_flag = True
        elif apply_mask is True:
            self.yoloDetector._set_mask(width, height, 0, 0, width, height)
            apply_mask_flag = True

        frames = []

        def _inner_batch_infer(frames_batch):
            ball_detections, yolo_results = self.ImgArraysBatchInfer(frames_batch, topK=topK, apply_mask=apply_mask_flag)
            ball_detection_per_frame.extend(ball_detections)
            bbox_points = [extect_form_yolo_result(result) for result in yolo_results]
            bbox_point_per_frame.extend(bbox_points)

        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                if len(frames) == self.yoloDetector.video_batch_size:
                    _inner_batch_infer(frames)
                    del frames
                    frames = []
                    pbar.update(self.yoloDetector.video_batch_size)
            if frames:
                _inner_batch_infer(frames)
                pbar.update(len(frames))
            del frames

        cap.release()

        return ball_detection_per_frame, bbox_point_per_frame
    
    def videoInfer_crop_ROI(self, video, crop_output_dir: str, topK: Optional[int] = None, apply_mask: Optional[bool | List[int]] = False):
        """Infer video and save cropped ROI images per detected object.

        Crop each YOLO detection in each frame and save to `crop_output_dir`.
        File name format: <frame_idx>_<class_name>_<obj_idx>.png

        Returns:
            tuple: (ball_detection_per_frame, bbox_point_per_frame, crop_file_paths)
        """
        os.makedirs(crop_output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video)
        ball_detection_per_frame = []
        bbox_point_per_frame = []
        crop_file_paths = []

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        apply_mask_flag = False
        if isinstance(apply_mask, list) and len(apply_mask) == 4:
            leftx, topy, rightx, downy = apply_mask
            self.yoloDetector._set_mask(width, height, leftx, topy, rightx, downy)
            apply_mask_flag = True
        elif apply_mask is True and self.yoloDetector.roi is not None:
            leftx, topy, rightx, downy = self.yoloDetector.roi
            self.yoloDetector._set_mask(width, height, leftx, topy, rightx, downy)
            apply_mask_flag = True
        elif apply_mask is True:
            self.yoloDetector._set_mask(width, height, 0, 0, width, height)
            apply_mask_flag = True

        frames = []
        start_frame_idx = 0

        def _inner_batch_infer_and_save(frames_batch, frame_base_idx):
            ball_detections, yolo_results = self.ImgArraysBatchInfer(frames_batch, topK=topK, apply_mask=apply_mask_flag)
            ball_detection_per_frame.extend(ball_detections)

            for batch_idx, result in enumerate(yolo_results):
                frame_idx = frame_base_idx + batch_idx
                bbox_point_per_frame.append(extect_form_yolo_result(result))

                if result.boxes is None:
                    continue

                for obj_idx in range(len(result.boxes)):
                    box = result.boxes[obj_idx]
                    xywh = box.xywh.cpu().numpy().reshape(4,)
                    cls_idx = int(box.cls.item()) if box.cls is not None else None
                    cls_name = result.names[cls_idx] if cls_idx is not None and cls_idx in result.names else f"cls{cls_idx}"

                    crop_img, _ = clip_image_with_minSize(xywh=xywh, im=result.orig_img, min_wh=self.min_wh)
                    fn = f"{frame_idx:06d}_{cls_name}_{obj_idx:03d}.png"
                    out_path = os.path.join(crop_output_dir, fn)
                    cv2.imwrite(out_path, crop_img)
                    crop_file_paths.append(out_path)

            return ball_detections, yolo_results

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        with tqdm(total=total_frames, desc="Processing video crop ROI") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                if len(frames) == self.yoloDetector.video_batch_size:
                    _inner_batch_infer_and_save(frames, start_frame_idx)
                    start_frame_idx += len(frames)
                    frames = []
                    pbar.update(self.yoloDetector.video_batch_size)
            if frames:
                _inner_batch_infer_and_save(frames, start_frame_idx)
                pbar.update(len(frames))

        cap.release()
        return ball_detection_per_frame, bbox_point_per_frame, crop_file_paths



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ball detection CLI via BallImproved_yoloDetector")
    parser.add_argument('-s','--source', type=Path, required=True, help="source path (video/image)")
    parser.add_argument('-o','--output', type=Path, default=None, required=False, help="output for video or image (for video: output video path when show mode)")
    parser.add_argument('-c','--crop_dir', type=Path, default=None, required=False, help="output directory to store cropped ROI images")
    parser.add_argument('-m','--model', type=Path, required=True, help="yolo model path")
    parser.add_argument('-k','--topK', type=int, default=None, help="topK for yolo detections")
    parser.add_argument('--yoloInfer', type=Path, default=None, required=False, help="yoloinfer config path (yaml)")
    parser.add_argument('--detection', type=Path, default=None, required=False, help="ball detection config path (yaml)")
    parser.add_argument('--apply_mask', action='store_true', help="apply ROI mask from model config (if set)")
    args = parser.parse_args()

    yolo_infer_para = load_yaml_as_dataclass(args.yoloInfer, YOLOInferPAra) if args.yoloInfer else DEFAULT_YOLO_INFER_PARA
    ball_detect_para = load_yaml_as_dataclass(args.detection, BallDetectionPara) if args.detection else DEFAULT_BALL_DETECTION_PARA

    model = YOLO(str(args.model))
    h_ranges = {
        'left': ((32, 45),),
        'right':((14, 27),(170,180)), 
    }
    detector = BallImproved_yoloDetector(model,H_ranges=h_ranges, yolo_infer_para=yolo_infer_para, ball_detect_para=ball_detect_para)

    if args.source.suffix.lower() in VIDEO_EXTENSIONS:
        if args.crop_dir is not None:
            ball_detection_per_frame, yolo_results_per_frame, crop_paths = detector.videoInfer_crop_ROI(
                str(args.source),
                str(args.crop_dir),
                topK=args.topK,
                apply_mask=args.apply_mask,
            )
            print(f"saved {len(crop_paths)} cropped ROI images under {args.crop_dir}")
        elif args.output is not None:
            ball_detection_per_frame, yolo_results_per_frame = detector.videoInferShow(
                str(args.source),
                str(args.output),
                topK=args.topK,
                apply_mask=args.apply_mask,
            )
        else:
            ball_detection_per_frame, yolo_results_per_frame = detector.videoInfer(
                str(args.source),
                topK=args.topK,
                apply_mask=args.apply_mask,
            )

        out_json = (args.output.with_suffix('.json') if args.output is not None else args.source.with_suffix('_ballInfer.json'))
        with open(out_json, 'w') as file:
            json.dump({
                'ball_detection_per_frame': to_jsonable(ball_detection_per_frame),
                'yolo_results_per_frame': to_jsonable(yolo_results_per_frame),
            }, fp=file)
        print(f"saved inference json to {out_json}")

    else:  # image input
        img = cv2.imread(str(args.source))
        if img is None:
            raise FileNotFoundError(f"Image not found: {args.source}")

        ball_detection, yolo_results = detector.ImgArraysBatchInfer([img], topK=args.topK, apply_mask=args.apply_mask)

        if args.crop_dir is not None:
            os.makedirs(args.crop_dir, exist_ok=True)
            if yolo_results and yolo_results[0].boxes is not None:
                for obj_idx in range(len(yolo_results[0].boxes)):
                    box = yolo_results[0].boxes[obj_idx]
                    xywh = box.xywh.cpu().numpy().reshape(4,)
                    cls_idx = int(box.cls.item()) if box.cls is not None else None
                    cls_name = yolo_results[0].names[cls_idx] if cls_idx is not None and cls_idx in yolo_results[0].names else f"cls{cls_idx}"
                    crop_img, _ = clip_image_with_minSize(xywh=xywh, im=img, min_wh=detector.min_wh)
                    out_path = os.path.join(str(args.crop_dir), f"img_{args.source.stem}_{cls_name}_{obj_idx:03d}.png")
                    cv2.imwrite(out_path, crop_img)
            print(f"saved cropped ROI images to {args.crop_dir}")

        if args.output is not None:
            # for an image we can save a visualized result
            show = yolo_results[0].plot(kpt_radius=2, line_width=1)
            show_bgr = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(args.output), show_bgr)
            print(f"saved visualization to {args.output}")

        print('ball_detection:', ball_detection)

"""
python src/markDetection.py -s infer/newData/ballbar-normal-yr/top-03102026224104_04m00s_04m15s_sample.avi -m runs/ballmany/weights/best.pt 
  --yoloInfer infer/newData/ballbar-normal-yr/top-yoloinfer.yaml --detection config/demo-ballMarkerdetection.yaml --crop_dir show/crops --output show/marked.mp4  -k 1  --apply_mask
"""