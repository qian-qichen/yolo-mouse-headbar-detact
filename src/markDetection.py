import cv2
import numpy as np
from typing import Optional, List, Callable, Protocol, Tuple, Dict
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
from src.util import VIDEO_EXTENSIONS, YOLO_KF_PROFIE
from src.util.cilHelp import load_cli_args

import line_profiler

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



def cornerDetect(img,detectPara_Harris:dict,detectPara_other_dict:dict,ts=0.01)->np.ndarray:
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


def extect_form_yolo_result(yolo_result:Results):
    return {
        'bbox':yolo_result.boxes.cpu().xyxy.tolist(), # type: ignore
        'point':yolo_result.keypoints.cpu().xy.tolist() # type: ignore
    }

class MarkerImproved_yoloDetector:
    """
    A marker detector that combines YOLO object detection with a fine marker detection algorithm.
    supports ROI mask in format [leftx,topy,rightx,downy] (on orginal image size)
    """
    def __init__(self,yolo_model:YOLO|dict,yolo_infer_para:YOLOInferPAra=DEFAULT_YOLO_INFER_PARA,markerDetector:MarkerDetectorProtocol=cornerDetect, detection_para:DetectionPara=DEFAULT_DETECTION_PARA,cores:int=8,video_batch_size:int=4):
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

    def ImgArraysBatchInfer(self,imgs:np.ndarray|torch.Tensor|List[np.ndarray],topK:int=None, apply_mask:bool=False):
        """
        Perform batch inference on a list of images or image arrays, returning fine marker detections and YOLO results.
        
        :param imgs: [B,H,W,C] numpy arrays (untested), [B,C,H,W] torch tensors (untested) or list of path to the img
        :type imgs: np.ndarray | torch.Tensor | List[np.ndarray]
        :param topK: Number of top confident (yolo infer confidence) detections to keep per image. If None, all detections are kept.
        :type topK: int | None
        :param apply_mask: whether to apply the ROI mask to the input images before inference. Default is False.
        :type apply_mask: bool
        """
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="when run directly, take a image and return detection result for parameter turning")
    parser.add_argument('-s','--source', type=Path,help="source path")
    parser.add_argument('-o','--output',type=Path,default=None,required=False, help="save detection here if given")
    parser.add_argument('-m', '--model', type=Path, required=False, help="yolo model path")
    parser.add_argument('-k','--topK', type=int, default=None, help="only use topK detected object for marker detection")
    args = parser.parse_args()

    if args.source.suffix.lower() in VIDEO_EXTENSIONS:
        assert args.output is not None, "please provide output video path"
        assert args.model is not None, "please provide yolo model path"
        model = YOLO(args.model)
        detector = MarkerImproved_yoloDetector(model)
        marker_detection_per_frame, yolo_results_per_frame = detector.videoInferShow(str(args.source), str(args.output),topK=args.topK)
        with open(args.output.with_suffix('.json'),'w') as file:
            json.dump({
                'marker_detection_per_frame':[i.tolist() for i in marker_detection_per_frame],
                'yolo_results_per_frame': yolo_results_per_frame,
            },fp=file)
    else:
        img = cv2.imread(args.source)
        if img is None:
            raise FileNotFoundError(f"Image not found: {args.source}")
        result = cornerDetect(img, **asdict(DEFAULT_DETECTION_PARA))
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
    