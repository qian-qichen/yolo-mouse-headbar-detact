from os import path
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import lstsq
import cv2
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor
import os
import json
import argparse
from itertools import combinations
from tqdm import tqdm
MAX_CPU_NUMBER = os.cpu_count()


## manul iter way, native
def triangulate_multi_view(point_2d, camera_matrices,init_x4d=None,verbose=0):
    """
    最小化重投影误差
    避免用于相机共线的退化情况
    args:
        param points_2d: (N, 2）
        param camera_matrices: [P_1,P_2,...P_N] 投影矩阵，P = K[R|t]
    return: (x, y, z)
    """
    def residual(X, points_2d, P_list):
        errors = []
        for i, P in enumerate(P_list):
            proj = P @ np.append(X, 1)  
            proj = proj[:2] / proj[2]   
            errors.append(proj - points_2d[i])
        return np.concatenate(errors)
    
    P1, P2 = camera_matrices[0], camera_matrices[1]
    pt1, pt2 = point_2d[0], point_2d[1]
    X4d = init_x4d if init_x4d is not None else cv2.triangulatePoints(P1, P2, pt1.reshape(-1,1), pt2.reshape(-1,1))
    X3d = (X4d[:3] / X4d[3]).flatten()
    result = least_squares(residual, X3d, args=(point_2d, camera_matrices),verbose=verbose)
    return result.x

## tensor equation way, should be faster, for 2d points observed from pixel coordinate
def pixcel_2Dto3D_point(point2d,fxs,fys,cxs,cys,Rs,Ts):
    """
    args:
        - points2d: (N, 2)
        - fxs, fys, cxs, cys: (N,)
        - Rs: (N, 3, 3)
        - Ts: (N, 3, 1)
    """
    N = point2d.shape[0]
    assistent_mertrxes = np.zeros((N, 2, 3))
    assistent_mertrxes[:, 0, 0] = fxs
    assistent_mertrxes[:, 0, 2] = cxs - point2d[:, 0]
    assistent_mertrxes[:, 1, 1] = fys
    assistent_mertrxes[:, 1, 2] = cys - point2d[:, 1]

    A = np.matmul(assistent_mertrxes, Rs)
    B = - np.matmul(assistent_mertrxes, Ts)

    A = A.reshape(-1, 3)
    B = B.reshape(-1, 1)
    return np.linalg.lstsq(A, B, rcond=None)[0].flatten()

## tensor equation way, should be faster, for 2d points observed from normalized coordinate
def normalized_2D_to_3D_point(point2d,Rs,Ts):
    """
    args:
        - points2d: [number of view, demention(2, x,y)]
    """
    N = point2d.shape[0]
    assistent_mertrxes = np.zeros((N, 2, 3))
    assistent_mertrxes[:, 0, 0] = 1
    assistent_mertrxes[:, 0, 2] = - point2d[:, 0]
    assistent_mertrxes[:, 1, 1] = 1
    assistent_mertrxes[:, 1, 2] = - point2d[:, 1]

    A = np.matmul(assistent_mertrxes, Rs)
    B = - np.matmul(assistent_mertrxes, Ts)

    A = A.reshape(-1, 3)
    B = B.reshape(-1, 1)
    return np.linalg.lstsq(A, B, rcond=None)[0].flatten()


def points2lines(points):
    """
    args:
    - points: np.bdarray in [numbers of line, 2(two points), 2(x,y)]
    return:
    - lines: np.ndarray in [numbers of line, 3([a b c], [a b c]@[x y z]^T = 0)]
    """
    lines = np.zeros((points.shape[0],3))
    p1 = points[:, 0, :]  # (N,2)
    p2 = points[:, 1, :]  # (N,2)
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]

    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1

    lines = np.stack([a, b, c], axis=1)  # (N,3)
    return lines
    

def line2plane(lines2d,Rs,Ts):
    """
    args:
        - lines2d: [number of view, 3([a,b,c][x,y,1]^sT=0)]
    """
    N = lines2d.shape[0]
    planes = np.zeros((N,4))
    lines2d = lines2d.reshape(N,1,1,3) # np.broadcast_to(lines2d,(N,1,1,3))
    Rs = Rs.reshape(N,1,3,3) # np.broadcast_to(Rs,(N,1,3,3))
    Ts = Ts.reshape(N,1,3,1) # np.broadcast_to(Ts,(N,1,3,1))
    planes[:,:3] = np.matmul(lines2d, Rs).reshape((N,3))
    planes[:,3] = np.matmul(lines2d,Ts).reshape((N))
    return planes


def lines_2D_to_3D(lines2d,Rs,Ts):
    """
    return a 3d line in format {x|x=v*t+p[:3]/p[3], t in R}
    args:
    - lines2d: [number of view, 3([a,b,c][x,y,1]^sT=0)]
    return:
    - v: [3,1] 
    - p: [4,1]
    """
    planes = line2plane(lines2d,Rs,Ts)
    # zeros = np.zeros((planes.shape[0],1))
    # v = np.linalg.lstsq(planes[:,3], zeros)
    _,_,vh = np.linalg.svd(planes[:,:3])
    v = vh[-1].reshape((-1,1))
    _,_,vh_points = np.linalg.svd(planes.T@planes)
    p = vh_points[-1].reshape((-1,1))
    
    return v, p
 
def undistorted_pixcel_point2line(pointses2d:np.ndarray,camera_martrixes:np.ndarray,dist:np.ndarray,Rs:np.ndarray,Ts:np.ndarray):
    """
    args:
    - pointses2d: np.ndarray:[cam views, 2 points that define a line,coordinates(2)]
    - camera_martrixes: in [number of cameras,3,3]
    - dist: in [number of cameras,N]
    - Rs:np.ndarray in [number of cameras,3,3]
    - Ts:np.ndarray in [number of cameras,3,1] ([number of cameras,3] may alse work)
    """
    cams_num, points_num = pointses2d.shape[:2]
    # def _undistort_cam(args):
    #     i, (K, dist) = args
    #     pts_cam = np.ascontiguousarray(pointses2d[i, :, :]).reshape(-1, 1, 2)
    #     pointses2d[i,:,:] = cv2.undistortPoints(pts_cam,K,dist).reshape(-1,2)

    # _ = list(map(_undistort_cam,enumerate(zip(camera_martrixes, dist))))
    pts_cam = np.ascontiguousarray(pointses2d).reshape(cams_num,-1, 1, 2)
    undistorted = [cv2.undistortPoints(pts_cam[i], camera_martrixes[i], dist[i]).reshape(-1, 2) for i in range(cams_num)]
    pointses2d = np.stack(undistorted, axis=0)
    lines = points2lines(pointses2d)
    return lines_2D_to_3D(lines, Rs, Ts)



def undistorted_pixcel_2Dto3D_multiPoints(pointses2d:np.ndarray,camera_martrixes:np.ndarray,dist:np.ndarray,Rs:np.ndarray,Ts:np.ndarray):
    """
    args:
    - pointses2d: np.ndarray:[points, cam views,coordinates(2)]
    - camera_martrixes: in [number of cameras,3,3]
    - dist: in [number of cameras,N]
    - Rs:np.ndarray in [number of cameras,3,3]
    - Ts:np.ndarray in [number of cameras,3,1] ([number of cameras,3] may alse work)
    
    """
    # undistort points for each camera
    points_num, cams_num = pointses2d.shape[:2]
    # for cam_idx in range(cams_num):
    #     pts_cam = np.ascontiguousarray(pointses2d[:, cam_idx, :]).reshape(-1, 1, 2)
    #     undistorted = cv2.undistortPoints(pts_cam, camera_martrixes[cam_idx], dist[cam_idx])
    #     pointses2d[:, cam_idx, :] = undistorted.reshape(-1, 2)

    pts_cam = np.ascontiguousarray(pointses2d).transpose(1, 0, 2).reshape(cams_num, -1, 1, 2)
    undistorted = [cv2.undistortPoints(pts_cam[i], camera_martrixes[i], dist[i]).reshape(-1, 2) for i in range(cams_num)]
    pointses2d = np.stack(undistorted, axis=1)

    Rs = np.broadcast_to(Rs, (points_num, cams_num, 3, 3))
    Ts = np.broadcast_to(Ts, (points_num, cams_num, 3, 1))

    assistent_mertrxes = np.zeros((points_num, cams_num, 2, 3))
    assistent_mertrxes[:, :, 0, 0] = 1
    assistent_mertrxes[:, :, 0, 2] = - pointses2d[:, :, 0]
    assistent_mertrxes[:, :, 1, 1] = 1
    assistent_mertrxes[:, :, 1, 2] = - pointses2d[:, :, 1]

    A = np.matmul(assistent_mertrxes, Rs)  # (points, cams, 2, 3)
    B = - np.matmul(assistent_mertrxes, Ts) # (points, cams, 2, 1)

    A = A.reshape(points_num, -1, 3)       # (points, 2*cams, 3)
    B = B.reshape(points_num, -1, 1)       # (points, 2*cams, 1)

    X, residues, rank, s = lstsq(A,B) # type: ignore
    return X.reshape(-1,3)

def pixcel_2Dto3D_multiCam(pointses2d:np.ndarray,fxs:np.ndarray,fys:np.ndarray,cxs:np.ndarray,cys:np.ndarray,Rs:np.ndarray,Ts:np.ndarray,P:Optional[np.ndarray]=None):
    """
    args:
    - pointses2d:
        - np.ndarray:[points,cam views,coordinates(2)] 
    """
    points_num, cams_num = pointses2d.shape[:2]

    fxs = np.broadcast_to(fxs, (points_num, cams_num))
    fys = np.broadcast_to(fys, (points_num, cams_num))
    cxs = np.broadcast_to(cxs, (points_num, cams_num))
    cys = np.broadcast_to(cys, (points_num, cams_num))
    Rs = np.broadcast_to(Rs, (points_num, cams_num, 3, 3))
    Ts = np.broadcast_to(Ts, (points_num, cams_num, 3, 1))

    assistent_mertrxes = np.zeros((points_num, cams_num, 2, 3))
    assistent_mertrxes[:, :, 0, 0] = fxs
    assistent_mertrxes[:, :, 0, 2] = cxs - pointses2d[:, :, 0]
    assistent_mertrxes[:, :, 1, 1] = fys
    assistent_mertrxes[:, :, 1, 2] = cys - pointses2d[:, :, 1]

    A = np.matmul(assistent_mertrxes, Rs)  # (points, cams, 2, 3)
    B = - np.matmul(assistent_mertrxes, Ts) # (points, cams, 2, 1)

    A = A.reshape(points_num, -1, 3)       # (points, 2*cams, 3)
    B = B.reshape(points_num, -1, 1)       # (points, 2*cams, 1)
    if A.shape!= (points_num, 2*cams_num, 3) or B.shape != (points_num, 2*cams_num, 1):
        import pdb;pdb.set_trace()
    # X = np.zeros((points_num,3))
    # for i in range(points_num):
    #     Ai = A[i].reshape(-1,3)   # (2*cams,3)
    #     # print(np.linalg.cond(Ai))
    #     Bi = B[i].reshape(-1,1)
    #     Xi, res, rank, s = np.linalg.lstsq(Ai, Bi, rcond=None)
    #     X[i] = Xi.ravel()
    X, residues, rank, s = lstsq(A,B) # type: ignore
    # X = X.squeeze()
    X = X.reshape(-1,3)
    if P is not None:
        for i in range(points_num):
            X[i] = triangulate_multi_view(pointses2d[i], P, init_x4d=np.append(X[i], 1),verbose=0)
            # X[i] = triangulate_multi_view(pointses2d[i], P,verbose=2)
    return X
  
@dataclass
class cameraPara:
    camera_matrix: np.ndarray  # shape: (3, 3)
    dist_coeffs: np.ndarray    # shape: (N,) or (1, N)
    rvec: np.ndarray           # shape: (3, 1) or (1, 3)
    tvec: np.ndarray           # shape: (3, 1) or (1, 3)


def load_camera_para_from_json(json_path: str) -> cameraPara:
    """
    {
        "cam0": {
            "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "distortion_coeffs": [k1, k2, p1, p2, k3],
            "rotation_vector": [rx, ry, rz],
            "translation_vector": [tx, ty, tz]
        },
        ...
    }
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["distortion_coeffs"], dtype=np.float32)
    rvec = np.array(data["rotation_vector"], dtype=np.float64)
    tvec = np.array(data["translation_vector"], dtype=np.float64)
    cam = cameraPara(camera_matrix, dist_coeffs, rvec, tvec)
    return cam

class Lifter:
    """
    make sure passing in points in the same camera order as cam_order
    """
    def __init__(self,cams:Dict[str,cameraPara],cpu_cores:int=1):
        self.cam_order = sorted(cams.keys())
        self.cam_order_name2index = {name: i for i, name in enumerate(self.cam_order)}
        num_cams = len(self.cam_order)
        self.camera_martrixes = np.zeros((num_cams, 3, 3))
        self.fxs = np.zeros(num_cams)
        self.fys = np.zeros(num_cams)
        self.cxs = np.zeros(num_cams)
        self.cys = np.zeros(num_cams)
        self.Rs = np.zeros((num_cams, 3, 3))
        self.Ts = np.zeros((num_cams, 3, 1))
        
        dist = []
        if MAX_CPU_NUMBER is not None:
            self.cpu_cores = min(cpu_cores,MAX_CPU_NUMBER)
        else:
            self.cpu_cores = cpu_cores
            print(f"failed to check if the assigned cpu_cores make sances, this meight be fine, or indicate hidden error.")

        for i, cam_name in enumerate(self.cam_order):
            cam = cams[cam_name]
            self.camera_martrixes[i] = cam.camera_matrix
            self.fxs[i] = cam.camera_matrix[0, 0]
            self.fys[i] = cam.camera_matrix[1, 1]
            self.cxs[i] = cam.camera_matrix[0, 2]
            self.cys[i] = cam.camera_matrix[1, 2]
            self.Rs[i] = cv2.Rodrigues(cam.rvec)[0]
            self.Ts[i] = cam.tvec.reshape(3, 1)
            dist.append(cams[cam_name].dist_coeffs)
        self.dist = np.asarray(dist)

        # 顺序生成triangulate_multi_view所需的camera_matrices
        triangulate_camera_matrices = []
        for i in range(num_cams):
            K = self.camera_martrixes[i]
            R_mat = self.Rs[i]
            T_vec = self.Ts[i].reshape(3, 1)
            RT = np.hstack((R_mat, T_vec))
            P = K @ RT
            triangulate_camera_matrices.append(P)
        self.triangulate_camera_matrices = np.array(triangulate_camera_matrices)

    def getCamsIndex(self,cams_name:List[str]):
        """Get indices of cameras by names in the Lifter camera order."""
        idxs = [self.cam_order_name2index[name] for name in cams_name]
        idx_arr = np.array(idxs, dtype=int)
        # idx_arr.sort()
        return idx_arr

    def undistortedlifting(self,pointses2d:list[np.ndarray]|np.ndarray|Dict[str,np.ndarray]):
        """Undistort input 2D points and triangulate to 3D points.

        args:
        - pointses2d:
            - np.ndarray:[points, cam views,coordinates(2)] 
            - list[np.ndarray]: list of array in shape [points,coordinates(2)]
            - Dict[str,np.ndarray]: {cam_name(in self.cam_order): array in shape [points,coordinates(2)]}
        """
        if isinstance(pointses2d,dict):
            pointses2d = [pointses2d[cam_name] for cam_name in self.cam_order]
        if isinstance(pointses2d,list):
            pointses2d = np.stack(pointses2d,axis=0)
        return undistorted_pixcel_2Dto3D_multiPoints(pointses2d,self.camera_martrixes,self.dist,self.Rs,self.Ts)


    def lifting(self,pointses2d:np.ndarray,valid_cams:Optional[List[str]]=None):
        """Triangulate pixel 2D points from multiple cameras into 3D coordinates.

        args:
        - pointses2d:
            - np.ndarray:[points,cam views,coordinates(2)] 
        """
        if valid_cams:
            idxs = self.getCamsIndex(valid_cams)
            return pixcel_2Dto3D_multiCam(pointses2d,self.fxs[idxs],self.fys[idxs],self.cxs[idxs],self.cys[idxs],self.Rs[idxs],self.Ts[idxs]) 
        else:
            return pixcel_2Dto3D_multiCam(pointses2d,self.fxs,self.fys,self.cxs,self.cys,self.Rs,self.Ts) 
    def undistortedliftingLine(self,points:np.ndarray,valid_cams:Optional[List[str]]=None):
        """Lift 2D line observations to a 3D line representation using camera geometry.

        unlike points lifting method, line lifting process involves SVD from numpy, thus can NOT handle batch processing, unless switch the whole process to torch.
        
        args:
        - points: np.ndarray:[cam views,2 points that define a line,coordinates(2)]
        - valid_cams: list[str], names of cams tobe used in this running process
        """
        if valid_cams:
            idxs = self.getCamsIndex(valid_cams)
            return undistorted_pixcel_point2line(points, self.camera_martrixes[idxs],self.dist[idxs],self.Rs[idxs],self.Ts[idxs])
        else:
            return undistorted_pixcel_point2line(points, self.camera_martrixes,self.dist,self.Rs,self.Ts)







def from_yolo_to_3d(data_dict, cpu_core=1,save_path=None, points_order = ['left', 'right']):
    # input data_dict should be:
    # {
    #     'cam name':{
    #         'para': path to json file contanting camera parameters,
    #         'data': path to json-line file contanting yolo infer result
    #     }
    # }
    """
    each output should be like
    {"points_3d": {"left": [, , ], "right": [, , ]}, "points_2d": {"side": {"left": [, ], "right": [, ]}, "top": {"left": [, ], "right": [, ]}}}
    """
    def yolo_bbox_to_center(box):
        return np.asarray([box['x1'] + box['x2'], box['y1'] + box['y2']]) / 2
    point_num = len(points_order)
    cam_para = {k:load_camera_para_from_json(v['para']) for k, v in data_dict.items()}
    yolo_data_path = {k: v['data'] for k, v in data_dict.items()}
    lift = Lifter(cam_para,cpu_cores=cpu_core)
    # 这里假设yolo_data中的每一行都是一个json数组

    to_save = []
    file_readers = {cam_name:open(yolo_data_path[cam_name],'r') for cam_name in lift.cam_order}
    
    failed = 0
    # for lines in tqdm(zip(*file_readers)):
    flag = True
    while flag:
        try:
            iter_item ={cam_name:next(v) for cam_name, v in file_readers.items()}
        except Exception as e:
            flag = False
            break
    
        detections = {cam_name: json.loads(v) for cam_name, v in iter_item.items()}

        if any(len(dets)!=point_num for cam_name, dets in detections.items()):
            to_save.append(None)
            failed += 1
            continue
        frame_points_2d = np.zeros((point_num, len(lift.cam_order), 2))
        point2index = {point: idx for idx, point in enumerate(points_order)}
        for cam_idx, cam_name in enumerate(lift.cam_order):
            dets = detections[cam_name]
            # points_in_cam = np.stack([yolo_bbox_to_center(dets[i]['box']) for i in range(point_num)])
            
            for det in dets:
                obj_name = det['name']
                index = point2index[obj_name]
                frame_points_2d[index, cam_idx, :] = yolo_bbox_to_center(det['box'])
        lifting_out = lift.lifting(frame_points_2d)
        lifting_out_save = {point_name: lifting_out[idx].tolist() for idx, point_name in enumerate(points_order)}

        # points_3d.append(lifting_out.tolist())
        # points_2d.append(frame_points_2d.tolist())
        save = {
            'points_3d': lifting_out_save,
            'points_2d':{
                cam_name: {name:frame_points_2d[index,cam_idx,:].tolist() for name, index in point2index.items()} for cam_idx, cam_name in enumerate(lift.cam_order)
            }
        }
        to_save.append(save)
    if save_path is not None: 
        with open(Path(save_path),'w') as f:
            json.dump(to_save,f)
        with open(Path(save_path).with_suffix('.failed_frames.txt'),'w') as f:
            f.write(f"failed frame ratio: {failed/len(to_save):.2f} ({failed}/{len(to_save)})")
    print(f"failed frame ratio: {failed/len(to_save):.2f} ({failed}/{len(to_save)})")

    for name, reader in file_readers.items():
        reader.close()
    return to_save


def _normalize_detection_point(raw_item):
    """从 ball/marker 点检测条目中提取 (x, y)。"""
    if raw_item is None:
        return None

    if isinstance(raw_item, dict):
        if 'balls' in raw_item:
            balls = np.asarray(raw_item['balls'])
            if balls.size == 0:
                return None
            if balls.ndim == 1:
                return np.asarray(balls[:2], dtype=np.float64)
            return np.asarray(balls[0, :2], dtype=np.float64)

        if 'x' in raw_item and 'y' in raw_item:
            return np.asarray([raw_item['x'], raw_item['y']], dtype=np.float64)

        if 'xy' in raw_item:
            xy = np.asarray(raw_item['xy'])
            if xy.ndim >= 1 and xy.size >= 2:
                return np.asarray(xy[:2], dtype=np.float64)

    if isinstance(raw_item, (list, tuple, np.ndarray)):
        arr = np.asarray(raw_item)
        if arr.ndim == 1 and arr.size >= 2:
            return np.asarray(arr[:2], dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return np.asarray(arr[0, :2], dtype=np.float64)

    return None


def _extract_detection_points(cam_frame_data, points_order):
    """从单相机单帧检测数据里按 points_order 取出每个点的 (x,y)。"""
    points = {p: None for p in points_order}
    if cam_frame_data is None:
        return points

    # 列表形式： [{'balls': ..., 'class': 'left'}, ...]
    if isinstance(cam_frame_data, (list, tuple)):
        if len(cam_frame_data) == 0:
            return points

        if len(points_order) == 1:
            target = points_order[0]
            # 首选与 class/name 匹配的检测项
            for item in cam_frame_data:
                if not isinstance(item, dict):
                    continue
                key = item.get('class') or item.get('name')
                if key == target:
                    xy = _normalize_detection_point(item)
                    if xy is not None:
                        points[target] = xy
                        return points
            # 回退到第一个检测项
            points[target] = _normalize_detection_point(cam_frame_data[0])
            return points

        for item in cam_frame_data:
            if not isinstance(item, dict):
                continue
            key = item.get('class') or item.get('name')
            if key in points and points[key] is None:
                xy = _normalize_detection_point(item)
                if xy is not None:
                    points[key] = xy
        return points

    # dict 形式：支持 {left: ..., right: ...} 也支持单点成分（class/name）
    if isinstance(cam_frame_data, dict):
        if all(p in cam_frame_data for p in points_order):
            for p in points_order:
                points[p] = _normalize_detection_point(cam_frame_data[p])
            return points

        if len(points_order) == 1:
            points[points_order[0]] = _normalize_detection_point(cam_frame_data)
            return points

        key = cam_frame_data.get('class') or cam_frame_data.get('name')
        if key in points:
            points[key] = _normalize_detection_point(cam_frame_data)
        return points

    # 其余类型，尝试单个点解析
    if len(points_order) == 1:
        points[points_order[0]] = _normalize_detection_point(cam_frame_data)

    return points


def from_ball_to_3d(data_dict, cpu_core=1, save_path=None, points_order=['ball']):
    """基于 ball_detection_per_frame（或 marker_detection_per_frame）计算 3D 坐标，输出格式尽量与 from_yolo_to_3d 一致。"""

    def _load_frame_list(file_path):
        with open(file_path, 'r') as f:
            raw = json.load(f)
        return raw['marker_detection_per_frame']

    cam_paras = {cam: load_camera_para_from_json(v['para']) for cam, v in data_dict.items()}
    frame_lists = {cam: _load_frame_list(v['data']) for cam, v in data_dict.items()}

    lifter = Lifter(cam_paras, cpu_cores=cpu_core)
    n_frames = min(len(lst) for lst in frame_lists.values())

    results = []
    failed = 0

    for frame_idx in range(n_frames):
        frame_points = {name: {} for name in points_order}
        points_2d_out = {}

        for cam_name in lifter.cam_order:
            cam_frame_data = frame_lists[cam_name][frame_idx]

            if cam_frame_data is None or (isinstance(cam_frame_data, (list, tuple)) and len(cam_frame_data) == 0):
                # 本相机本帧无检测点，点位输出留空
                if len(points_order) == 1:
                    points_2d_out[cam_name] = None
                else:
                    points_2d_out[cam_name] = {p: None for p in points_order}
                continue

            point_xy_map = _extract_detection_points(cam_frame_data, points_order)

            if len(points_order) == 1:
                xy = point_xy_map.get(points_order[0])
                points_2d_out[cam_name] = xy.tolist() if xy is not None else None
                if xy is not None:
                    frame_points[points_order[0]][cam_name] = xy
            else:
                points_2d_out[cam_name] = {}
                for point_name in points_order:
                    xy = point_xy_map.get(point_name)
                    points_2d_out[cam_name][point_name] = xy.tolist() if xy is not None else None
                    if xy is not None:
                        frame_points[point_name][cam_name] = xy

        frame_points_3d = {}
        frame_has_any_3d = False

        for point_name in points_order:
            cams_with_point = list(frame_points[point_name].keys())
            if len(cams_with_point) < 2:
                frame_points_3d[point_name] = None
                continue

            points_for_triangulation = np.stack([frame_points[point_name][cam] for cam in cams_with_point], axis=0)
            points_for_triangulation = points_for_triangulation[np.newaxis, :, :]

            world_coords = lifter.lifting(points_for_triangulation, valid_cams=cams_with_point)
            frame_points_3d[point_name] = world_coords[0].tolist()
            frame_has_any_3d = True

        if not frame_has_any_3d:
            results.append(None)
            failed += 1
            continue

        results.append({
            'points_3d': frame_points_3d,
            'points_2d': points_2d_out,
        })

    if save_path is not None:
        with open(Path(save_path), 'w') as f:
            json.dump(results, f)
        with open(Path(save_path).with_suffix('.failed_frames.txt'), 'w') as f:
            f.write(f"failed frame ratio: {failed/len(results):.2f} ({failed}/{len(results)})")

    print(f"failed frame ratio: {failed/len(results):.2f} ({failed}/{len(results)})")
    return results


def test():
    cams = {}
    real_camera_params_1 = {
        "camera_matrix": [[8937.705543490209, 0.0, 950.7819861356494],
                          [0.0, 14368.283336983313, 513.936115749223],
                          [0.0, 0.0, 1.0]],
        # "distortion_coeffs": [[-21.673199291644583], [797.7476535345186], [0.26307255199082663], [-0.5223006128869417], [3.5881003094542856]],
        "distortion_coeffs": [[0.2], [0.2], [0.2], [0.2], [0.2]],
        # "distortion_coeffs": [[0], [0], [0], [0], [0]],
        "rotation_vector": [[0.6247834354253683], [0.6924619868272612], [-1.5279827646296855]],
        "translation_vector": [[-186.4901166177019], [105.21741354228368], [3034.1935811283956]]
    }

    real_camera_params_2 = {
        "camera_matrix": [[8281.750631110628, 0.0, 935.2157147299188],
                          [0.0, 5974.804983645658, 543.0653653889324],
                          [0.0, 0.0, 1.0]],
        # "distortion_coeffs": [[-3.132897503995883], [-450.6019343027715], [-0.06647612272623674], [-0.3519367512937638], [24662.624227984514]],
        "distortion_coeffs": [[0.2], [0.2], [0.2], [0.2], [0.2]],
        # "distortion_coeffs": [[0], [0], [0], [0], [0]],
        "rotation_vector": [[-0.7119731683597983], [0.6266018947089165], [-1.5682713716336332]],
        "translation_vector": [[-197.87892876213232], [272.9103191690772], [3361.943204793558]]
    }

    camera_matrix_1 = np.array(real_camera_params_1["camera_matrix"], dtype=np.float64)
    dist_coeffs_1 = np.array(real_camera_params_1["distortion_coeffs"], dtype=np.float32).flatten()
    rvec_1 = np.array(real_camera_params_1["rotation_vector"], dtype=np.float64).flatten()
    tvec_1 = np.array(real_camera_params_1["translation_vector"], dtype=np.float64).flatten()

    camera_matrix_2 = np.array(real_camera_params_2["camera_matrix"], dtype=np.float64)
    dist_coeffs_2 = np.array(real_camera_params_2["distortion_coeffs"], dtype=np.float32).flatten()
    rvec_2 = np.array(real_camera_params_2["rotation_vector"], dtype=np.float64).flatten()
    tvec_2 = np.array(real_camera_params_2["translation_vector"], dtype=np.float64).flatten()

    cams["real_cam_1"] = cameraPara(camera_matrix_1, dist_coeffs_1, rvec_1, tvec_1)
    cams["real_cam_2"] = cameraPara(camera_matrix_2, dist_coeffs_2, rvec_2, tvec_2)

    # for i in range(3):
    #     camera_matrix = np.array([[800+10*i, 0, 320+10*i],
    #                               [0, 800+10*i, 240+10*i],
    #                               [0,   0,   1]], dtype=np.float64)
    #     dist_coeffs = np.random.uniform(-0.001, 0.001, size=5).astype(np.float32)
    #     rvec = R.random().as_rotvec() 
    #     tvec = np.random.uniform(-100, 100, size=3)
    #     print(tvec)

    #     cams[f"cam{i}"] = cameraPara(camera_matrix, dist_coeffs, rvec, tvec)

    
    lf = Lifter(cams)
    # 构造3D点
    points_3d = np.array([
        [0.2, 0.1, 2.0],
        [0.0, -0.1, 2.2],
        [-0.1, 0.2, 1.8],
        [0.5,0.5,1.5]
    ])*1000

    # 投影到各相机
    points_2d = []
    for cam_name in lf.cam_order:
        cam = cams[cam_name]
        pts, _ = cv2.projectPoints(points_3d, cam.rvec, cam.tvec, cam.camera_matrix, cam.dist_coeffs)
        points_2d.append(pts.squeeze(1))
    points_2d = np.stack(points_2d, axis=1)  # [points, cams, 2]


    # 测试 lifting 方法
    print("=== 测试 lifting 方法 ===")
    
    points_3d_pred = lf.lifting(points_2d.copy())
    print("真实3D点:\n", points_3d)
    print("lifting重建3D点:\n", points_3d_pred)
    dists = np.linalg.norm(points_3d - points_3d_pred, axis=1)
    print("每个点的重建误差:", dists)
    print("平均重建误差:", np.mean(dists))
    

    # 测试 undistortedlifting 方法
    print("\n=== 测试 undistortedlifting 方法 ===")
    points_3d_pred_undist = lf.undistortedlifting(points_2d.copy())
    print("undistortedlifting重建3D点:\n", points_3d_pred_undist)
    dists_undist = np.linalg.norm(points_3d - points_3d_pred_undist, axis=1)
    print("每个点的undistorted重建误差:", dists_undist)
    print("平均undistorted重建误差:", np.mean(dists_undist))

    # 测试 undistortedliftingLine 方法
    print("\n=== 测试 undistortedliftingLine 方法 ===")
    # 构造线段
    lines_3d = np.array([
        [[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]],
        [[-0.5, 0.5, 1.5], [0.5, -0.5, 2.5]],
        [[0.2, -0.2, 1.2], [-0.2, 0.2, 2.2]]
    ])*500
    print(f"lines:\n{lines_3d}")
    # 投影到各相机
    lines_2d = []
    for cam_name in lf.cam_order:
        cam = cams[cam_name]
        R_mat, _ = cv2.Rodrigues(cam.rvec)
        tvec = cam.tvec.reshape(3, 1)
        line_2d_cam = []
        for line in lines_3d:
            pts, _ = cv2.projectPoints(line, cam.rvec, cam.tvec, cam.camera_matrix, cam.dist_coeffs)
            line_2d_cam.append(pts.squeeze(1))
        lines_2d.append(np.array(line_2d_cam))
    lines_2d = np.transpose(np.array(lines_2d), (1, 0, 2, 3))  # [lines, cams, 2 points, 2]

    # 测试 undistortedliftingLine 方法
    for i, line_2d in enumerate(lines_2d):
        v, p = lf.undistortedliftingLine(line_2d)
        p = p[:3]/p[3]
        print(f"Line {i + 1}:")
        print("方向向量 v:\n", v.flatten())
        print("线上一点 p:\n", p.flatten())
        # 计算点到直线的距离
        distances0 = np.linalg.norm(np.cross(v.flatten(), lines_3d[i,0] - p.flatten())) / np.linalg.norm(v.flatten())
        distances1 = np.linalg.norm(np.cross(v.flatten(), lines_3d[i,1] - p.flatten())) / np.linalg.norm(v.flatten())
        distances = (distances0 + distances1)/2
        # distances = np.linalg.norm(np.cross(v.flatten(), points_3d[i] - p.flatten())) / np.linalg.norm(v.flatten())
        print("点到直线的距离:", distances)


## should be departured
def iter_togeter(intervals:Dict):
    intervals = {name: iter(value) for name, value in intervals.items()}
    names = list(intervals.keys())
    while True:
        try:
            yield {name: next(intervals[name]) for name in names}
        except StopIteration:
            break



def main1():
    parser = argparse.ArgumentParser(description="Lift 2D lines to 3D using undistortedliftingLine")
    parser.add_argument("-c","--config", type=str, required=True, help="Path to config JSON file")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
        points_num = config.get("points_num", 2)
        save_path  = config.get("save_path", "lifter_output.json")
    # Load camera parameters
    cams = {}
    cam_names = []
    for cam_name, params_file in config["camera_params"].items():
        with open(params_file,'r') as f:
            params = json.load(f)
        cams[cam_name] = cameraPara(
            np.array(params["camera_matrix"], dtype=np.float64),
            np.array(params["distortion_coeffs"], dtype=np.float32),
            np.array(params["rotation_vector"], dtype=np.float64),
            np.array(params["translation_vector"], dtype=np.float64)
        )
        cam_names.append(cam_name)
    lifter = Lifter(cams)
    # Load 2D detections (assuming shape: [num_lines, num_cams, 2, 2])
    detections = {}
    for cam_name, detect_file in config["detections_2d"].items():
        with open(detect_file, "r") as f:
            file_data = json.load(f)
            # detections[cam_name] = {
            #     "marker": np.asarray(file_data['marker_detection_per_frame']),
            #     # "yolo": file_data['yolo_results_per_frame']
            # }
            detections[cam_name] = [np.asarray(markers) if len(markers) > 0 else None for markers in file_data['marker_detection_per_frame']]
    # Process each line
    lifter_outs = []
    failed_count = 0
    # for i, outs in enumerate(detections):
    # import pdb;pdb.set_trace()
    for i,outs in enumerate(iter_togeter(detections)):
        print(f"\rframe {i}",end="")
        try:
            # points = np.stack([out[0][0][:2,:2] for name,out in outs.items()])
            valid_points = {name:out[:2,:2] for name,out in outs.items()}
            points = np.stack(list(valid_points.values()))
            assert points.shape[1] == points_num, f"detected points num {points.shape[1]} not equal to required {points_num}"
            valid_camName = list(valid_points.keys())
        except Exception as e:
            print(' failed at frame ',i,e)
            failed_count +=1
            lifter_outs.append(None)
            # import pdb;pdb.set_trace()
            continue
        lifting_out = lifter.undistortedliftingLine(points.copy(),valid_camName)
        middle_point = lifter.lifting(points.mean(axis=1)[np.newaxis,:,:],valid_camName)
        out = {"v":lifting_out[0].tolist(),'p':lifting_out[1].tolist(),'middle':middle_point.tolist()}
        pt2d = {name:points[i].tolist() for i,name in enumerate(valid_camName)}
        true_middle = {f"{name}_middle":points[i].mean(axis=0).tolist() for i,name in enumerate(valid_camName)}
        out = {**out, **pt2d, **true_middle}
        lifter_outs.append(out)
        tosave = {
            "lifter_outs": lifter_outs,
            "cam_params": config["camera_params"]

        }
    with open(save_path, "w") as f:
        json.dump(tosave, f)
    
    print(f"\nFinished processing. Failed frames: {failed_count}/{i+1}")
    # Save results
    
def main2():
    """CLI entrypoint for from_yolo_to_3d."""
    parser = argparse.ArgumentParser(description="Lift YOLO outputs to 3D points using camera parameters")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to JSON config containing camera entries {'cam': {'para':..., 'data':...}}")
    parser.add_argument("-o", "--output", default="from_yolo_to_3d_output.json",
                        help="Output JSON path for 3D points (default: from_yolo_to_3d_output.json)")
    parser.add_argument("-c", "--cpu", type=int, default=1,
                        help="Number of CPU cores to use for processing (default: 1)")
    parser.add_argument("-p", "--points", "--points-order", dest="points_order", default="left,right",
                        help="Comma-separated ordered key names in YOLO results (default: left,right)")
    args = parser.parse_args()

    if not path.exists(args.input):
        raise FileNotFoundError(f"Input config not found: {args.input}")

    with open(args.input, "r") as f:
        data_dict = json.load(f)

    points_order = [p.strip() for p in args.points_order.split(",") if p.strip()]
    if len(points_order) == 0:
        raise ValueError("points_order must contain at least one point name")

    points_3d = from_yolo_to_3d(data_dict,
                                 cpu_core=args.cpu,
                                 save_path=args.output,
                                 points_order=points_order)

    print(f"Saved 3D results to {args.output}")
    print(f"Processed {len(points_3d)} frames, failed ratio available in outputs.")


def main3():
    """CLI entrypoint for from_ball_to_3d."""
    parser = argparse.ArgumentParser(description="Lift ball/marker detection per frame outputs to 3D points")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to JSON config containing camera entries {'cam': {'para':..., 'data':...}}")
    parser.add_argument("-o", "--output", default="from_ball_to_3d_output.json",
                        help="Output JSON path for 3D points (default: from_ball_to_3d_output.json)")
    parser.add_argument("-c", "--cpu", type=int, default=1,
                        help="Number of CPU cores to use for processing (default: 1)")
    parser.add_argument("-p", "--points", "--points-order", dest="points_order", default="left, right",
                        help="Comma-separated ordered key names in detection results (default: ball)")
    args = parser.parse_args()

    if not path.exists(args.input):
        raise FileNotFoundError(f"Input config not found: {args.input}")

    with open(args.input, "r") as f:
        data_dict = json.load(f)

    points_order = [p.strip() for p in args.points_order.split(",") if p.strip()]
    if len(points_order) == 0:
        raise ValueError("points_order must contain at least one point name")

    points_3d = from_ball_to_3d(data_dict,
                                 cpu_core=args.cpu,
                                 save_path=args.output,
                                 points_order=points_order)

    print(f"Saved 3D results to {args.output}")
    print(f"Processed {len(points_3d)} frames, failed ratio available in outputs.")


if __name__ == "__main__":
    # main3()
    main2()
    # test()