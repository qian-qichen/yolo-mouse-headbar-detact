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
MAX_CPU_NUMBER = os.cpu_count()


## manul iter way, native
def triangulate_multi_view(point_2d, camera_matrices,verbose=2):
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
    X4d = cv2.triangulatePoints(P1, P2, pt1.reshape(-1,1), pt2.reshape(-1,1))
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
    args:
    - lines2d: [number of view, 3([a,b,c][x,y,1]^sT=0)]
    return:
    - v: [3,1]
    - p: [3,1]
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
    def _undistort_cam(args):
        i, (K, dist) = args
        pts_cam = np.ascontiguousarray(pointses2d[i, :, :]).reshape(-1, 1, 2)
        pointses2d[i,:,:] = cv2.undistortPoints(pts_cam,K,dist).reshape(-1,2)

    _ = list(map(_undistort_cam,enumerate(zip(camera_martrixes, dist))))
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
    def _undistort_cam(args):
        i, (K, dist) = args
        pts_cam = np.ascontiguousarray(pointses2d[:, i, :]).reshape(-1, 1, 2)
        pointses2d[:,i,:] = cv2.undistortPoints(pts_cam,K,dist).reshape(-1,2)

    _ = list(map(_undistort_cam,enumerate(zip(camera_martrixes, dist))))    
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

def pixcel_2Dto3D_multiCam(pointses2d:np.ndarray,fxs:np.ndarray,fys:np.ndarray,cxs:np.ndarray,cys:np.ndarray,Rs:np.ndarray,Ts:np.ndarray):
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

    X, residues, rank, s = lstsq(A,B) # type: ignore
    return X.reshape(-1,3)
  
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
            self.cou_cores = min(cpu_cores,MAX_CPU_NUMBER)
        else:
            self.cou_cores = cpu_cores
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

    def undistortedlifting(self,pointses2d:list[np.ndarray]|np.ndarray|Dict[str,np.ndarray]):
        """
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


    def lifting(self,pointses2d:np.ndarray):
        """
        args:
        - pointses2d:
            - np.ndarray:[points,cam views,coordinates(2)] 
        """
        return pixcel_2Dto3D_multiCam(pointses2d,self.fxs,self.fys,self.cxs,self.cys,self.Rs,self.Ts)

    def undistortedliftingLine(self,points:np.ndarray):
        """
        unlike points lifting method, line lifting process involves SVD from numpy, thus can NOT handle batch processing, unless switch the whole process to torch.
        args:
            points: np.ndarray:[cam views,2 points that define a line,coordinates(2)]
        """
        return undistorted_pixcel_point2line(points, self.camera_martrixes,self.dist,self.Rs,self.Ts)



if __name__ == "__main__":
    # 构造虚拟相机参数
    cams = {}
    for i in range(3):
        camera_matrix = np.array([[800, 0, 320],
                                  [0, 800, 240],
                                  [0,   0,   1]], dtype=np.float64)
        dist_coeffs = np.random.default_rng(42).uniform(-0.01, 0.01, size=5).astype(np.float32)
        rvec = np.array([0.0, 0.0, 0.0])
        tvec = np.array([i * 0.1, 0.0, 0.0])
        cams[f"cam{i}"] = cameraPara(camera_matrix, dist_coeffs, rvec, tvec)

    # 构造3D点
    points_3d = np.array([
        [0.2, 0.1, 2.0],
        [0.0, -0.1, 2.2],
        [-0.1, 0.2, 1.8]
    ])

    # 投影到各相机
    points_2d = []
    for cam in cams.values():
        R_mat, _ = cv2.Rodrigues(cam.rvec)
        tvec = cam.tvec.reshape(3, 1)
        pts, _ = cv2.projectPoints(points_3d, cam.rvec, cam.tvec, cam.camera_matrix, cam.dist_coeffs)
        points_2d.append(pts.squeeze(1))
    points_2d = np.stack(points_2d, axis=1)  # [points, cams, 2]

    # 实例化lifter
    lf = Lifter(cams)

    # 测试 lifting 方法
    print("=== 测试 lifting 方法 ===")
    points_3d_pred = lf.lifting(points_2d)
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
    ])

    # 投影到各相机
    lines_2d = []
    for cam in cams.values():
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