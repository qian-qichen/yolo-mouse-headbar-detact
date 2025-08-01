import numpy as np
from scipy.optimize import least_squares
import open3d as o3d

def triangulate_multi_view(points_2d, camera_matrices):
    """
    多视角三角化：最小化重投影误差
    :param points_2d: (N, 2) 数组，N 个视角下的像素坐标
    :param camera_matrices: (N, 3, 4) 数组，每个视角的 P = K[R|t]
    :return: 3D 坐标 (x, y, z)
    """
    def residual(X, points_2d, P_list):
        errors = []
        for i, P in enumerate(P_list):
            proj = P @ np.append(X, 1)  # 齐次投影
            proj = proj[:2] / proj[2]   # 归一化到图像平面
            errors.append(proj - points_2d[i])
        return np.concatenate(errors)

    # 初始估计：用前两个视角三角化（OpenCV）
    P1, P2 = camera_matrices[0], camera_matrices[1]
    pt1, pt2 = points_2d[0], points_2d[1]
    X4d = cv2.triangulatePoints(P1, P2, pt1.reshape(-1,1), pt2.reshape(-1,1))
    X3d = (X4d[:3] / X4d[3]).flatten()

    # 用所有视角优化
    result = least_squares(residual, X3d, args=(points_2d, camera_matrices))
    return result.x

# 使用示例
points_2d = np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])  # 4 个视角的像素坐标
P_list = [P1, P2, P3, P4]  # 4 个 3x4 相机矩阵

X_3d = triangulate_multi_view(points_2d, P_list)
print("Reconstructed 3D point:", X_3d)