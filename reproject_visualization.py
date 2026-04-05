import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_camera_param(camparams: Any, camera_name: str) -> Dict[str, Any]:
    """支持新的 camparam 格式（cam_name 包含 top/side）或直接单个相机参数。"""
    if isinstance(camparams, dict):
        if camera_name in camparams:
            return camparams[camera_name]
        if all(k in camparams for k in ['camera_matrix', 'rotation_vector', 'translation_vector']):
            return camparams
    raise ValueError(f"无法从 camparam 中解析相机参数：camera_name={camera_name}, camparams keys={list(camparams.keys()) if isinstance(camparams, dict) else type(camparams)}")


def normalize_points3d_entry(frame: Any) -> Optional[np.ndarray]:
    """严格解析 3D 点：仅接受 points_3d/3d/point_3d 或直接点列表。"""

    def _coerce_points3d_array(data: Any) -> Optional[np.ndarray]:
        if data is None:
            return None
        arr = np.asarray(data, dtype=np.float64)
        if arr.size == 0:
            return None

        if arr.ndim == 1:
            if arr.size % 3 != 0:
                raise ValueError(f"3D 点长度不是 3 的整数倍: shape={arr.shape}")
            arr = arr.reshape(-1, 3)
        elif arr.ndim == 2 and arr.shape[1] == 3:
            pass
        else:
            raise ValueError(f"3D 点形状非法，期望 (N,3)，实际 {arr.shape}")
        return arr

    if frame is None:
        return None

    if isinstance(frame, dict):
        payload = None
        for key in ['points_3d', '3d', 'point_3d']:
            if key in frame:
                payload = frame.get(key)
                break

        # 对 dict 帧，不再回退到 frame 本体，避免把 2D 信息误当 3D。
        if payload is None:
            return None

        if isinstance(payload, dict):
            ordered_keys = sorted(payload.keys())
            pts = [payload[k] for k in ordered_keys if payload[k] is not None]
            if len(pts) == 0:
                return None
            return _coerce_points3d_array(pts)

        return _coerce_points3d_array(payload)

    if isinstance(frame, list):
        return _coerce_points3d_array(frame)

    raise ValueError(f"未知 frame 格式: {type(frame)}")


def normalize_points2d_entry(frame: Any, cam_name: str = 'top') -> Optional[np.ndarray]:
    """支持 {'points_2d': {'top': {'left':[x,y], 'right':[x2,y2]}, ...}} 或 list 格式"""
    if frame is None:
        return None
    if isinstance(frame, dict):
        payload = frame.get('points_2d')
        if payload is None:
            return None

        if isinstance(payload, dict):
            cam_pts = payload.get(cam_name, None)
            if cam_pts is None:
                return None
            if isinstance(cam_pts, dict):
                ordered_keys = sorted(cam_pts.keys())
                return np.asarray([cam_pts[k] for k in ordered_keys], dtype=np.float64)
            if isinstance(cam_pts, list):
                return np.asarray(cam_pts, dtype=np.float64)

        if isinstance(payload, list):
            return np.asarray(payload, dtype=np.float64)

    if isinstance(frame, list):
        return np.asarray(frame, dtype=np.float64)

    raise ValueError(f"未知 2D frame 格式: {type(frame)}")


def compute_angle_to_ground(pair: List[List[float]] | np.ndarray) -> float:
    """按 #sym:compute_angle_to_ground 计算 left-right 点对地面角度（度）。"""
    pts = np.asarray(pair, dtype=float)
    if pts.shape != (2, 3):
        raise ValueError(f"角度计算需要两个 3D 点，获得 {pts.shape}")
    p1, p2 = pts[0], pts[1]
    dz = float(p2[2]) - float(p1[2])
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    horiz = math.hypot(dx, dy)
    return math.degrees(math.atan2(dz, horiz))


def project_points(points_3d: np.ndarray, cam_para: Dict[str, Any]) -> np.ndarray:
    """将 N x 3 3D 点投影到图像平面。"""
    K = np.array(cam_para['camera_matrix'], dtype=np.float64)
    dist = np.array(cam_para.get('distortion_coeffs', [0, 0, 0, 0, 0]), dtype=np.float64)
    rvec = np.array(cam_para['rotation_vector'], dtype=np.float64).reshape(3, 1)
    tvec = np.array(cam_para['translation_vector'], dtype=np.float64).reshape(3, 1)

    pts = np.asarray(points_3d, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"project_points 输入必须是 (N,3)，实际 {pts.shape}")
    imgpts, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
    return imgpts.reshape(-1, 2)


def visualize_reprojection(
    points3d_path: str,
    camparam_path: str,
    video_path: str,
    output_path: str,
    camera_name: str = 'top',
    draw_mode: str = 'points',
    frame_range: Optional[Tuple[int, int]] = None,
    circle_color: Tuple[int, int, int] = (0, 0, 255),
    circle_radius: int = 4,
    line_color: Tuple[int, int, int] = (0, 255, 0),
    line_thickness: int = 2,
) -> Dict[str, Any]:
    points3d_data = load_json(points3d_path)
    camparams = load_json(camparam_path)
    cam_para = resolve_camera_param(camparams, camera_name)
    

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    Path(os.path.dirname(output_path) or '.').mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_start, frame_end = 0, len(points3d_data) if isinstance(points3d_data, list) else total_frames
    if frame_range is not None:
        frame_start = max(0, frame_range[0])
        frame_end = min(frame_end, frame_range[1])

    gx, gy = np.meshgrid(np.linspace(0, 120, num=13), np.linspace(0, 120, num=13))
    xy_ground = np.stack([gx.ravel(), gy.ravel()], axis=1)
    xy_ground = np.hstack([xy_ground, np.zeros((xy_ground.shape[0], 1), dtype=np.float64)])  # 添加 z=0

    xy_ground_proj = project_points(xy_ground, cam_para)
    overlay_frame = np.zeros((height, width, 3), dtype=np.uint8)
    for x, y in xy_ground_proj:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < width and 0 <= iy < height:
            cv2.circle(overlay_frame, (ix, iy), circle_radius, (255, 0, 0), -1)

    # 迭代视频帧并画点
    frame_idx = 0
    projected_frames = 0
    non_null_3d = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= frame_start and frame_idx < frame_end:
            if frame_idx < len(points3d_data):
                frame_item = points3d_data[frame_idx]
                pts3d = normalize_points3d_entry(frame_item)
            else:
                pts3d = None

            if pts3d is not None and pts3d.size > 0:
                non_null_3d += 1
                try:
                    pts2d = project_points(pts3d, cam_para)
                    for x, y in pts2d:
                        ix, iy = int(round(x)), int(round(y))
                        if 0 <= ix < width and 0 <= iy < height:
                            cv2.circle(frame, (ix, iy), circle_radius, circle_color, -1)

                    if draw_mode in ['lines', 'points+lines'] and pts2d.shape[0] >= 2:
                        pts_int = np.round(pts2d).astype(int)
                        for i in range(len(pts_int) - 1):
                            cv2.line(frame, tuple(pts_int[i]), tuple(pts_int[i + 1]), line_color, line_thickness)

                    projected_frames += 1
                except Exception as exc:
                    print(f"第{frame_idx}帧投影错误: {exc}")
        # 画地面参考
        frame = cv2.addWeighted(frame, 1.0, overlay_frame, 0.5, 0)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    stats = {
        'video_path': str(video_path),
        'output_path': str(output_path),
        'camera': camera_name,
        'frame_range': [frame_start, frame_end],
        'total_video_frames': total_frames,
        'processed_video_frames': frame_idx,
        'targeted_frames': max(0, frame_end - frame_start),
        'non_null_3d_frames': non_null_3d,
        'projected_frames': projected_frames,
        'occupancy': float(projected_frames) / max(1, frame_end - frame_start) * 100.0,
    }
    return stats



def visualize_reprojection_with_detected_2d(
    points_path: str,
    camparam_path: str,
    video_path: str,
    output_path: str,
    camera_name: str = 'top',
    draw_mode: str = 'points',
    frame_range: Optional[Tuple[int, int]] = None,
    detect_color: Tuple[int, int, int] = (255, 255, 0),
    proj_color: Tuple[int, int, int] = (0, 0, 255),
    line_color: Tuple[int, int, int] = (0, 255, 0),
    circle_radius: int = 4,
    line_thickness: int = 2,
) -> Dict[str, Any]:
    data = load_json(points_path)
    camparams = load_json(camparam_path)
    cam_para = resolve_camera_param(camparams, camera_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    Path(os.path.dirname(output_path) or '.').mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_start, frame_end = 0, len(data) if isinstance(data, list) else total_frames
    if frame_range is not None:
        frame_start = max(0, frame_range[0])
        frame_end = min(frame_end, frame_range[1])

    gx, gy = np.meshgrid(np.linspace(0, 120, num=13), np.linspace(0, 120, num=13))
    xy_ground = np.stack([gx.ravel(), gy.ravel()], axis=1)
    xy_ground = np.hstack([xy_ground, np.zeros((xy_ground.shape[0], 1), dtype=np.float64)])  # 添加 z=0

    xy_ground_proj = project_points(xy_ground, cam_para)
    overlay_frame = np.zeros((height, width, 3), dtype=np.uint8)
    for x, y in xy_ground_proj:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < width and 0 <= iy < height:
            cv2.circle(overlay_frame, (ix, iy), circle_radius, (255, 0, 0), -1)


    frame_idx = 0
    projected_frames = 0
    non_null_3d = 0
    angle_list = []


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= frame_start and frame_idx < frame_end:
            frame_item = data[frame_idx] if frame_idx < len(data) else None
            pts3d = normalize_points3d_entry(frame_item)
            pts2d_detected = normalize_points2d_entry(frame_item, camera_name)

            pts2d_proj = None
            angle_deg = None
            if pts3d is not None and pts3d.size > 0:
                try:
                    pts2d_proj = project_points(pts3d, cam_para)
                    non_null_3d += 1
                    if pts3d.shape[0] >= 2:
                        angle_deg = compute_angle_to_ground(pts3d[:2])
                        angle_list.append(angle_deg)
                except Exception as exc:
                    print(f"第{frame_idx}帧 3D->2D 投影异常: {exc}")

            if pts2d_detected is not None and pts2d_detected.size > 0:
                pts2d_detected = pts2d_detected.reshape(-1, 2)
                for x, y in pts2d_detected:
                    ix, iy = int(round(x)), int(round(y))
                    if 0 <= ix < width and 0 <= iy < height:
                        cv2.circle(frame, (ix, iy), circle_radius, detect_color, -1)

                if draw_mode in ['lines', 'points+lines'] and pts2d_detected.shape[0] >= 2:
                    pts_int = np.round(pts2d_detected).astype(int)
                    for i in range(len(pts_int) - 1):
                        cv2.line(frame, tuple(pts_int[i]), tuple(pts_int[i + 1]), line_color, line_thickness)

            if pts2d_proj is not None and pts2d_proj.size > 0:
                pts2d_proj = pts2d_proj.reshape(-1, 2)
                for x, y in pts2d_proj:
                    ix, iy = int(round(x)), int(round(y))
                    if 0 <= ix < width and 0 <= iy < height:
                        cv2.circle(frame, (ix, iy), circle_radius, proj_color, -1)

                if draw_mode in ['lines', 'points+lines'] and pts2d_proj.shape[0] >= 2:
                    pts_int = np.round(pts2d_proj).astype(int)
                    for i in range(len(pts_int) - 1):
                        cv2.line(frame, tuple(pts_int[i]), tuple(pts_int[i + 1]), line_color, line_thickness)

            # 若检测和投影点个数相同，画跟踪线
            if pts2d_detected is not None and pts2d_proj is not None and pts2d_detected.shape == pts2d_proj.shape:
                for (x1, y1), (x2, y2) in zip(pts2d_detected, pts2d_proj):
                    cv2.line(frame, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (0, 128, 255), 1)

            if angle_deg is not None:
                cv2.putText(frame, f"{angle_deg:.1f}", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            if pts2d_proj is not None:
                projected_frames += 1
        # 画地面参考
        frame = cv2.addWeighted(frame, 1.0, overlay_frame, 0.5, 0)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    stats = {
        'video_path': str(video_path),
        'output_path': str(output_path),
        'camera': camera_name,
        'frame_range': [frame_start, frame_end],
        'total_video_frames': total_frames,
        'processed_video_frames': frame_idx,
        'targeted_frames': max(0, frame_end - frame_start),
        'non_null_3d_frames': non_null_3d,
        'projected_frames': projected_frames,
        'occupancy': float(projected_frames) / max(1, frame_end - frame_start) * 100.0,
        'angle_mean': float(np.mean(angle_list)) if angle_list else None,
        'angle_std': float(np.std(angle_list)) if angle_list else None,
        'angle_count': len(angle_list),
    }
    return stats



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D 点重投影到视频可视化')
    parser.add_argument('--points3d', required=True, help='3D点列表JSON路径(含None或字典格式)')
    parser.add_argument('--camparam', required=True, help='相机参数JSON路径（可含top/side）')
    parser.add_argument('--video', required=True, help='输入视频路径')
    parser.add_argument('--output', required=True, help='输出视频路径')
    parser.add_argument('--camera', default='top', help='使用的相机名称')
    parser.add_argument('--draw', default='points', choices=['points', 'lines', 'points+lines'], help='绘制模式')
    parser.add_argument('--frame_start', type=int, default=0, help='开始帧索引（含）')
    parser.add_argument('--frame_end', type=int, default=-1, help='结束帧索引（不含），-1 表示到末尾')
    parser.add_argument('--mode', default='detected2d', choices=['simple', 'detected2d'], help='可视化模式')
    args = parser.parse_args()

    points3d = load_json(args.points3d)
    if args.frame_end < 0:
        frame_end = len(points3d) if isinstance(points3d, list) else -1
    else:
        frame_end = args.frame_end

    if args.mode == 'detected2d':
        stats = visualize_reprojection_with_detected_2d(
            points_path=args.points3d,
            camparam_path=args.camparam,
            video_path=args.video,
            output_path=args.output,
            camera_name=args.camera,
            draw_mode=args.draw,
            frame_range=(args.frame_start, frame_end),
        )
    else:
        stats = visualize_reprojection(
            points_path=args.points3d,
            camparam_path=args.camparam,
            video_path=args.video,
            output_path=args.output,
            camera_name=args.camera,
            draw_mode=args.draw,
            frame_range=(args.frame_start, frame_end),
        )

    print('重投影可视化完成')
    print(json.dumps(stats, ensure_ascii=False, indent=2))
