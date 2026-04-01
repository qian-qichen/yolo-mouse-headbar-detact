import argparse
import json
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
    """支持新旧点保存格式：直接 [p1,p2] 或 {'points_3d': [...]}."""
    if frame is None:
        return None
    if isinstance(frame, dict):
        payload = frame.get('points_3d') or frame.get('3d') or frame.get('point_3d')
        if payload is None:
            return None
        return np.asarray(payload, dtype=np.float64)
    if isinstance(frame, list):
        return np.asarray(frame, dtype=np.float64)
    raise ValueError(f"未知 frame 格式: {type(frame)}")


def project_points(points_3d: np.ndarray, cam_para: Dict[str, Any]) -> np.ndarray:
    """将 N x 3 3D 点投影到图像平面。"""
    K = np.array(cam_para['camera_matrix'], dtype=np.float64)
    dist = np.array(cam_para.get('distortion_coeffs', [0, 0, 0, 0, 0]), dtype=np.float64)
    rvec = np.array(cam_para['rotation_vector'], dtype=np.float64).reshape(3, 1)
    tvec = np.array(cam_para['translation_vector'], dtype=np.float64).reshape(3, 1)

    pts = points_3d.reshape(-1, 3)
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
    args = parser.parse_args()

    points3d = load_json(args.points3d)
    if args.frame_end < 0:
        frame_end = len(points3d) if isinstance(points3d, list) else -1
    else:
        frame_end = args.frame_end

    stats = visualize_reprojection(
        args.points3d,
        args.camparam,
        args.video,
        args.output,
        camera_name=args.camera,
        draw_mode=args.draw,
        frame_range=(args.frame_start, frame_end),
    )

    print('重投影可视化完成')
    print(json.dumps(stats, ensure_ascii=False, indent=2))
