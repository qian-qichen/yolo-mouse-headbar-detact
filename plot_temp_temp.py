# plot_temp_temp.py
# 1) 读取 lifting_points.json 数据，并计算 3D 线与地面夹角，绘制直方图
# 2) 读取 lifting_points.2d_points.json + lifting_points.cam_para.json 做反投影误差统计
# 3) 取均值作为阈值过滤高误差帧，输出有效帧占比，保留 figure 对象便于后续保存
# %%
import json
import math
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.lift2Dto3D import load_camera_para_from_json
# %%
def compute_angle_to_ground(pair: List[List[float]]|np.ndarray) -> float:

    p1, p2 = pair
    dz = float(p2[2]) - float(p1[2])
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    horiz = math.hypot(dx, dy)
    return math.atan2(dz, horiz)


def collect_angles(points_3d) -> np.ndarray:
    if not isinstance(points_3d, list):
        raise ValueError('lifting_points文件必须是列表')

    angles = []
    for frame in points_3d:
        if frame is None:
            continue
        try:
            angles.append(np.degrees(compute_angle_to_ground(frame)))
        except Exception as e:
            print('忽略非法帧(角度) :', e)

    if len(angles) == 0:
        raise RuntimeError('没有非空帧可分析角度')

    return np.asarray(angles, dtype=float)


def gaussian_filter_angles(angles: np.ndarray, sigma: float = 3.0) -> Tuple[np.ndarray, dict]:
    mean_deg = float(np.mean(angles))
    std_deg = float(np.std(angles))
    threshold = sigma * std_deg

    if std_deg == 0.0:
        mask = np.ones_like(angles, dtype=bool)
    else:
        mask = np.abs(angles - mean_deg) <= threshold

    filtered = angles[mask]
    stats = {
        'raw_mean': mean_deg,
        'raw_std': std_deg,
        'filtered_mean': float(np.mean(filtered)) if len(filtered) else float('nan'),
        'filtered_std': float(np.std(filtered)) if len(filtered) else float('nan'),
        'keep_ratio': float(len(filtered)) / float(len(angles)),
        'sigma': sigma,
    }
    return filtered, stats


def plot_angles_with_gaussian(angles: np.ndarray, filtered_angles: np.ndarray, stats: dict, bins: int = 60, figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#f7f7f7')
    ax.set_facecolor('#fcfcfc')

    q1, q99 = np.percentile(angles, [1, 99])
    x_pad = max(1.0, 0.1 * (q99 - q1))
    x_min = q1 - x_pad
    x_max = q99 + x_pad
    hist_range = (x_min, x_max)

    ax.hist(
        angles,
        bins=bins,
        range=hist_range,
        color='#6baed6',
        alpha=0.35,
        density=True,
        histtype='stepfilled',
        linewidth=0,
        label='raw angles',
        zorder=1,
    )
    ax.hist(
        filtered_angles,
        bins=bins,
        range=hist_range,
        density=True,
        histtype='step',
        linewidth=1.6,
        color='#2ca25f',
        label='filtered (Gaussian)',
        zorder=3,
    )

    mean_f = stats['filtered_mean']
    std_f = stats['filtered_std']
    if not math.isnan(mean_f) and std_f > 0:
        x = np.linspace(x_min, x_max, 300)
        pdf = (1.0 / (std_f * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mean_f) / std_f) ** 2)
        ax.plot(x, pdf, color='#111111', linewidth=2.2, label='Gaussian fit', zorder=4)
        ax.axvline(mean_f, color='#e31a1c', linestyle='-', linewidth=2, label=f'mean {mean_f:.3f}°', zorder=5)
        ax.axvline(mean_f + std_f, color='#33a02c', linestyle='--', linewidth=1.2, label=f'+1σ {mean_f + std_f:.3f}°', zorder=5)
        ax.axvline(mean_f - std_f, color='#33a02c', linestyle='--', linewidth=1.2, label=f'-1σ {mean_f - std_f:.3f}°', zorder=5)
        ax.axvspan(mean_f - std_f, mean_f + std_f, color='#33a02c', alpha=0.10, zorder=0)

    ax.set_title('lifting_points Gaussian filter')
    ax.set_xlabel('angle (deg)')
    ax.set_ylabel('density')
    ax.set_xlim(x_min, x_max)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', framealpha=0.9)

    ax.text(
        0.98,
        0.95,
        f'raw n={len(angles)}\nraw mean={stats["raw_mean"]:.4f}°\nraw std={stats["raw_std"]:.4f}°\nkeep ratio={stats["keep_ratio"]:.2%}\nfiltered mean={stats["filtered_mean"]:.4f}°\nfiltered std={stats["filtered_std"]:.4f}°',
        transform=ax.transAxes,
        ha='right',
        va='top',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='#cccccc', boxstyle='round,pad=0.35'),
    )

    return fig, ax


def build_angle_groups_by_error(points_3d: list, errors: list, thresholds: List[float]) -> dict:
    error_array = np.asarray(errors, dtype=float)
    groups = {}
    for thr in thresholds:
        selected_frames = [frame for frame, e in zip(points_3d, error_array) if frame is not None and not math.isnan(e) and e <= thr]
        if len(selected_frames) == 0:
            continue
        groups[f'err<= {thr:.3f}'] = collect_angles(selected_frames)
    return groups


def plot_angle_groups(angle_groups: dict, bins: int = 50, figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    if len(angle_groups) == 0:
        raise RuntimeError('没有可用的角度组用于绘制')

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#f7f7f7')
    ax.set_facecolor('#fcfcfc')

    # Keep legend/order stable by threshold value in label: err<= value
    sorted_groups = sorted(angle_groups.items(), key=lambda kv: float(kv[0].split('<=')[-1]))
    all_angles = np.concatenate([v for _, v in sorted_groups], axis=0)
    q1, q99 = np.percentile(all_angles, [1, 99])
    x_pad = max(1.0, 0.1 * (q99 - q1))
    x_min = q1 - x_pad
    x_max = q99 + x_pad

    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(sorted_groups)))

    for (label, angles), color in zip(sorted_groups, colors):
        ax.hist(
            angles,
            bins=bins,
            range=(x_min, x_max),
            density=True,
            histtype='step',
            linewidth=1.8,
            color=color,
            label=f'{label} (n={len(angles)})',
        )

    ax.set_title('Angle distributions under different reprojection errors')
    ax.set_xlabel('angle (deg)')
    ax.set_ylabel('density')
    ax.set_xlim(x_min, x_max)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', framealpha=0.9)
    return fig, ax


def ordered_points_3d(points_3d_dict: dict, points_order: List[str]) -> np.ndarray:
    return np.asarray([points_3d_dict[k] for k in points_order], dtype=float)


def ordered_points_2d(points_2d_dict: dict, points_order: List[str]) -> np.ndarray:
    return np.asarray([points_2d_dict[k] for k in points_order], dtype=float)

# %%
base_dir = Path('infer') / 'newData' / 'ballbar-sick-yr'
saved = base_dir / 'lifting_points.json'
json_condig = base_dir / 'lifting.json'
points_order = ['right','left']
# %%
saved = json.loads(saved.read_text())
condig = json.loads(json_condig.read_text())
cam_para = {name:load_camera_para_from_json(v['para']) for name, v in condig.items()}


# {"points_3d": {"left": [187.2218067331509, 120.86840186853458, -191.79835520397614], "right": [215.05694636935942, 122.8455368787877, -201.7790799650012]}, "points_2d": {"side": {"left": [352.44847, 575.6190799999999], "right": [317.46895, 726.918]}, "top": {"left": [2233.027955, 795.620545], "right": [2270.101925, 622.81967]}}}

# %%
# 计算重投影误差,汇总points_3ds
points_3ds = []
frame_errors = []
for frame in saved:
    if frame is None:
        frame_errors.append(float('nan'))
        points_3ds.append(None)
        continue
    frame_error = []

    points_3d_frame = ordered_points_3d(frame['points_3d'], points_order)
    points_3ds.append(points_3d_frame)
    for name, points in frame['points_2d'].items():
        points_2d_frame = ordered_points_2d(points, points_order)
        proj,_ = cv2.projectPoints(points_3d_frame, cam_para[name].rvec, cam_para[name].tvec, cam_para[name].camera_matrix, cam_para[name].dist_coeffs)[0].squeeze()
        frame_error.append(np.linalg.norm(proj - points_2d_frame, axis=1))
    frame_errors.append(np.mean(frame_error))


# %%
fig_reproj, ax_reproj = plt.subplots(figsize=(10, 6))
ax_reproj.hist(frame_errors, bins=20, color='tab:orange', alpha=0.7, edgecolor='k')
ax_reproj.set_title('Reprojection Errors')
ax_reproj.set_xlabel('Error (pixels)')
ax_reproj.set_ylabel('Frame Count')
ax_reproj.grid(True, linestyle=':', alpha=0.4)
# %%
## 过滤 (均值阈值和高斯模型)
# ts = np.nanmean(frame_errors)
# valid_frames = [i for i, e in enumerate(frame_errors) if not np.isnan(e) and e <= ts]
# print(f"Total frames: {len(frame_errors)}, Valid frames (error <= {ts}): {len(valid_frames)}, Ratio: {len(valid_frames)/len(frame_errors):.2f}")
# points_3ds_flitered = [points_3ds[i] for i in valid_frames]

# %%
angles = collect_angles(points_3ds)
filtered_angles, gaussian_stats = gaussian_filter_angles(angles,sigma=1)
fig_ang, ax_ang = plot_angles_with_gaussian(angles, filtered_angles, gaussian_stats)



print(
    '角度统计: 原始样本数',
    len(angles),
    '均值(°)',
    gaussian_stats['raw_mean'],
    '标准差(°)',
    gaussian_stats['raw_std'],
    '过滤后样本数',
    len(filtered_angles),
    '均值(°)',
    gaussian_stats['filtered_mean'],
    '标准差(°)',
    gaussian_stats['filtered_std'],
)

fig_ang.savefig(base_dir / 'lifting_angles_histogram.png', dpi=200, bbox_inches='tight')

# %%
# 不同error阈值的角度分布可视化
percentiles = [90,25 ]
thr_values = np.nanpercentile(np.asarray(frame_errors, dtype=float), percentiles)
angle_groups = build_angle_groups_by_error(points_3ds, frame_errors, thr_values)
fig_multi, ax_multi = plot_angle_groups(angle_groups, bins=50)
fig_multi.savefig(base_dir / 'angle_by_error.png', dpi=200, bbox_inches='tight')

# 举例：筛选后有效帧索引（>mean_err）也可返回
# 这里可进一步输出各帧结果
# 不关闭绘图，这样在交互环境中还能继续处理/保存
# plt.show()  # 可选
# 返回值适用于函数调用接口
# if 需要对外使用: analyze_reprojection_errors/ analyze_lifting_angles
 # %%

# %%
