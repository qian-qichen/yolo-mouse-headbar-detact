# %%
%load_ext autoreload
%autoreload 2
# %%
import json
import csv
import math
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.lift2Dto3D import load_camera_para_from_json
from sklearn.mixture import GaussianMixture
import copy
# %%
def compute_angle_to_ground(pair: List[List[float]]|np.ndarray) -> float:

    p1, p2 = pair
    dz = float(p2[2]) - float(p1[2])
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    r = math.sqrt(dx**2 + dy**2 + dz**2)
    if r == 0:
        return 0.0
    return math.degrees(math.asin(dz / r))

    # dz = float(p2[1]) - float(p1[1])
    # dx = float(p2[2]) - float(p1[2])
    # dy = float(p2[0]) - float(p1[0])


    
def collect_angles(points_3d) -> np.ndarray:
    if not isinstance(points_3d, list):
        raise ValueError('lifting_points文件必须是列表')

    angles = []
    for frame in points_3d:
        if frame is None:
            angles.append(np.nan)
        else:
            angles.append(compute_angle_to_ground(frame))
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


def plot_angles_with_note(angles: np.ndarray, filtered_angles: np.ndarray, stats: dict, bins: int = 60, figsize: Tuple[int, int] = (10, 6), note_pose: str = 'upper left') -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#f7f7f7')
    ax.set_facecolor('#fcfcfc')

    q1, q99 = np.percentile(angles, [1, 99])
    q25, q75 = np.percentile(angles, [25, 75])
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
    ax.vlines(q25, ymin=0, ymax=ax.get_ylim()[1], color='#ff7f00', linestyle='--', linewidth=1.2, label=f'Q1 {q25:.3f}°', zorder=4)
    ax.vlines(q75, ymin=0, ymax=ax.get_ylim()[1], color='#ff7f00', linestyle='--', linewidth=1.2, label=f'Q3 {q75:.3f}°', zorder=4)
    
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
    ax.legend(loc=note_pose, framealpha=0.9)

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
        ax.vlines(angles.mean(), ymin=0, ymax=ax.get_ylim()[1], color=color, linestyle='--', linewidth=1.2, label=f'{label} mean {angles.mean():.3f}°')

    ax.set_title('Angle distributions under different reprojection errors')
    ax.set_xlabel('angle (deg)')
    ax.set_ylabel('density')
    ax.set_xlim(x_min, x_max)
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', framealpha=0.9)
    return fig, ax


def smooth_series(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    if window_size <= 1:
        return data.copy()
    data = np.asarray(data, dtype=float)
    valid = ~np.isnan(data)
    if not valid.any():
        return data.copy()
    kernel = np.ones(window_size, dtype=float)
    padded = np.where(valid, data, 0.0)
    numerator = np.convolve(padded, kernel, mode='same')
    denominator = np.convolve(valid.astype(float), kernel, mode='same')
    smoothed = np.full_like(data, np.nan, dtype=float)
    mask = denominator > 0
    smoothed[mask] = numerator[mask] / denominator[mask]
    return smoothed


def _standardize_points(points_arr, expected_len: int, dims: int) -> np.ndarray:
    pts = np.asarray(points_arr, dtype=float)
    if pts.ndim == 1:
        if pts.size == expected_len * dims:
            pts = pts.reshape(expected_len, dims)
        elif pts.size % dims == 0:
            pts = pts.reshape(-1, dims)
    elif pts.ndim == 2 and pts.shape[1] == dims:
        pass

    if pts.ndim != 2 or pts.shape[1] != dims:
        raise ValueError(f"Points must be a 2D array of shape (n,{dims}), got {pts.shape}")
    return pts


def ordered_points_3d(points_3d, points_order: List[str]) -> np.ndarray:
    if isinstance(points_3d, dict):
        missing = [k for k in points_order if k not in points_3d]
        if missing:
            raise KeyError(f"points_3d missing keys: {missing}; available keys: {list(points_3d.keys())}")
        pts = [points_3d[k] for k in points_order]
        return _standardize_points(pts, len(points_order), 3)

    if isinstance(points_3d, (list, tuple, np.ndarray)):
        return _standardize_points(points_3d, len(points_order), 3)

    raise TypeError(f"Unsupported points_3d type: {type(points_3d)}")


def ordered_points_2d(points_2d, points_order: List[str]) -> np.ndarray:
    if isinstance(points_2d, dict):
        missing = [k for k in points_order if k not in points_2d]
        if missing:
            raise KeyError(f"points_2d missing keys: {missing}; available keys: {list(points_2d.keys())}")
        pts = [points_2d[k] for k in points_order]
        return _standardize_points(pts, len(points_order), 2)

    if isinstance(points_2d, (list, tuple, np.ndarray)):
        return _standardize_points(points_2d, len(points_order), 2)

    raise TypeError(f"Unsupported points_2d type: {type(points_2d)}")

# %% data

base_dir = Path('infer') / 'ballbar-newCboard' / '3'
saved = base_dir / '3dpoints.json'
json_condig = base_dir / 'lifting-yolo.json'
points_order = ['left', 'right']
saved = json.loads(saved.read_text())
condig = json.loads(json_condig.read_text())
cam_para = {name:load_camera_para_from_json(v['para']) for name, v in condig.items()}


# {"points_3d": {"left": [187.2218067331509, 120.86840186853458, -191.79835520397614], "right": [215.05694636935942, 122.8455368787877, -201.7790799650012]}, "points_2d": {"side": {"left": [352.44847, 575.6190799999999], "right": [317.46895, 726.918]}, "top": {"left": [2233.027955, 795.620545], "right": [2270.101925, 622.81967]}}}

# %% 算重投影误差汇总points_3ds
points_3ds = []
frame_errors = []
frame_error_vectors = []
cam_names = sorted(cam_para.keys())
camera_point_errors = {cam: {pt: [] for pt in points_order} for cam in cam_names}
camera_point_error_vectors = {cam: {pt: [] for pt in points_order} for cam in cam_names}
camera_frame_errors = {cam: [] for cam in cam_names}
point_frame_errors = {pt: [] for pt in points_order}

for frame in saved:
    if frame is None:
        frame_errors.append(np.nan)
        points_3ds.append(None)
        continue
    frame_error = []
    frame_error_vector = []
    frame_point_error_accum = {pt: [] for pt in points_order}

    points_3d_frame = ordered_points_3d(frame['points_3d'], points_order)
    # points_3d_frame = np.asarray(points_3d_frame, dtype=float)
    # points_3d_frame[:, 2] = -points_3d_frame[:, 2]
    points_3ds.append(points_3d_frame)
    for name, points in frame['points_2d'].items():
        if name not in cam_para:
            continue

        points_2d_frame = ordered_points_2d(points, points_order)
        proj, _ = cv2.projectPoints(
            points_3d_frame,
            cam_para[name].rvec,
            cam_para[name].tvec,
            cam_para[name].camera_matrix,
            cam_para[name].dist_coeffs,
        )

        proj = np.asarray(proj, dtype=float).reshape(-1, 2)
        if proj.ndim != 2 or proj.shape[1] != 2:
            raise ValueError(f"Projection result for '{name}' has invalid shape {proj.shape}")

        points_2d_frame = np.asarray(points_2d_frame, dtype=float)
        if points_2d_frame.ndim == 1 and points_2d_frame.size % 2 == 0:
            points_2d_frame = points_2d_frame.reshape(-1, 2)
        if points_2d_frame.ndim != 2 or points_2d_frame.shape[1] != 2:
            raise ValueError(f"2D points for '{name}' has invalid shape {points_2d_frame.shape}")
        if proj.shape != points_2d_frame.shape:
            raise ValueError(
                f"Point count mismatch in camera '{name}': proj {proj.shape} vs 2d {points_2d_frame.shape}"
            )
    
        err_vec = proj - points_2d_frame
        err_norm = np.linalg.norm(err_vec, axis=1)

        frame_error.append(err_norm)
        frame_error_vector.append(err_vec)
        camera_frame_errors[name].append(float(np.mean(err_norm)))

        for idx, pt in enumerate(points_order):
            if idx >= len(err_norm):
                continue
            camera_point_errors[name][pt].append(float(err_norm[idx]))
            camera_point_error_vectors[name][pt].append(err_vec[idx].astype(float))
            frame_point_error_accum[pt].append(float(err_norm[idx]))

    if len(frame_error) > 0:
        frame_errors.append(float(np.mean(np.concatenate(frame_error))))
        frame_error_vectors.append(np.stack(frame_error_vector))

    for pt in points_order:
        if len(frame_point_error_accum[pt]) > 0:
            point_frame_errors[pt].append(float(np.mean(frame_point_error_accum[pt])))

# %% 可视化重投影误差分布
fig_vec, axes_vec = plt.subplots(
    nrows=len(cam_names),
    ncols=len(points_order),
    figsize=(6 * len(points_order), 4 * len(cam_names)),
    squeeze=False,
)

for r, cam_name in enumerate(cam_names):
    for c, point_name in enumerate(points_order):
        ax = axes_vec[r, c]
        vecs = np.asarray(camera_point_error_vectors[cam_name][point_name], dtype=float)
        if vecs.size > 0:
            ax.scatter(vecs[:, 0], vecs[:, 1], alpha=0.35, s=8)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_title(f'Error vectors | cam={cam_name}, point={point_name}')
        ax.set_xlabel('du (px)')
        ax.set_ylabel('dv (px)')
        ax.grid(True, linestyle=':', alpha=0.35)

fig_vec.tight_layout()
fig_vec.savefig(base_dir / 'reprojection_error_vectors_by_cam_point.png', dpi=200, bbox_inches='tight')

fig_sep, axes_sep = plt.subplots(
    nrows=len(cam_names),
    ncols=len(points_order),
    figsize=(6 * len(points_order), 4 * len(cam_names)),
    squeeze=False,
)

for r, cam_name in enumerate(cam_names):
    for c, point_name in enumerate(points_order):
        ax = axes_sep[r, c]
        errs = np.asarray(camera_point_errors[cam_name][point_name], dtype=float)
        if errs.size > 0:
            ax.hist(errs, bins=120, color='tab:blue', alpha=0.7, edgecolor='k')
            mean_err = float(np.mean(errs))
            p90_err = float(np.percentile(errs, 90))
            ax.axvline(mean_err, color='tab:red', linewidth=1.5, label=f'mean={mean_err:.3f}px')
            ax.axvline(p90_err, color='tab:green', linestyle='--', linewidth=1.2, label=f'p90={p90_err:.3f}px')
            ax.legend(loc='upper right', framealpha=0.9)
        ax.set_title(f'Reproj error | cam={cam_name}, point={point_name}')
        ax.set_xlabel('Error (pixels)')
        ax.set_ylabel('Count')
        ax.grid(True, linestyle=':', alpha=0.35)

fig_sep.tight_layout()
fig_sep.savefig(base_dir / 'reprojection_error_hist_by_cam_point.png', dpi=200, bbox_inches='tight')

for cam_name in cam_names:
    cam_err = np.asarray(camera_frame_errors[cam_name], dtype=float)
    if cam_err.size == 0:
        continue
    print(f"camera={cam_name}: mean={cam_err.mean():.4f}px, p90={np.percentile(cam_err, 90):.4f}px, n={len(cam_err)}")

for point_name in points_order:
    pt_err = np.asarray(point_frame_errors[point_name], dtype=float)
    if pt_err.size == 0:
        continue
    print(f"point={point_name}: mean={pt_err.mean():.4f}px, p90={np.percentile(pt_err, 90):.4f}px, n={len(pt_err)}")



frame_errors_arr = np.asarray(frame_errors, dtype=float)
frame_errors_valid = frame_errors_arr[~np.isnan(frame_errors_arr)]

fig_reproj, ax_reproj = plt.subplots(figsize=(10, 6))
ax_reproj.hist(frame_errors_valid, bins=200, color='tab:orange', alpha=0.7, edgecolor='k')
ax_reproj.set_title('Reprojection Errors')
ax_reproj.set_xlabel('Error (pixels)')
ax_reproj.set_ylabel('Frame Count')
ax_reproj.grid(True, linestyle=':', alpha=0.4)
# %%
## 过滤 
# ts = np.nanmean(frame_errors)
ts = 5
valid_frames = [i for i, e in enumerate(frame_errors_arr) if not np.isnan(e) and e <= ts]
print(f"Total frames: {len(frame_errors_arr)}, Valid frames (error <= {ts}): {len(valid_frames)}, Ratio: {len(valid_frames)/len(frame_errors_arr):.2f}")
points_3ds_flitered = [points_3ds[i] for i in valid_frames]

# %%
angles = collect_angles(points_3ds_flitered)
gaussan_filtered_angles, gaussian_stats = gaussian_filter_angles(angles,sigma=3)
fig_ang, ax_ang = plot_angles_with_note(angles, gaussan_filtered_angles, gaussian_stats)

print(
    '角度统计: 原始样本数',
    len(angles),
    '均值(°)',
    gaussian_stats['raw_mean'],
    '标准差(°)',
    gaussian_stats['raw_std'],
    '过滤后样本数',
    len(gaussan_filtered_angles),
    '均值(°)',
    gaussian_stats['filtered_mean'],
    '标准差(°)',
    gaussian_stats['filtered_std'],
)


fig_ang.savefig(base_dir / 'lifting_angles_histogram.png', dpi=200, bbox_inches='tight')


# %%
# 不同error阈值的角度分布可视化
percentiles = [90,50,1]
thr_values = np.nanpercentile(frame_errors_arr, percentiles)
angle_groups = build_angle_groups_by_error(points_3ds, frame_errors_arr, thr_values)
fig_multi, ax_multi = plot_angle_groups(angle_groups, bins=50)
fig_multi.savefig(base_dir / 'angle_by_error.png', dpi=200, bbox_inches='tight')

# %% angle 取绝对值后可视化
angles_abs = np.abs(angles)

filtered_angles_abs, gaussian_stats_abs = gaussian_filter_angles(angles_abs,sigma=3)
fig_ang, ax_ang = plot_angles_with_note(angles_abs, filtered_angles_abs, gaussian_stats_abs, note_pose='lower right')

print(
    '角度统计: 原始样本数',
    len(angles_abs),
    '均值(°)',
    gaussian_stats_abs['raw_mean'],
    '标准差(°)',
    gaussian_stats_abs['raw_std'],
    '过滤后样本数',
    len(filtered_angles_abs),
    '均值(°)',
    gaussian_stats_abs['filtered_mean'],
    '标准差(°)',
    gaussian_stats_abs['filtered_std'],
)

fig_ang.savefig(base_dir / 'lifting_angles_abs_histogram.png', dpi=200, bbox_inches='tight')



#  需要对外使用: analyze_reprojection_errors/ analyze_lifting_angles
 # %%
fig, ax = plt.subplots(figsize=(25, 5))
ax.plot(angles, label='Angle (deg)')
mean = np.mean(angles)
std = np.std(angles)
ax.axhline(mean, color='r', linestyle='-', linewidth=1.5,
              label=f'Mean {mean:.3f}°')
ax.axhline(mean + std, color='g', linestyle='--', linewidth=1.2,
              label=f'+1 Std {mean + std:.3f}°')
ax.axhline(mean - std, color='g', linestyle='--', linewidth=1.2,
              label=f'-1 Std {mean - std:.3f}°')
ax.set_xlabel('Frame')
ax.set_ylabel('Angle (deg)')
ax.set_title('Angle vs. Frame')
ax.grid(True, linestyle=':')
ax.legend()
fig.savefig(base_dir / 'lifting_angle_vs_time.png', dpi=200, bbox_inches='tight')
# %% save
# 保存角度序列为pickle文件
import pickle
to_save = {
    'angles': angles,
    'gaussan_filtered_angles': gaussan_filtered_angles,
    'gaussian_stats': gaussian_stats,
    'angle_groups': angle_groups,
}
with open(base_dir / 'lifting_angles.pkl', 'wb') as f:
    pickle.dump(to_save, f)


# %% ===== GMM 拟合与可视化
GMM_info = {
    'num_components': [],
    'score': [],
    'bic': [],
    'aic': [],
}
GMM_model = {}
color_map = plt.cm.get_cmap('tab10')
for i in range(1, 10):

    gmm = GaussianMixture(n_components=i, random_state=0)
    angles_reshape = angles.reshape(-1, 1)
    gmm.fit(angles_reshape)
    score = gmm.score(angles_reshape)
    bic = gmm.bic(angles_reshape)
    aic = gmm.aic(angles_reshape)
    GMM_info['num_components'].append(i)
    GMM_info['score'].append(score)
    GMM_info['bic'].append(bic)
    GMM_info['aic'].append(aic)
    GMM_model[i] = copy.copy(gmm)

    x_plot = np.linspace(angles.min(), angles.max(), 500).reshape(-1, 1)
    logprob = gmm.score_samples(x_plot)
    pdf = np.exp(logprob)

    fig_gmm, ax_gmm = plt.subplots(figsize=(10, 5))
    ax_gmm.hist(angles, bins=60, density=True, alpha=0.5, color='gray', label='Histogram')
    ax_gmm.plot(x_plot, pdf, color='red', lw=2, label='GMM fit')
    for i in range(gmm.n_components):
        mean = gmm.means_[i, 0]
        std = np.sqrt(gmm.covariances_[i, 0, 0])
        weight = gmm.weights_[i]
        colo= color_map(i % 10)
        ax_gmm.plot(x_plot, weight * 1/(std * np.sqrt(2 * np.pi)) * np.exp(-(x_plot - mean)**2 / (2 * std**2)),
                    '--', lw=1.5, color=colo)
        ax_gmm.axvline(mean, color=colo, linestyle='--', linewidth=1, alpha=0.7,label=f'{i}: {mean:.3f}° ({weight:.3f})')


    ax_gmm.set_xlabel('Angle (deg)')
    ax_gmm.set_ylabel('Density')
    ax_gmm.set_title('Angle Distribution with GMM Fit')
    ax_gmm.legend()
    fig_gmm.savefig(base_dir / f'lifting_angles_gmm_{gmm.n_components}components_score_{score}_bic_{bic}_aic_{aic}.png', dpi=200, bbox_inches='tight')

# 保存GMM_info为CSV
csv_path = base_dir / 'lifting_angles_gmm_info.csv'
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['num_components', 'score', 'bic', 'aic'])
    for i in range(len(GMM_info['num_components'])):
        writer.writerow([
            GMM_info['num_components'][i],
            GMM_info['score'][i],
            GMM_info['bic'][i],
            GMM_info['aic'][i],
        ])
with open(base_dir / 'lifting_angles_gmm_models.pkl', 'wb') as f:
    pickle.dump(GMM_model, f)
# %%

middle_points = []
for frame in points_3ds:
    if frame is None:
        middle_points.append(None)
        continue
    pts = ordered_points_3d(frame, points_order)
    middle = (pts[0] + pts[1]) / 2.0
    middle_points.append(middle)

xy_speed = []
for i in range(1, len(middle_points)):
    if middle_points[i] is None or middle_points[i-1] is None:
        xy_speed.append(np.nan)
        continue
    dist = np.linalg.norm(middle_points[i][:2] - middle_points[i-1][:2])
    xy_speed.append(dist)
xy_speed.append(np.nan)
angle_full = collect_angles(points_3ds)

# %%
frame_count = len(points_3ds)
frame_index = np.arange(frame_count)
xy_speed_arr = np.asarray(xy_speed, dtype=float)
xy_speed_smoothed = smooth_series(xy_speed_arr, window_size=30)



xy_threshold = 0.5
selected_mask = np.zeros(frame_count, dtype=bool)
selected_mask[valid_frames] = True
selected_ranges = []
start_idx = None
for idx, selected in enumerate(selected_mask):
    if selected and start_idx is None:
        start_idx = idx
    elif not selected and start_idx is not None:
        selected_ranges.append((start_idx, idx - 1))
        start_idx = None
if start_idx is not None:
    selected_ranges.append((start_idx, frame_count - 1))

fig, axes = plt.subplots(2,1,figsize=(60, 8))
ax1, ax2 = axes
ax1.plot(frame_index, angle_full, color='tab:blue', label='angle_full', linewidth=1.4)
ax2.plot(frame_index, xy_speed_smoothed, color='tab:orange', label='xy_speed_smoothed', linewidth=1.4)
ax2.axhline(xy_threshold, color='red', linestyle='--', linewidth=1.2, label=f'xy_speed threshold {xy_threshold}')
for start, end in selected_ranges:
    ax1.axvspan(start, end + 1, color='green', alpha=0.12)
    ax2.axvspan(start, end + 1, color='green', alpha=0.12)
ax1.set_xlabel('Frame index')
ax1.set_ylabel('Angle (deg)', color='tab:blue')
ax2.set_ylabel('XY speed (px/frame)', color='tab:orange')
ax1.set_title('Angle and XY speed over time with selected frame segments')
ax1.grid(True, linestyle=':', alpha=0.4)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
fig.tight_layout()
fig.savefig(base_dir / 'angle_xy_speed_over_time.png', dpi=200, bbox_inches='tight')

# %%
