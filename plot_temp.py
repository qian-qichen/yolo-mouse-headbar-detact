import json
import pandas as pd
from src.lift2Dto3D import load_camera_para_from_json,cameraPara
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from src.util.cilHelp import load_cli_args
def calculate_angle(row):
    v = np.array([row['v_0'], row['v_1'], row['v_2']])
    norm_v = np.linalg.norm(v,ord=2)
    if norm_v == 0:
        return 0
    sin_phi = abs(v[2]) / norm_v
    phi = np.arcsin(sin_phi)
    return np.degrees(phi)


def calculate_distance(row):
    v = np.array([row['v_0'], row['v_1'], row['v_2']])
    p = np.array([row['p_0'], row['p_1'], row['p_2']])/ np.array([row['p_3']])
    middle = np.array([row['middle_0'], row['middle_1'], row['middle_2']])
    pm = middle - p
    cross = np.cross(pm, v)
    dist = np.linalg.norm(cross) / np.linalg.norm(v)
    return dist


def middle_point_reprojected_distance(row,cam,name):
    middle_3d = np.array([row['middle_0'], row['middle_1'], row['middle_2']])
    projected = cv2.projectPoints(middle_3d.reshape(-1,3),
                                  cam.rvec,
                                  cam.tvec,
                                  cam.camera_matrix,
                                  cam.dist_coeffs)[0].flatten()
    real_middle_2d = np.array([row[f'{name}_middle_0'], row[f'{name}_middle_1']])
    reprojected_middle_distance = np.linalg.norm(projected - real_middle_2d, ord=2) 
    return reprojected_middle_distance

def main(json_data):
    json_data = Path(json_data)
    with open(json_data, 'r') as f:
        saved = json.load(f)
    data = saved['lifter_outs']
    cam_params_paths = saved['cam_params']
    cam_params = {}
    for k,v in cam_params_paths.items():
        cam_params[k] = load_camera_para_from_json(v)

    df_list = []
    for record in data:
        if record is None or (isinstance(record, dict) and len(record) == 1 and list(record.values())[0] is None):
            continue  # 跳过无效记录
        if isinstance(record, dict):
            new_record = {}
            for key, value in record.items():
                if isinstance(value, list):
                    for i, v in enumerate(value):
                        if isinstance(v, list) and len(v) == 1:
                            new_record[f"{key}_{i}"] = v[0]
                        else:
                            new_record[f"{key}_{i}"] = v
                else:
                    new_record[key] = value
            df_list.append(new_record)

    df = pd.DataFrame(df_list)
    df['angle_with_xy_plane'] = df.apply(calculate_angle, axis=1)
    df['distance_to_line'] = df.apply(calculate_distance, axis=1)

    for name,cam in cam_params.items():
        df[f'{name}_reprojected_middle_distance'] = df.apply(lambda row: middle_point_reprojected_distance(row, cam, name), axis=1)



    df['side_reprojected_middle_distance'].plot(kind='hist', bins=90, alpha=0.7,logy=True)
    plt.xlabel('side_reprojected_middle_distance')
    plt.ylabel('Frequency')
    plt.title('side_reprojected_middle_distance')
    side_mean_dis = df['side_reprojected_middle_distance'].mean()
    side_median_dis = df['side_reprojected_middle_distance'].median()
    plt.axvline(side_mean_dis, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {side_mean_dis:.2f}')
    plt.axvline(side_median_dis, color='green', linestyle='dashed', linewidth=2, label=f'Median: {side_median_dis:.2f}')
    plt.legend()
    # # plt.show()
    plt.savefig(str(json_data.with_name('side_reprojected_middle_distance_hist.svg')))
    plt.close()


    df['top_reprojected_middle_distance'].plot(kind='hist', bins=90, alpha=0.7,logy=True)
    plt.xlabel('top_reprojected_middle_distance')
    plt.ylabel('Frequency')
    plt.title('top_reprojected_middle_distance')
    top_mean_dis = df['top_reprojected_middle_distance'].mean()
    top_median_dis = df['top_reprojected_middle_distance'].median()
    plt.axvline(top_mean_dis, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {top_mean_dis:.2f}')
    plt.axvline(top_median_dis, color='green', linestyle='dashed', linewidth=2, label=f'Median: {top_median_dis:.2f}')
    plt.legend()
    # # plt.show()
    plt.savefig(str(json_data.with_name('top_reprojected_middle_distance_hist.svg')))
    plt.close()

    # side_threshold_value = side_median_dis
    # top_threshold_value = top_median_dis

    side_threshold_value = 1
    top_threshold_value = 1

    valid_rows = np.bitwise_and( df['top_reprojected_middle_distance'] <= top_threshold_value, df['side_reprojected_middle_distance'] <= side_threshold_value)
    df_filtered = df[valid_rows]
    df_filtered['angle_with_xy_plane'].plot(kind='hist', bins=90, alpha=0.7, logy=False)
    plt.xlabel('Angle with XY Plane (degrees)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Angle with XY Plane')
    mean_angle = df_filtered['angle_with_xy_plane'].mean()
    median_angle = df_filtered['angle_with_xy_plane'].median()
    plt.axvline(mean_angle, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_angle:.2f}°')
    plt.axvline(median_angle, color='blue', linestyle='dashed', linewidth=2, label=f'median_angle: {median_angle:.2f}°')
    plt.legend()
    # # plt.show()
    plt.savefig(str(json_data.with_name('angle_with_xy_plane_hist.svg')))
    plt.close()

defaults ={
    'lifted_json': None,
}
helps = {
    'lifted_json': 'Path to the JSON file containing lifted 3D data',
}
if __name__ == "__main__":
    args,_ = load_cli_args(defaults,helps)
    main(args.lifted_json)