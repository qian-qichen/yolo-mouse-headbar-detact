import os
import json
from lift2Dto3D import load_camera_para_from_json,cameraPara
import cv2
import numpy as np
from tqdm import tqdm
from util.cilHelp import load_cli_args
def plot_line(img,point1,point2,cameraPara:cameraPara,color,width):
    point1_2d = cv2.projectPoints(point1, cameraPara.rvec, cameraPara.tvec, cameraPara.camera_matrix, cameraPara.dist_coeffs)[0].flatten()
    point2_2d = cv2.projectPoints(point2, cameraPara.rvec, cameraPara.tvec, cameraPara.camera_matrix, cameraPara.dist_coeffs)[0].flatten()
    # import pdb;pdb.set_trace()
    return cv2.line(img, point1_2d.astype(np.int32), point2_2d.astype(np.int32), color, width)# type: ignore

def draw_infinite_line(img, pt1, pt2, color=(0, 255, 0), thickness=2):
    height, width = img.shape[:2]
    # Ax + By + C = 0
    x1, y1 = pt1
    x2, y2 = pt2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2  
    intersections = []
    if B != 0:
        y_l = (-C - A * 0) / B
        if 0 <= y_l < height:
            intersections.append((0, int(round(y_l))))
        y_r = (-C - A * (width - 1)) / B
        if 0 <= y_r < height:
            intersections.append((width - 1, int(round(y_r))))
    if A != 0:
        x_up = (-C - B * 0) / A
        if 0 <= x_up < width:
            intersections.append((int(round(x_up)), 0))
        x_down = (-C - B * (height - 1)) / A
        if 0 <= x_down < width:
            intersections.append((int(round(x_down)), height - 1))
    # line onside screen, plot this as hint
    if len(intersections) < 2:
        intersections = [(0, 0), (width - 1, height - 1)]

    pt1_draw = intersections[0]
    pt2_draw = intersections[1]
    return cv2.line(img, pt1_draw, pt2_draw, color, thickness)

def plot_3d_line(img,point1,point2,cameraPara:cameraPara,color,width):
    point1_2d = cv2.projectPoints(point1, cameraPara.rvec, cameraPara.tvec, cameraPara.camera_matrix, cameraPara.dist_coeffs)[0].flatten()
    point2_2d = cv2.projectPoints(point2, cameraPara.rvec, cameraPara.tvec, cameraPara.camera_matrix, cameraPara.dist_coeffs)[0].flatten()
    return draw_infinite_line(img,point1_2d,point2_2d,color,width)

def calculate_angle(v):
    norm_v = np.linalg.norm(v,ord=2)
    if norm_v == 0:
        return 0
    sin_phi = abs(v[2]) / norm_v
    phi = np.arcsin(sin_phi)
    return np.degrees(phi)


def show_lifting_out(source_dir,video_name:str,NameliftingOut_name,show_path,color,width):
    liftingOut_path = os.path.join(source_dir,NameliftingOut_name)
    with open(liftingOut_path, 'r') as file:
        save = json.load(file)
    base_name = video_name.split('.')[0]
    if "side" in base_name:
        dict_name = "side"
    elif "top" in base_name:
        dict_name = "top"
    else:
        raise ValueError("video_name should contain 'side' or 'top' to indicate the view angle")
    camPara_path = save['cam_params'][dict_name]
    camPara = load_camera_para_from_json(camPara_path)

    lifting_out = save['lifter_outs']
    
    video_path = os.path.join(source_dir,video_name)
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # type: ignore
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter(show_path, fourcc, fps, (frame_width, frame_height))
    for out in tqdm(lifting_out):
        flag,frame = video_capture.read()
        if not flag:
            break
        if out == None:
            out_video.write(frame)
        else:
            v = np.asarray(out['v'])
            p = np.asarray(out['p'])
            p = p[:3]/p[3]
            middle = np.asarray(out['middle'])
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-6:
                v = v/v_norm
            else:
                v= v*1e6

            middle_2d = cv2.projectPoints(middle, camPara.rvec, camPara.tvec, camPara.camera_matrix, camPara.dist_coeffs)[0].flatten()
            middle_2d = middle_2d.astype(np.int32)
            cv2.circle(frame, middle_2d, 10, (0, 255, 0), 3)  # type: ignore
            frame = plot_3d_line(frame,p-100*v, p+100*v, camPara,color,width)
            points = np.asarray(out[dict_name])
            for i in range(points.shape[0]):
                cv2.circle(frame, points[i].astype(np.int32), 10, (0, 0, 0), 3) 
            
            frame = plot_line(frame, np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0]), camPara, (0, 0, 255), width)
            frame = plot_line(frame, np.array([0.0, 0.0, 0.0]), np.array([0.0, 10.0, 0.0]), camPara, (0, 255, 0), width)
            frame = plot_line(frame, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 10.0]), camPara, (255, 0, 0), width)
            frame = plot_line(frame, np.array([0.0, 0.0, 0.0]), v*25, camPara, (255, 255, 255), width)

            angle = calculate_angle(v)[0].__float__()
            
            cv2.putText(frame, f"angle: {angle:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out_video.write(frame)
    video_capture.release()
    out_video.release()
    
if __name__ == "__main__":
    source_dir = "headbar/0113/0_check"
    video_name = "checkside-01132026164518.mp4"
    color = (255,0,0)
    width = 3
    ARGS = {
        "source_dir": source_dir,
        "video_name": video_name,
        "NameliftingOut_name": "lifting_out.json",
        "show_path": source_dir+"/show_side_lifting.mp4",
        "color": color,
        "width": width
    }
    HELPS = {}
    args,args_dict = load_cli_args(ARGS, HELPS)

    show_lifting_out(**args_dict)

    
    