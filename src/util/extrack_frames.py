# 本文件中的代码参考了[deeplabcut](https://github.com/DeepLabCut/DeepLabCut)中的相关实现。
# 部分功能和结构借鉴[deeplabcut](https://github.com/DeepLabCut/DeepLabCut)项目，以适应本项目需求。
import os
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from skimage import io
from skimage.util import img_as_ubyte
import cv2
from tqdm import tqdm
from argparse import ArgumentParser

class capVideo:
    """
    function require:
    - 
    """
    def __init__(self, video_path, bbox=None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.dimensions_wh = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._set_bbox(bbox)

    def __len__(self):
        return self.frame_count
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    def close(self):
        self.cap.release()
    def set_to_frame(self, frame_index):
        """跳转到指定帧索引，并验证实际跳转位置"""
        # 确保帧索引在有效范围内
        frame_index = max(0, min(frame_index, self.frame_count - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        # 验证实际跳转位置（根据项目规范要求）
        actual_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if abs(actual_pos - frame_index) > 1:
            print(f"Warning: seek inaccurate! Wanted {frame_index}, got {actual_pos}")
    
    def _set_bbox(self, bbox):
        """
        bbox: [x1,y1,x2,y2] top-left and bottom-right
        """
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # 确保 bbox 在视频尺寸范围内
            x1 = max(0, min(x1, self.dimensions_wh[0]))
            y1 = max(0, min(y1, self.dimensions_wh[1]))
            x2 = max(0, min(x2, self.dimensions_wh[0]))
            y2 = max(0, min(y2, self.dimensions_wh[1]))
            # 确保 x1 < x2 且 y1 < y2
            if x1 >= x2:
                x1, x2 = 0, self.dimensions_wh[0]
            if y1 >= y2:
                y1, y2 = 0, self.dimensions_wh[1]
            self.bbox = [x1, y1, x2, y2]
        else:
            self.bbox = [0, 0, self.dimensions_wh[0], self.dimensions_wh[1]]
            
    def read_frame(self, crop=False):
        ret, frame = self.cap.read()
        if not ret:
            return None
        if crop:
            x1, y1, x2, y2 = self.bbox
            frame = frame[y1:y2, x1:x2]
        return frame




def KmeansbasedFrameselectioncv2(
    cap: capVideo,
    numframes2pick: int,
    start: float,
    stop: float,
    Index=None,
    step=1,
    resizewidth=30,
    batchsize=100,
    max_iter=50,
    color=False,
):
    """This code downsamples the video to a width of resizewidth.
    The video is extracted as a numpy array, which is then clustered with kmeans, whereby each frames is treated as a vector.
    Frames from different clusters are then selected for labeling. This procedure makes sure that the frames "look different",
    i.e. different postures etc. On large videos this code is slow.

    Consider not extracting the frames from the whole video but rather set start and stop to a period around interesting behavior.

    Note: this method can return fewer images than numframes2pick.

    Attention: the flow of commands was not optimized for readability, but rather speed. This is why it might appear tedious and repetitive.
    """
    nframes = len(cap)
    print(f"Video has {nframes} frames at {cap.fps} fps, dimensions (wh): {cap.dimensions_wh}")
    nx, ny = cap.dimensions_wh
    ratio = resizewidth * 1.0 / nx
    if ratio > 1:
        raise Exception("Choice of resizewidth actually upsamples!")

    print(
        "Kmeans-quantization based extracting of frames from",
        round(start * nframes * 1.0 / cap.fps, 2),
        " seconds to",
        round(stop * nframes * 1.0 / cap.fps, 2),
        " seconds.",
    )
    startindex = int(np.floor(nframes * start))
    stopindex = int(np.ceil(nframes * stop))

    if Index is None:
        Index = np.arange(startindex, stopindex, step)
    else:
        Index = np.array(Index)
        Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!

    nframes = len(Index)
    if batchsize > nframes:
        batchsize = nframes // 2

    ny_ = np.round(ny * ratio).astype(int)
    nx_ = np.round(nx * ratio).astype(int)
    DATA = np.empty((nframes, ny_, nx_ * 3 if color else nx_))
    if len(Index) >= numframes2pick:
        if (
            np.mean(np.diff(Index)) > 1
        ):  # then non-consecutive indices are present, thus cap.set is required (which slows everything down!)
            print("Extracting and downsampling...", nframes, " frames from the video.")
            if color:
                for counter, index in tqdm(enumerate(Index), total=len(Index)):
                    cap.set_to_frame(index)  # extract a particular frame
                    frame = cap.read_frame(crop=True)
                    if frame is not None:
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        DATA[counter, :, :] = np.hstack(
                            [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
                        )
            else:
                for counter, index in tqdm(enumerate(Index), total=len(Index)):
                    cap.set_to_frame(index)  # extract a particular frame
                    frame = cap.read_frame(crop=True)
                    if frame is not None:
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        DATA[counter, :, :] = np.mean(image, 2)
        else:
            print("Extracting and downsampling...", nframes, " frames from the video.")
            if color:
                for counter, index in tqdm(enumerate(Index), total=len(Index)):
                    frame = cap.read_frame(crop=True)
                    if frame is not None:
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        DATA[counter, :, :] = np.hstack(
                            [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
                        )
            else:
                for counter, index in tqdm(enumerate(Index), total=len(Index)):
                    frame = cap.read_frame(crop=True)
                    if frame is not None:
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        DATA[counter, :, :] = np.mean(image, 2)

        print("Kmeans clustering ... (this might take a while)")
        data = DATA - DATA.mean(axis=0)
        data = data.reshape(nframes, -1)  # stacking

        kmeans = MiniBatchKMeans(
            n_clusters=numframes2pick, tol=1e-3, batch_size=batchsize, max_iter=max_iter
        )
        kmeans.fit(data)
        frames2pick = []
        for clusterid in range(numframes2pick):  # pick one frame per cluster
            clusterids = np.where(clusterid == kmeans.labels_)[0]

            numimagesofcluster = len(clusterids)
            if numimagesofcluster > 0:
                frames2pick.append(
                    Index[clusterids[np.random.randint(numimagesofcluster)]]
                )

        return list(np.array(frames2pick))
    else:
        return list(Index)

def extract_frames(video_path, output_dir, index=None):
    if index is None:
        return None
    os.makedirs(output_dir, exist_ok=True)
    
    index = np.array(sorted(index))  # 确保有序，便于判断连续性
    video_pure_name = Path(video_path).stem
    with capVideo(video_path) as cap:
        total_frames = len(cap)
        # 过滤有效索引
        valid_index = index[(index >= 0) & (index < total_frames)]
        if len(valid_index) == 0:
            print("No valid frame indices provided.")
            return []

        # 判断是否使用 seek
        use_seek = (
            len(valid_index) < 100 or 
            (len(valid_index) > 1 and np.mean(np.diff(valid_index)) > 5)
        )

        saved_paths = []
        if use_seek:
            # 方式1: 随机访问（适合稀疏、少量帧）
            for idx in tqdm(valid_index, total=len(valid_index), desc="Extracting frames (seek)"):
                cap.set_to_frame(idx)
                frame = cap.read_frame(crop=True)
                if frame is not None:
                    out_path = os.path.join(output_dir, f"{video_pure_name}_frame_{idx:06d}.png")
                    cv2.imwrite(out_path, frame)
                    saved_paths.append(out_path)
        else:
            # 方式2: 顺序读取（适合密集或连续帧）
            target_set = set(valid_index.tolist())
            current_idx = 0
            pbar = tqdm(total=valid_index[-1] + 1 - valid_index[0], desc="Extracting frames (sequential)")
            cap.set_to_frame(valid_index[0])
            while current_idx <= valid_index[-1]:
                frame = cap.read_frame(crop=True)
                if frame is None:
                    break
                if current_idx in target_set:
                    out_path = os.path.join(output_dir, f"{video_pure_name}_frame_{current_idx:06d}.png")
                    cv2.imwrite(out_path, frame)
                    saved_paths.append(out_path)
                current_idx += 1
                pbar.update(1)
            pbar.close()

        return saved_paths
    
def main(video_path, output_dir, frame_wanted):
    with capVideo(video_path) as cap:
        selected_frames = KmeansbasedFrameselectioncv2(
            cap=cap,
            numframes2pick=frame_wanted,
            start=0.0,
            stop=1.0,
            Index=None,
            step=1,
            resizewidth=30,
            batchsize=100,
            max_iter=50,
            color=False,
        )
    _ = extract_frames(video_path, output_dir, index=selected_frames)

VIDEO_EXTENSIONS = (".mp4", ".avi")

if __name__ == "__main__":
    pauser = ArgumentParser(description="Extract frames from video")
    pauser.add_argument("-v","--video_path",nargs="+", help="Path to the video file") 
    pauser.add_argument("-i","--input_dir",nargs="+", help="directory to scan for video")
    pauser.add_argument("-o","--output_dir",default="extracted_frames", help="Output directory for extracted frames")
    pauser.add_argument("-n","--frame_wanted", type=int, help="Number of frames to extract")

    args = pauser.parse_args()
    videos = args.video_path or []
    dirs = args.input_dir
    out_dir = args.output_dir
    frame_wanted = args.frame_wanted
    if dirs:
        for dir in dirs:
            for video in os.listdir(dir):
                if video.endswith(VIDEO_EXTENSIONS):
                    video_path = os.path.join(dir, video)
                    videos.append(video_path)
                else:
                    print(f"{video} is not a video file.")
    video_count = len(videos)
    print(f"found {video_count} videos: {videos}")
    _ = input("enter to continue")
    for i, video_path in enumerate(videos):
        print(f"processing video {i+1}/{len(videos)}: {video_path}")
        main(video_path, out_dir, frame_wanted)
        

