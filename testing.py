import cv2, numpy as np
# tests = [
#     ('mp4','mp4v'),
#     ('mp4','avc1'),
#     ('mp4','x264'),
#     ('mp4','H265 '),
#     ('mkv','X264'),
#     ('avi','XVID'),
#     ('avi','MJPG'),
# ]
# frame = np.zeros((64,64,3), dtype=np.uint8)
# for ext, code in tests:
#     fname = f'test_out.{ext}'
#     fourcc = cv2.VideoWriter_fourcc(*code)
#     w = cv2.VideoWriter(fname, fourcc, 10.0, (64,64))
#     ok = w.isOpened()
#     if ok:
#         w.write(frame)
#         w.release()
#     print(f"{fname}: fourcc={code} -> opened={ok}")

# from ultralytics.models.yolo import YOLO
# model = YOLO("runs/headbar/weights/last.pt")
# # model = YOLO("./config/yolov11x-pose-headbar.yaml")
# print(model.info(detailed=True))
# Open a video file (replace 'video.mp4' with your actual video path)
cap = cv2.VideoCapture('/home/qichen/headbar/data/headbarDual/side1127.mp4')

# Get the nominal frame count from video properties
nominal_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Nominal frame count: {nominal_frames}")

# Read frames one by one and count them
actual_frames = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    actual_frames += 1

cap.release()
print(f"Actual frame count: {actual_frames}")

# Compare the counts
if nominal_frames == actual_frames:
    print("Frame counts match.")
else:
    print(f"Frame counts differ: nominal={nominal_frames}, actual={actual_frames}")