from typing import List, Tuple
import os
VIDEO_EXTENSIONS: Tuple[str] = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
]
IMG_EXTENSIONS: Tuple[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
]
YOLO_KF_PROFIE = os.path.join(os.path.dirname(__file__),'.yaml')
