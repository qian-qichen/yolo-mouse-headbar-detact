from src.save_yolo_detect_result import yoloinfer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
pathes = [
    # 'data/5.avi',
    # 'data/6.avi',
    # 'data/10_2025-09-13_18_16_14_1_0.avi',
    # 'data/10_2025-09-13_18_22_51_1_0.avi',
    # 'data/10_2025-09-13_22_18_35_1_0.avi',
    # 'data/10_2025-09-13_22_20_58_1_0.avi',
    # 'data/10_2025-09-16_11_29_19_1_0.avi',
    # 'data/2025-11-5-dev.avi',
    # 'data/11.avi',
    # 'data/12.avi',
    # 'data/13.avi',
    # 'data/14_rep.avi',
    # 'data/15_rep.avi',
    # 'data/16.avi',
    # 'data/17.avi',
    # 'data/2025-11-14_14_26_30_1_0.avi',
    # 'data/2025-11-14_14_33_24_1_0.avi',
    # 'data/2025-11-17_14_49_58_1_0.avi',
    # 'data/2025-11-17_14_58_29_1_0.avi',
    ''
]

for path in pathes:
    yoloinfer(model_path='runs/glass/weights/best.pt', source_path=path, output_path='show', batch_size=24, iou=0.7, imgsz=(1088,1440), conf=0.4, best_only=True)
