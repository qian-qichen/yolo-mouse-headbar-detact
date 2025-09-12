from ultralytics import YOLO
import torch
data_config = "v1_demo_marker.yaml"
project = 'runs'
name = 'marker'
img_size = 1024
if __name__ =="__main__":
   torch.cuda.empty_cache()
   model = YOLO("yolov11x-pose.yaml")
   model = model.load(r"./pretrained/yolo11x-pose.pt")
   # Train the model
   results1 = model.train(data=data_config,
                           epochs=2000,
                           batch=12,
                           workers=6,
                           exist_ok=True,
                           seed=42,
                           imgsz=img_size,
                           project=project,
                           name=name,
                           single_cls=True,
                           freeze=10,
                           optimizer='AdamW',
                           # momentum = 0.3,
                           # cos_lr=True,
                           lr0=0.005,
                           lrf=0.5,
                           cache='ram',
                           patience=2000,
                           warmup_epochs=20,
                        # weight_decay
                        box=15,
                        # cls=0
                        pose=3,
                        # kobj
                        # dropout
                           )
   # Unfreeze all layers and fine-tune with a smaller learning rate for 5 epochs
   # model = YOLO(f"{project}/{name}/weights/last.pt")

   # results2 = model.train(data=data_config,
   #                      pretrained=f"{project}/{name}/weights/last.pt",
   #                      epochs=1000,
   #                      batch=8,
   #                      workers=4,
   #                      exist_ok=True,
   #                      seed=42,
   #                      imgsz=img_size,
   #                      project=project,
   #                      name=name,
   #                      single_cls=True,
   #                      freeze=0,  # Unfreeze all layers
   #                      optimizer='SGD',
   #                      momentum = 0.3,
   #                      lr0=0.0001,  # Smaller learning rate 
   #                      lrf=0.001,  # Learning rate final value
   #                      cache='ram',
   #                      patience=500,
   #                      # pose=36
   #                      )


