from ultralytics import YOLO
import torch
data_config = "v1_demo.yaml"
project = 'runs'
name = 'record'
img_size = 1024
if __name__ =="__main__":
   torch.cuda.empty_cache()
   model = YOLO("yolov11x-pose.yaml").load(r"./pretrained/yolo11x-pose.pt")
   # Train the model
   results1 = model.train(data=data_config,
                           epochs=5000,
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
                           momentum = 0.3,
                           # cos_lr=True,
                           lr0=0.005,
                           lrf=0.01,
                           cache='ram',
                           patience=2000,
                        # weight_decay
                        # box
                        # cls=0
                        pose=18,
                        # kobj
                        # dropout
                           )
   # Unfreeze all layers and fine-tune with a smaller learning rate for 5 epochs
   # model = YOLO(f"{project}/{name}/weights/last.pt")

   # results2 = model.train(data=data_config,
   #                      pretrained=f"{project}/{name}/weights/last.pt",
   #                      epochs=100,
   #                      batch=2,
   #                      workers=2,
   #                      exist_ok=True,
   #                      seed=42,
   #                      imgsz=img_size,
   #                      project=project,
   #                      name=name,
   #                      single_cls=True,
   #                      freeze=0,  # Unfreeze all layers
   #                      optimizer='SGD',
   #                      momentum = 0.02,
   #                      lr0=0.0001,  # Smaller learning rate 
   #                      lrf=0.001,  # Learning rate final value
   #                      cache='ram',

   #                      # pose=36
   #                      )


