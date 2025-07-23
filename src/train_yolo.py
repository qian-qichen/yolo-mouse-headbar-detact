from ultralytics import YOLO
import torch
if __name__ =="__main__":
   torch.cuda.empty_cache()
   model = YOLO(r".\yolov11m-pose.yaml").load(r"pretrained\yolo11m-pose.pt")  # build from YAML and transfer weights
   # Train the model
   results = model.train(data="demo.yaml",
                           epochs=50,
                           batch=2,
                           workers=1,
                           exist_ok=True,
                           seed=2025,
                           imgsz=848,
                           single_cls=True,
                           freeze=10,
                           lr0=0.01,
                           lrf=0.001,
                        # weight_decay
                        # box
                        # cls=0
                        pose=36
                        # kobj
                        # dropout
                           )
   # Unfreeze all layers and fine-tune with a smaller learning rate for 5 epochs
   results = model.train(data="demo.yaml",
                        epochs=5,
                        batch=2,
                        workers=1,
                        exist_ok=True,
                        seed=2025,
                        imgsz=848,
                        single_cls=True,
                        freeze=0,  # Unfreeze all layers
                        optimizer='SGD',
                        momentum = 0.937,
                  
                        lr0=0.0001,  # Smaller learning rate
                        lrf=0.01,  # Learning rate final value
                        pose=36)
   # torch.cuda.empty_cache()

