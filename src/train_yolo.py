from ultralytics.models.yolo import YOLO
import torch
data_config = "glassBoard.yaml"
project = 'runs'
name = 'glass'
img_size = 1008
if __name__ =="__main__":
   torch.cuda.empty_cache()
   model = YOLO("yolov11x-pose.yaml")
   model = model.load(r"./pretrained/yolo11x-pose.pt")
   # Train the model
   results1 = model.train(data=data_config,
                           epochs=2000,
                           batch=20,
                           workers=12,
                           exist_ok=True,
                           seed=42,
                           imgsz=img_size,
                           project=project,
                           name=name,
                           single_cls=True,
                           freeze=10,
                           optimizer='AdamW',
                           # momentum = 0.3,
                           cos_lr=True,
                           lr0=0.001,
                           lrf=0.01,
                           cache='ram', # ram stotage
                           patience=500,
                           warmup_epochs=5,
                           # weight_decay
                           # loss weight
                           box=15,
                           # cls=0
                           pose=7,
                           # kobj
                           # dropout
                           ## data argument
                           flipud=0.5,
                           fliplr=0.5,
                           shear=20,
                           degrees=90,
                           ## train speed up
                           # compile=True
                           )
   # Unfreeze all layers and fine-tune with a smaller learning rate
   # # model = YOLO(f"{project}/{name}/weights/last.pt")
   # model.load(f"{project}/{name}/weights/best.pt")
   # model = model.load(r"./pretrained/yolo11x-pose.pt")
   # results2 = model.train(data=data_config,
   #                      # pretrained=f"{project}/{name}/weights/last.pt",
   #                      epochs=2000,
   #                      batch=10,
   #                      workers=8,
   #                      exist_ok=True,
   #                      seed=42,
   #                      imgsz=img_size,
   #                      project=project,
   #                      name=name,
   #                      single_cls=True,
   #                      freeze=0,  
   #                      optimizer='AdamW',
   #                      # optimizer='SGD',
   #                      # momentum = 0.3,
   #                      lr0=0.001,
   #                      lrf=0.001, 
   #                      cache='ram',
   #                      patience=1000,
   #                      ## loss weight
   #                      box=15,
   #                      # cls=0
   #                      pose=0,
   #   #                  ## data argument
   #                      flipud=0.5,
   #                      fliplr=0.5,
   #                      shear=20,
   #                      degrees=90,
   #                      ## train speed up
   #                      compile=True
   #                      )


