from ultralytics import YOLO
# from torch import to

model = YOLO('yolov8x-pose.pt')  # load a pretrained model (recommended for training)
# model.to('cuda')

model.train(data='config.yaml', epochs=300, imgsz=640,lr0 = 0.00001,device=[0] ,optimizer ='Adam', batch = 4, patience=25, resume = False)