
#import statement
from ultralytics import YOLO

# Enter the path of video in which you want to Test
video_path = ""

## Enter the path of  Model1 or Model2 or Model3 that you have downloaded from repo

model_path = ""

# Load a model
model = YOLO(model_path)  # load a custom model

# Tracking the Model
results = model.track(source=video_path,show = True, conf=0.5, iou=0.7)
