## Threshold Values
Conf_threshold = 0.4
NMS_threshold = 0.4

## Model Configration and Weights file location
model_config_file = "./models/yolov4-tiny.cfg"
model_weights = "./models/yolov4-tiny.weights"

## Colours of bounding boxes
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]


## Get all the class names
class_names = []
with open("./models/coco.names", 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

## Names of classes imagenet
classes_imagenet = "./data/imagenet_classes.txt"

## Page names for streamlit sidebar
PAGES = [
    'Learn Animals Names',
    'Learn Animals',
    'Learn Counting',
    'About Us'
]