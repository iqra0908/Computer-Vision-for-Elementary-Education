#### Trains a yolo object detection model using a custom dataset in data/toys_data
#### Borrows heavily from the code in the class repo (YOLO notbebook)


import time
import cv2
import torch
import matplotlib.pyplot as plt
import urllib.request
from pathlib import Path
import os

from utils import plot_boxes, plot_boxes_webcam
from darknet.darknet import Darknet

# Download Darknet weights
path = Path()
filename = path/'darknet/weights/yolov3.weights'
url = 'https://pjreddie.com/media/files/yolov3.weights'
if not os.path.exists(filename):
    os.mkdir(path/'darknet/weights/')
    urllib.request.urlretrieve(url,filename)

# Set path for the cfg file
cfg_file = './darknet/yolov3.cfg'
# Set path for the pre-trained weights file
weight_file = './darknet/weights/yolov3.weights'
# Set path for the COCO object classes file
namesfile = '../data/coco/coco.names'

# Load the COCO class names
def load_class_names(namesfile):
    # Load the COCO class names
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names
    
class_names = load_class_names(namesfile)

# Set the NMS threshold.  You can adjust this to see what works best
nms_thresh = 0.2

# Set the IOU threshold. You can adjust this to see what works best
iou_thresh = 0.4


# Load the network architecture
model = Darknet(cfg_file)

# Load the pre-trained weights
model.load_weights(weight_file)

# Display the Darknet YOLOv3 architecture
#model.print_network()

# Calculates the IOU of two bounding boxes
def boxes_iou(box1, box2):

    # Get the Width and Height of each bounding box
    width_box1 = box1[2]
    height_box1 = box1[3]
    width_box2 = box2[2]
    height_box2 = box2[3]
    
    # Calculate the area of the each bounding box
    area_box1 = w
        area_box2 = width_box2 * height_box2
    
    # Find the vertical edges of the union of the two bounding boxes
    mx = min(box1[0] - width_box1/2.0, box2[0] - width_box2/2.0)
    Mx = max(box1[0] + width_box1/2.0, box2[0] + width_box2/2.0)
    
    # Calculate the width of the union of the two bounding boxes
    union_width = Mx - mx
    
    # Find the horizontal edges of the union of the two bounding boxes
    my = min(box1[1] - height_box1/2.0, box2[1] - height_box2/2.0)
    My = max(box1[1] + height_box1/2.0, box2[1] + height_box2/2.0)    
    
    # Calculate the height of the union of the two bounding boxes
    union_height = My - my
    
    # Calculate the width and height of the area of intersection of the two bounding boxes
    intersection_width = width_box1 + width_box2 - union_width
    intersection_height = height_box1 + height_box2 - union_height
   
    # If the the boxes don't overlap then their IOU is zero
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0

    # Calculate the area of intersection of the two bounding boxes
    intersection_area = intersection_width * intersection_height
    
    # Calculate the area of the union of the two bounding boxes
    union_area = area_box1 + area_box2 - intersection_area
    
    # Calculate the IOU
    iou = intersection_area/union_area
    
    return iou

# Non-maximum suppression of the other bounding boxes
def nms(boxes, iou_thresh):
    
    # If there are no bounding boxes do nothing
    if len(boxes) == 0:
        return boxes
    
    # Create a PyTorch Tensor to keep track of the detection confidence
    # of each predicted bounding box
    det_confs = torch.zeros(len(boxes))
    
    # Get the detection confidence of each predicted bounding box
    for i in range(len(boxes)):
        det_confs[i] = boxes[i][4]

    # Sort the indices of the bounding boxes by detection confidence value in descending order.
    # We ignore the first returned element since we are only interested in the sorted indices
    _,sortIds = torch.sort(det_confs, descending = True)
    
    # Perform Non-Maximal Suppression 
    best_boxes = []
    for i in range(len(boxes)):
        
        # Get the bounding box with the highest detection confidence first
        box_i = boxes[sortIds[i]]
        
        # Check that the detection confidence is not zero
        if box_i[4] > 0:
            
            # Save the bounding box 
            best_boxes.append(box_i)
            
            # Go through the rest of the bounding boxes in the list and calculate their IOU with
            # respect to the previous selected box_i. 
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]
                
                # If the IOU of box_i and box_j is higher than the given IOU threshold set
                # box_j's detection confidence to zero. 
                if boxes_iou(box_i, box_j) > iou_thresh:
                    box_j[4] = 0
                    
    return best_boxes

# Detects objects
def detect_objects(model, img, iou_thresh, nms_thresh):
    
    # Set the model to evaluation mode
    model.eval()
    
    # Convert the image from a NumPy ndarray to a PyTorch Tensor of the correct shape
    # The image is transposed, then converted to a FloatTensor of dtype float32, then
    # normalized to values between 0 and 1, and finally unsqueezed to have the correct
    # shape of 1 x 3 x 416 x 416
    img = torch.from_numpy(img.transpose(2,0,1)).float().div(255.0).unsqueeze(0)
    
    # Feed the image to the neural network with the corresponding NMS threshold
    # The first step in NMS is to remove all bounding boxes that have a very low
    # probability of detection. All predicted bounding boxes with a value less than
    # the given NMS threshold will be removed.
    list_boxes = model(img, nms_thresh)
    
    # Make a new list with all the bounding boxes returned by the neural network
    boxes = list_boxes[0][0] + list_boxes[1][0] + list_boxes[2][0]
    
    # Perform NMS step 2 on the boxes returned from the model to remove duplicates
    boxes = nms(boxes, iou_thresh)
    
    return boxes

    # Open webcam
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    
    # Convert the image to RGB
    original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the image to the input width and height of the first layer of the Darknet model  
    resized_image = cv2.resize(original_image, (model.width, model.height))

    # Detect objects in the image
    boxes = detect_objects(model, resized_image, iou_thresh, nms_thresh)

    #Plot the image with bounding boxes and corresponding object class labels
    img = plot_boxes_webcam(original_image, boxes, class_names, plot_labels = True)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('Results',rgb_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)