# import useful libraries
import os
import numpy as np
import cv2
from yolo_utils import *

# Camera
frame_width = 160
frame_height = 120
cam_idx = 0

def main():
    # set up video
    video = cv2.VideoCapture(cam_idx)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    ## --------- CODE NEED TO BE INSERT -----------
    # test our function read_classes
    # img_file = '../traffic-signs-detection/data/obj.names'
    # classNames = read_classes(img_file)
    # print("Classes' names :\n", classNames)

    # load the model config and weights
    modelConfig_path = './cfg/yolov4-rds.cfg'
    modelWeights_path = '../traffic-signs-detection/weights/yolov4-rds_best_2000.weights'

    # read the model cfg and weights with the cv2 DNN module
    neural_net = cv2.dnn.readNetFromDarknet(modelConfig_path, modelWeights_path)
    # set the preferable Backend to GPU
    neural_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    neural_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # defining the input frame resolution for the neural network to process
    network = neural_net
    height, width = 160,120

    # confidence and non-max suppression threshold for this YoloV3 version
    confidenceThreshold = 0.5
    nmsThreshold = 0.2
    ## --------------------------------------------

    while True:
        ret, original_frame = video.read()
        frame = cv2.resize(original_frame, (416, 416))
        ## --------- CODE NEED TO BE INSERT -----------
        ## load image
        # using convert_to_blob function : 
        outputs = convert_to_blob(frame, network, 416, 416)    
        # apply object detection on the video file
        bounding_boxes, class_objects, confidence_probs = object_detection(outputs, frame, confidenceThreshold)  
        print(bounding_boxes)
        print(class_objects)
        print(confidence_probs)
        ## 

if __name__ == "__main__":
    main()