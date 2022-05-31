#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import os

from .config import *
# set conf, nms thresholds,inp width/height
# confThreshold = 0.55
# nmsThreshold = 0.40
# inpWidth = 416
# inpHeight = 416
# VEHICLE_TAG = "vehicle"
# PERSON_TAG = "person"

file_path = os.path.dirname(__file__)

# load names of classes and turn that into a list
classesFile = os.path.join(file_path,"coco.names")
classes = None

vehicles_objects = ["bicycle","car","motorbike","aeroplane","bus","train","truck","boat"]

with open(classesFile,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# model configuration
modelConf = os.path.join(file_path,'yolov3.cfg')
modelWeights = os.path.join(file_path,'yolov3.weights')

def load_model():
    # set up the net
    net = cv.dnn.readNetFromDarknet(modelConf, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    return net

# function that processes frames 
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIDs = []
    confidences = []
    boxes = []

    # detected boxes associated to person and vehicles
    boxes_dict = {VEHICLE_TAG:[],
                  PERSON_TAG:[]}
    # iterate 
    for out in outs:
        for detection in out:
            
            scores = detection [5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confThreshold:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)

                width = int(detection[2]* frameWidth)
                height = int(detection[3]*frameHeight )

                left = int(centerX - width/2)
                top = int(centerY - height/2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # create a list of indices
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    # iterate through indices to draw bounding boxes around detected objects
    #left -> x ; top -> y # remember to invert when acces to image
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        object_name = classes[classIDs[i]]
        if object_name in vehicles_objects:
            object_name = VEHICLE_TAG
        
        if object_name == VEHICLE_TAG or object_name == PERSON_TAG :
            boxes_dict[object_name].append([(left,top),width,height])


        frame = drawPred(frame,classIDs[i], confidences[i], left, top, left + width, top + height)
    return frame,boxes_dict

# function for drawing bounding boxes around detected objects
def drawPred(frame,classId, conf, left, top, right, bottom):
    
    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        object_name = classes[classId]
        if object_name in vehicles_objects:
            object_name = VEHICLE_TAG
        label = '%s:%s' % (object_name, label)

    # Draw a bounding box.
    if object_name == VEHICLE_TAG:
        color_rect = (0, 0, 255)
    else:
        color_rect = (0, 255, 255)
    cv.rectangle(frame, (left, top), (right, bottom), color_rect, 3)
    
    #cv.putText(frame, label, (left,top), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

def predict(net,frame):
    
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)
   
    # set the net
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    return outs


    # function for getting the names outputted in the output layers of the net
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
   

    # Get the names of the output layers, i.e. the layers with unconnected outputs
    #return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return  [layersNames[i-1] for i in net.getUnconnectedOutLayers()] 

if __name__ == "__main__":
    net = load_model()

    # process inputs
    winName = 'DL OD with OpenCV'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)
    cv.resizeWindow(winName, 1000,1000)

    # read video input
    cap = cv.VideoCapture('rec.mov')

    # program loop
    while cv.waitKey(1) < 0:

        # get frame from video
        hasFrame, frame = cap.read()
        
        # create a 4D blob from the frame
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)
        
        # set the net
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))

        frame, _ = postprocess(frame, outs)

        # show the image
        cv.imshow(winName, frame)