
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pandas as pd
from sklearn.metrics import accuracy_score 
import itertools
import math
import time
import argparse


sys.path.append("..")


from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'


CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','mscoco_label_map.pbtxt')


NUM_CLASSES = 1


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)


image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')


num_detections = detection_graph.get_tensor_by_name('num_detections:0')


cap = cv2.VideoCapture('23 (1).mp4')
ret, frame = cap.read()

fromCenter=False
showCrosshair=False
r = cv2.selectROI(frame,fromCenter,showCrosshair)
print(r)
cv2.destroyWindow("ROI selector")

while(True):

   
    ret, frame = cap.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)

    # selecting class 'person' only
   # indices = np.argwhere(classes == 3)
   # boxes = np.squeeze(boxes[indices])
   # scores = np.squeeze(scores[indices])
   # classes = np.squeeze(classes[indices])

    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        boxes,
        (classes).astype(np.int32),
        scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4,
        min_score_thresh=0.5)
    

    # points for roi rectangle
    points=r
    cvbox=list(points)
    #Image frame width and height
    im_width,im_height=1280,720
    cv2.rectangle(frame,r,(255, 0, 0), 2)
    count=0
    for i,boxy in enumerate(np.squeeze(boxes)):
        #print(type(boxy))
        
        if( np.squeeze(scores)[i] >0.9):
            #print(np.squeeze(scores)[i])
            #defining iou box
           xa = max(int(boxy[0]*im_height),cvbox[0])
           ya = max(int(boxy[1]*im_width),cvbox[1])
           xb = min(int(boxy[2]*im_height),cvbox[2])
           yb = min(int(boxy[3]*im_width),cvbox[3])
           
           # area  calculation
           interarea = max(0, xb - xa + 1) * max(0, yb - ya + 1)           
           boxaarea = (int(boxy[2]*im_height) - int(boxy[0]*im_height) + 1 *
            int(boxy[3]*im_width) - int(boxy[1]*im_width) + 1)
           boxbarea = (cvbox[2] - cvbox[0] + 1) * (cvbox[3] - cvbox[1] + 1)
           
           #IOU calculation
           iou = interarea / float(boxaarea + boxbarea - interarea)
           
           print(iou)
           
           if(iou>0.01):
            count=count+1
            
    #Text parameters       
    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (50, 50)
    fontScale = 1

    cv2.putText(frame,"count:{}".format(count),org,font,fontScale,(255, 0, 0),2)
    cv2.imshow('Objectdeector',frame)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

