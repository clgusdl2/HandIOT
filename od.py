import cv2
import os
import numpy as np
import handTrackingModule_default as htm
thres_person = 0.60 # Threshold to detect object
nms_threshold = 0.2
cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)
# cap.set(10,150)
os.chdir('C:/coding/vscode_files/handTrackingProject')
classNames= ['person']
# classFile = 'coco.names'
# with open(classFile,'rt') as f:
#     classNames = f.read().rstrip('n').split('n')

#print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
hands = htm.handDetector(detectionCon=0.8)
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
detected = 0
poseon = 0
while True:
    success,img = cap.read()
    # width = int(cap.get(3))
    # height = int(cap.get(4))
    # image = np.zeros(img.shape, np.uint8)
    # smaller_f = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    # image[:height//2, :width//2] = smaller_f
    classIds, confs, bbox = net.detect(img,confThreshold=thres_person)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    #print(type(confs[0]))
    #print(confs)

    indices = cv2.dnn.NMSBoxes(bbox,confs,thres_person,nms_threshold)
    #print(indices)
    for i in indices:
        if classIds[i][0] == 1:
            i = i[0]
            detected = 1
            box = bbox[i]
            x,y,w,h = box[0],box[1],box[2],box[3]
            
            # cv2.putText(img,classNames[classIds[i][0]-1].upper(),(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            image = np.zeros((h,w), np.uint8)
            image = img[y:h+y, x:x+w]
            image = cv2.resize(image,dsize=(720,640),interpolation=cv2.INTER_AREA)
            # cv2.rectangle(image, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
            handImg, ifHands = hands.findHands(image)
            if ifHands == 1:
                if poseon == 0:
                    poseon = 1
                if poseon == 1:
                    print('poseDetection on')
                    poseon = 2


            # cropped = img[box[1]:box[3], box[0]:box[2]]
            # resizeCrop = cv2.resize(cropped,(720,480))
        else:
            detected = 2
    if detected == 1 or detected == 2:
        cv2.imshow('Output', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break