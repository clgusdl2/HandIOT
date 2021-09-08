import cv2
import numpy as np
import os
import HandTrackingModule_volume as htm
import math
from time import *
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import tensorflow as tf

### init
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                                # use tensorflow-cpu
os.system('net use \\\\192.168.82.34\\realsense /user:realsense 123468') # for use raspi network drive
os.chdir('C:/coding/vscode_files/handTrackingProject') # dir set


### detection setup
mp_hands = mp.solutions.hands           # Hand detection model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def extract_keypoints(results):
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 1:                             # only one hand detected
            for _, hand_handedness in enumerate(results.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                if handedness_dict['classification'][0]['label'] == 'Left':    # distinguish right or left hand
                    rh = np.zeros(21*3)
                    lh = np.array([[res.x, res.y, res.z] for _, res in enumerate(results.multi_hand_landmarks[0].landmark)]).flatten() # each landmark has 3 factors(x, y, z)            
                    # print(lh)
                else:
                    rh = np.array([[res.x, res.y, res.z] for _, res in enumerate(results.multi_hand_landmarks[0].landmark)]).flatten()
                    lh = np.zeros(21*3)
                    # print(rh)
        elif len(results.multi_hand_landmarks) == 2:
            rh = np.array([[res.x, res.y, res.z] for _, res in enumerate(results.multi_hand_landmarks[0].landmark)]).flatten()
            lh = np.array([[res.x, res.y, res.z] for _, res in enumerate(results.multi_hand_landmarks[1].landmark)]).flatten()
    else:
        rh = np.zeros(21*3)
        lh = np.zeros(21*3)
    return np.concatenate([rh, lh])


def volume_brightness_mode(deviceType):
    volPer=0
    hand = 0
    # print(deviceType)
    pTime = 0 #이전 시간
    cTime = 0 #현재 시간
    hand = 0               # hand deteced : 1
    pVol = 0
    signalSign = 0
    ctlFlag = 0
    leftOrRight = 'none'
    hand = 0
    initTime = time()
    while True:
        _, img = cap.read()

        #FIND HAND
        if time() - initTime > 1:
            img = detector.findHands(img,draw=False)
            lmList,bbox = detector.findPosition(img,draw=False)
            
            if len(lmList) !=0:
                for _, hand_handedness in enumerate(detector.results.multi_handedness):
                    handedness_dict = MessageToDict(hand_handedness)
                    if handedness_dict['classification'][0]['label'] == 'Left':
                        leftOrRight = 'Left'
                    else:
                        leftOrRight = 'Right'
                # print(leftOrRight)
                hand = 1
                handImg = []
                handImg = img[bbox[1]-20:bbox[3]+20, bbox[0]-20:bbox[2]+20]
                handImg = cv2.resize(handImg,dsize=(480,480),interpolation=cv2.INTER_AREA)
                handImg = detector2.findHands(handImg,draw=False)
                suc,_ = detector2.findPosition(handImg,draw=False)
                if len(suc) != 0:
                    #FInd Distance between index and Thumb 
                    length,handImg, _ = detector2.findDistance(4,8,handImg)
                    #Convert Volume               
                    volPer = np.interp(length,[50,300],[0,100])

                    
                    #Reduce Resolution to make it smoother
                    smoothness = 2 #몇씩 올릴건지?? 
                    volPer =smoothness * round(volPer/smoothness)
                    
                    
                    #Check fingers up 
                    fingers = detector2.fingersUp()
                    # print(fingers)
                    

                    #약지 
                    if leftOrRight == 'Right':
                        if fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                            break
                    else:
                        if not fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                            break
                    #Drawings 

                    x1,y1= lmList[4][1],lmList[4][2] #엄지좌표 끝 
                    x2,y2= lmList[8][1],lmList[8][2] #검지좌표 끝
                    length = math.hypot(x2-x1,y2-y1) #사이거리볼륨조절

                    if  not fingers[4]:
                        if ctlFlag == -1:
                            pVol = volPer
                        ctlFlag = 1
                    else:
                        ctlFlag = -1
                    if ctlFlag == 1:
                        if pVol - volPer < 0 and volPer % 4 == 0:
                            if signalSign == 1:
                                sendToRaspi(deviceType,'Up')
                            pVol = volPer
                            signalSign = 1
                        elif pVol - volPer > 0 and volPer % 4 == 0:
                            if signalSign == -1:
                                sendToRaspi(deviceType,'Down')
                            pVol = volPer
                            signalSign = -1


            #fps 출력  
            # cTime = time()
            # fps = 1/(cTime-pTime)
            # pTime = cTime
            # cv2.putText(img, f'FPS: {int(fps)}',(20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

            if hand == 1:
                cv2.imshow("volume_mode",handImg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyWindow("volume_mode")


def sendToRaspi(deviceType, command):
    commandFile = open('//192.168.82.34/realsense/Command.txt','w')
    commandFile.write(deviceType)
    commandFile.write(' ')
    commandFile.write(command)
    commandFile.close()



DATA_PATH = os.path.join(os.getcwd(), 'MP_DATA') # cur path + 'MP_DATA'

# Actions that we try to detect
actionsDict = {'Hello':1, 'TV':2, 'Light':2, 'Brightness':3, 'Channel':3, 'Volume':3, 'On':99, 'Off':99, 'Next':4, 'Prev':4, 'OK':99}       # use dict to sequence control
actions = np.array(['Hello', 'TV', 'Channel', 'Volume', 'On', 'Off', 'Next', 'Prev', 'OK', 'Light', 'Brightness'])                          # can add actions


### load tflite model
model = tf.lite.Interpreter(model_path='./model.tflite')
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()
input_shape = input_details[0]['shape']

### volume detection object 
detector = htm.handDetector(detectionCon=0.7, maxHands=1)
detector2 = htm.handDetector(detectionCon=0.7, maxHands=1)


# detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.95         # min confidence of prediction
prev_predicted = 'none'
actionVal = 0
startTime = 0
curTime = 0 
timerFlag = 0
sleepMode = 0
helloFlag = 0
pTime = 0
channelPTime = 0
channelCTime = 0
channelFlag = 0
fpsPtime = 0
FPS = 15
command = 'None'
deviceType = 'None'
channelStat = 'None'

cap = cv2.VideoCapture(0)
cap.set(10,350)         # cam brightness

# Set mediapipe model 
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        fpsCTime = time() - fpsPtime
        if fpsCTime > 1./FPS:
            fpsPtime = time()
                
            # Make detections
            image, results = mediapipe_detection(frame, hands)
            
            
            # Draw landmarks
            if results.multi_hand_landmarks:
                timerFlag = 0
                sleepMode = 0
                for handLms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)
            else:
                # sleep mode (decrease FPS to save memory)
                if timerFlag == 0:
                    startTime = time()
                    timerFlag = 1
                curTime = time()
                if curTime - startTime > 5:
                    print('timeout! sleep mode')
                    sleep(0.3)
                    sleepMode = 1
                    actionVal = 0
            
            # 2. Prediction logic
            if results.multi_hand_landmarks:

                if sleepMode != 1:
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    # predict gesture with 30 frames
                    if len(sequence) == 30:
                        input_data = np.array(np.expand_dims(sequence, axis=0), dtype=np.float32)
                        model.set_tensor(input_details[0]['index'], input_data)
                        model.invoke()
                        res = model.get_tensor(output_details[0]['index'])
                        res = res[0]
                        
                        # channel next and prev logic
                        if prev_predicted == 'Next' or prev_predicted == 'Prev':
                            if max(res) > threshold:
                                if actions[np.argmax(res)] == 'Prev' or actions[np.argmax(res)] == 'Next':
                                    if channelFlag == 0:
                                        channelPTime = time()
                                        channelFlag = 1
                                    if time() - channelPTime > 1.3:
                                        if prev_predicted != channelStat:
                                            channelStat = actions[np.argmax(res)]
                                        else:
                                            print('{} channel'.format(actions[np.argmax(res)]))
                                            channelFlag = 0
                                            command = channelStat


                        # basic detection sequence logic
                        if prev_predicted != actions[np.argmax(res)]:
                            if max(res) > threshold:
                                actionName = actions[np.argmax(res)]
                                if actionName == 'OK':
                                    # back to device selecting stage
                                    actionVal = 1
                                    prev_predicted = actionName
                                    print(actionName)
                                elif actionsDict[actionName] - 1 == actionVal:
                                    actionVal += 1
                                    if actionVal == 1 and helloFlag == 0:
                                        # 명령 시작 표시
                                        helloFlag = 1
                                    elif actionName == 'Volume' or actionName == 'Brightness':
                                        cv2.destroyWindow('Idle')
                                        volume_brightness_mode(deviceType)
                                        actionVal = 1
                                    elif actionName == 'Next' or actionName == 'Prev':
                                        print('{} channel'.format(actionName))
                                        actionVal -= 1
                                        command = actionName
                                    elif actionVal == 2:
                                        deviceType = actionName
                                    prev_predicted = actionName
                                    print(actionVal, actionName)
                                elif actionName == 'On' or actionName == 'Off':
                                    if actionVal == 2:
                                        command = actionName
                                        actionVal = 1
                                        prev_predicted = actionName
                                        print(actionVal, actionName)
                                if actionVal == 2:
                                    # to improve device selection
                                    sequence = []

                        predictions.append(np.argmax(res))

                    if command != 'None':
                        #send to raspi using samba
                        sendToRaspi(deviceType,command)
                        command = 'None'

            # show FPS on the screen
            cTime = time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            cv2.putText(image, f'FPS: {int(fps)}',(20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

            
            cv2.imshow('Idle', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
#################################################################################
