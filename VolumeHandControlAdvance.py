from cv2 import cv2 #그냥 import cv2 해서 에러가 나면 이렇게 해주자.
import mediapipe as mp
from time import *
import HandTrackingModule_volume as htm
import math
import numpy as np
from google.protobuf.json_format import MessageToDict

#볼륨조절 모듈  
cap = cv2.VideoCapture(0) #자신의 웹캠 장치 번호를 적어야 한다. 0은 첫번째 장치를 사용하겠다는 것

pTime = 0 #이전 시간
cTime = 0 #현재 시간


detector = htm.handDetector(detectionCon=0.9,maxHands=1)
detector2 = htm.handDetector(detectionCon=0.8,maxHands=1)

#pycaw 볼륨조절 모듈 이니셜라이즈

#volume.GetMute()
#volume.GetMasterVolumeLevel()
#파라미터/미니멈/맥시멈
#volume.SetMasterVolumeLevel(0, None)
#볼륨조절 -65~0 = 0~100

val=0
volBar=400
volPer=0
area=0
colorVol=(255,0,0)
hand = 0


pTime = 0 #이전 시간
cTime = 0 #현재 시간
# sleep(1)
hand = 0
pVol = 0
signalSign = 0
ctlFlag = 0
leftOrRight = 'none'
while True:
    success, img = cap.read()

    #FIND HAND

    img = detector.findHands(img,draw=False)
    lmList,bbox = detector.findPosition(img,draw=False)
    
    if len(lmList) !=0:
        for _, hand_handedness in enumerate(detector.results.multi_handedness):
            handedness_dict = MessageToDict(hand_handedness)
            if handedness_dict['classification'][0]['label'] == 'Left':
                leftOrRight = 'Left'
            else:
                leftOrRight = 'Right'

        hand = 1
        handImg = []
        handImg = img[bbox[1]-20:bbox[3]+20, bbox[0]-20:bbox[2]+20]
        handImg = cv2.resize(handImg,dsize=(480,480),interpolation=cv2.INTER_AREA)
        handImg = detector2.findHands(handImg,draw=True)
        suc,_ = detector2.findPosition(handImg,draw=True)
        if len(suc) != 0:
            # print(suc[4][1],suc[4][2])
            #Filter based on size

            #FInd Distance between index and Thumb 
            length,handImg,lineinfo = detector2.findDistance(4,8,handImg)
            #Convet Volume               
            volBar = np.interp(length,[50,220],[400,150])
            volPer = np.interp(length,[50,300],[0,100])

            
            #Reduce Resolution to make it smoother
            smoothness = 2 #몇씩 올릴건지?? 
            volPer =smoothness * round(volPer/smoothness)
            
            
            #Check fingers up 
            fingers = detector2.fingersUp()
            # print(fingers)
            #if pinky is down set volume 
            

            #약지 
            if leftOrRight == 'Left':
                if fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                    break
            else:
                if not fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                    break
            #Drawings 

            x1,y1= lmList[4][1],lmList[4][2] #엄지좌표 끝 
            x2,y2= lmList[8][1],lmList[8][2] #검지좌표 끝
            # cx,cy= (x1+x2)//2 , (y1+y2) //2
            # cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            # cv2.circle(img,(x2,y2),15,(255,0,255),cv2.FILLED)
            # cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
            # #동그라미 만드는 부분
            # cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
            #선긋기
            length = math.hypot(x2-x1,y2-y1) #사이거리볼륨조절
            # print(length)

            #Hand range 50 - 300 
            # Volume Range is from -65 - 0 
            # np 이용 
            if  not fingers[4]:
                if ctlFlag == -1:
                    pVol = volPer
                ctlFlag = 1
            else:
                ctlFlag = -1
            print(volPer)
            if ctlFlag == 1:
                if pVol - volPer < 0 and volPer % 4 == 0:
                    if signalSign == 1:
                        print('send vol up signal')
                    pVol = volPer
                    signalSign = 1
                elif pVol - volPer > 0 and volPer % 4 == 0:
                    if signalSign == -1:
                        print('send vol dn signal')
                    pVol = volPer
                    signalSign = -1
            

                #if length<50: #50이하 거리 초록색점 버튼같은느낌도 만들수있을수도?

            #Drawings
            # cv2.rectangle(handImg,(50,150),(85,400),(255,0,0),3)
            # cv2.rectangle(handImg,(50,int(volBar)),(85,400),(255,0,0),cv2.FILLED)
            # cv2.putText(handImg, f' {int(volPer)} %', (60, 300), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
            # print(volPer)
    #fps 출력  
    cTime = time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # cv2.putText(img, f'FPS: {int(fps)}',(20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
    #이미지 텍스트 좌표 글꼴 폰트스케일 컬러 두께 라인유형 
    if hand == 1:
        cv2.imshow("volume_mode",handImg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow("volume_mode")