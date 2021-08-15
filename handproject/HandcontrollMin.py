from cv2 import cv2 #그냥 import cv2 해서 에러가 나면 이렇게 해주자.
import mediapipe as mp
import time
import HandTrackingModule as htm
import math
import numpy as np
#볼륨조절 모듈  
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities,IAudioEndpointVolume
cap = cv2.VideoCapture(0) #자신의 웹캠 장치 번호를 적어야 한다. 0은 첫번째 장치를 사용하겠다는 것

pTime = 0 #이전 시간
cTime = 0 #현재 시간


detector = htm.handDetector(detectionCon=0.7)


#pycaw 볼륨조절 모듈 이니셜라이즈
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange() # min -65 max 0 무시 
#파라미터/미니멈/맥시멈
#volume.SetMasterVolumeLevel(0, None)
#볼륨조절 -65~0 = 0~100
minVol = volRange[0]
maxVol = volRange[1]
print(minVol)
val=0
volBar=400
volPer=0




while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    if len(lmList) !=0:
        #print(lmList[4],lmList[8])


        x1,y1= lmList[4][1],lmList[4][2] #엄지좌표 끝 
        x2,y2= lmList[8][1],lmList[8][2] #검지좌표 끝
        cx,cy= (x1+x2)//2 , (y1+y2) //2
        cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),15,(255,0,255),cv2.FILLED)
        cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        #동그라미 만드는 부분
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        #선긋기
        length = math.hypot(x2-x1,y2-y1) #사이거리볼륨조절
        #print(length)

        #Hand range 50 - 300 
        # Volume Range is from -65 - 0 
        # np 이용 
        vol = np.interp(length,[50,300],[minVol,maxVol])
        volBar = np.interp(length,[50,300],[400,150])
        volPer = np.interp(length,[50,300],[0,100])
        print(int(length),vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length<50: #50이하 거리 초록색점 버튼같은느낌도 만들수있을수도?
            cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)

    cv2.rectangle(img,(50,150),(85,400),(255,0,0),3)
    cv2.rectangle(img,(50,int(volBar)),(85,400),(255,0,0),cv2.FILLED)
    cv2.putText(img, f' {int(volPer)} %',(20, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
    #fps 출력 
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}',(20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
    #이미지 텍스트 좌표 글꼴 폰트스케일 컬러 두께 라인유형 

    cv2.imshow("IMG",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break