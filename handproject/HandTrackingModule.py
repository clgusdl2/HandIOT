import cv2
import mediapipe as mp
import time
import math


class handDetector(): # 시작과 함께 다음을 진행한다.
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode 
        # static_image_mode : 한손이 감지되면 추가 손에 대해서 무시함.(max_num_hands와 연동됨) defalt값은 False
        self.maxHands = maxHands
        # max_num_hands : 최대 감지 가능한 손을 입력. defalt값은 2
        self.detectionCon = detectionCon
        # min_detection_confidence : 감지에 성공한 것으로 간주될 손 모델의 최소 신뢰도 값. 
        # 0.5~ 1 사이
        self.trackCon = trackCon
        # min_tracking_confidence : Tracking 감지에 성공한 것으로 간주될 손모델의 최소 신뢰도 값. dfalt값은 0.5(.0 ~ 1.)

    
        self.mpHands = mp.solutions.hands
         # mediapipe에서 handstracking을 위한 모듈을 변수로 설정한다.
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
        self.detectionCon, self.trackCon)
        # 이때의 변수가 가지는 속성은 위에서 설명한 것과 같이 설정하여 hands변수로 설정한다.
        self.mpDraw = mp.solutions.drawing_utils
        # tracking을 보여줄(그려줄) 모듈을 다음과 같은 변수에 설정한다.
        self.tipIds = [4, 8, 12, 16, 20]
    def findHands(self, img, draw=True):  
        # tracking할 손을 찾는 함수(인자로 사용할 Image를 적용시켜준다.)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # openCV함수인 이미지 색상 표현 방식 함수를 통하여 선택된 이미지를 RGB값을 역전하여(BGR) 적용시킵니다.
        # openCV의 함수는 통상 RGB로 사용하는 수치를 역으로 BGR로 사용한다.
        # 컴퓨팅에서의 행렬 생성을 생각하면 어떤 의미인지 알 수 있다.

        self.results = self.hands.process(imgRGB) 
        # 이때의 이미지 값을 앞서 지정해둔 hands변수에 process함수를 거쳐 입력해준다. 이때의 결과값을 results로 저장한다.
        # print(results.multi_hand_landmarks)
        # 저장되는 값들은 각 마디에 대한 위치 정보가 담겨있다.

        # 이 결과값에서 multi_hand_landmarks의 값이 존재한다면 다음을 진행한다
        if self.results.multi_hand_landmarks:   
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)
                #img/location draw 적용할 위치 지정  
                #data handLms drawing 사용될 데이터
                #target 그데이터에 사용될 목표 각 좌표들을 어떻게 사용하여 그릴지에 대한 정보가 들어있음
                # 이렇게 함수를 거치면 img위에 tracking이 그려져 반환됩니다.
        return img


    def findPosition(self, img, handNo=0, draw=True):
       # xList = []
       # yList = []
       # bbox = []
        lmList = [] # 손의 위치값을 담아 둘 리스트를 작성합니다.
        
        if self.results.multi_hand_landmarks: # 만약 결과값이 존재한다면 다음을 진행합니다.
            myHand = self.results.multi_hand_landmarks[handNo]  # 결과값의 0번 인덱스를 myHand변수로 저장합니다.

            for id, lm in enumerate(myHand.landmark):# myHand의 landmark값을 열거합니다.
                # print(id, lm)
                h, w, c = img.shape  # 이미지의 높이, 너비, 채널(이미지의 층)을 추출한다.
                # 이때 myHand의 tracking정보의 각 좌표는 화면상 비율로서 값이 존재합니다.
                # 이는 화면의 비율이 바뀌어도 지속적으로 따라다니기 위함입니다. 
                #그래서 화면상에 좌표로 바꾸기 위해서는 위에서 전체의 값에서 해당 비율을 곱해줍니다.
                cx, cy = int(lm.x * w), int(lm.y * h)
                #xList.append(cx)
                #yList.append(cy)
                # print(id, cx, cy)
                lmList.append([id, cx, cy]) # 이 때의 값들을 lmList에 저장합니다.
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
           # xmin, xmax = min(xList), max(xList)
           # ymin, ymax = min(yList), max(yList)
           # bbox = xmin, ymin, xmax, ymax
            
           # if draw:
            #    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),(bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
    
        return lmList
        #, bbox
    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    def findDistance(self, p1, p2, img, draw=True):

        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,
        (255,0,255),3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()