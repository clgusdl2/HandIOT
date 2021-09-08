import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 
                                        self.detectionCon, self.trackCon) # Same as default parameter
        
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(self.results.multi_hand_landmarks)
        self.ifHands = 0
        if self.results.multi_hand_landmarks:
            self.ifHands = 1
            # for handLms in self.results.multi_hand_landmarks:
            #     if draw:
            #         self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img, self.ifHands

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            print(len(self.results.multi_hand_landmarks))
            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)

        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        # if len(lmList) != 0:9
        #     print(lmList)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime


        cv2.imshow("Image", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    print(lmList)

if __name__ == "__main__":
    main()