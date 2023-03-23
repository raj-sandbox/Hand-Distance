import cv2
import math
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import cvzone

vid = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8,maxHands=1)

x = [300,245,200,170,145,130,112,103,93,87,80,75,70,67,62,59,57]
y = [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
coff = np.polyfit(x,y,2)

while(True):
    boolean, frame = vid.read()
    hands = detector.findHands(frame,draw=False)

    if hands:
        lmList= hands[0]['lmList']
        x,y,w,h = hands[0]['bbox']
        x1,y1,z1 = lmList[5]
        x2,y2,z2 = lmList[17]

        distance = int(math.sqrt((y2-y1)**2+(x2-x1)**2))
        A,B,C = coff
        distanceCm = A*distance**2 + B*distance + C

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),3)
        cvzone.putTextRect(frame,f'{int(distanceCm)} cm',(x,y))

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindow()
