import cv2
import numpy as np
# 웹캠 비딣 가져오기 , 일반 저장된 비디오로 감지하고 싶다면 "파일이름" 작성
cap = cv2.VideoCapture(0)
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280, 720))

ret, frame1 = cap.read() # 첫번째 프레임
ret, frame2 = cap.read() # 두번째 프레임
print(frame1.shape)

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2) # 첫번쨰 프레임과 다음 프레임 차이 계산
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # 움직이는 프레임 감지
    blur = cv2.GaussianBlur(gray, (5,5), 0)  # 가우시안 처리
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) # 임계각 결정
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour) # 윤곽을 그리는 대신직사각형으로 움직임 감지 표시

        if cv2.contourArea(contour) < 900: # 움직이는 범위가 매우 사소할경우 감지 하지 않기로 한다
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2) # 사작형 크기 , 색 , 두께 잡기
        cv2.putText(frame1, "Status: {}".format('AVi'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3) # 텍스트 작성
        cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2) # 윤관석 그리기

    image = cv2.resize(frame1, (1280, 720))
    out.write(image)
    cv2.imshow("feed", frame1)
    frame1 = frame2 # 다음 프레임 계산을 위해 현재 프레임저장하기
    ret, frame2 = cap.read()

# eSC 입력시 종료
    if cv2.waitKey(40) == 27:
        break

# 종료 하기
cv2.destroyAllWindows()
cap.release()
out.release()

##gdgd
print('a')