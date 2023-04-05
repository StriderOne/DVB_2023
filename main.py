import cv2

cap = cv2.VideoCapture('data/cam_video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame_thresh = cv2.threshold(frame_gray, 100, 255, cv2.THRESH_BINARY_INV)

    cnt, _ = cv2.findContours(frame_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnt, key=cv2.contourArea)
    # cv2.drawContours(frame, cnt, -1, (255,0,0))
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])           
    cY = int(M["m01"] / M["m00"])
    delta = 20
    cv2.line(frame, (cX - delta, cY), (cX + delta, cY), (0, 0, 255))
    cv2.line(frame, (cX, cY - delta), (cX, cY + delta), (0, 0, 255))
    cv2.imshow('frame', frame)
    if cv2.waitKey(30) == ord('q'):
        break