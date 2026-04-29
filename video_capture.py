import cv2 #OpenCV library

video_capture = cv2.VideoCapture("videoplayback.mp4") #load video file

while video_capture.isOpened():
    success, frame = video_capture.read()

    if not success:   #if no frame is returned , means video is finished
        break

    cv2.imshow("Traffic video", frame) #show the current frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release() #release video
cv2.destroyAllWindows()    


