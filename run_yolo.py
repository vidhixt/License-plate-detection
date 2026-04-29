from ultralytics import YOLO
import cv2


yolo_model = YOLO("yolov8n.pt") # Load YOLO model

video_capture = cv2.VideoCapture("videoplayback.mp4")

while video_capture.isOpened():
    success, frame = video_capture.read()

    if not success:
        break

    
    detection_results = yolo_model(frame)# Run YOLO detection

    annotated_frame = detection_results[0].plot() # Draw bounding boxes on frame

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()