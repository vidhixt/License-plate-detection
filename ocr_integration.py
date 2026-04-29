from ultralytics import YOLO
import easyocr
import cv2


yolo_model = YOLO("license_plate_detector.pt") # Load YOLO model
ocr_reader = easyocr.Reader(['en'])

video_capture = cv2.VideoCapture("videoplayback.mp4")

while video_capture.isOpened():
    success, frame = video_capture.read()

    if not success:
        break

    
    detection_results = yolo_model(frame) # Run YOLO detection

    #annotated_frame = detection_results[0].plot() # Draw bounding boxes on frame

    for box in detection_results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
         # Draw rectangle manually
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cropped_plate = frame[y1:y2, x1:x2]
        cropped_plate = cv2.resize(cropped_plate, None, fx=2, fy=2)

        #ocr_results = ocr_reader.readtext(cropped_plate) # run ocr on cropped plate

        gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        resized_plate = cv2.resize(gray_plate, None, fx=2, fy=2)
        _, processed_plate = cv2.threshold(resized_plate, 150, 255, cv2.THRESH_BINARY)

        ocr_results = ocr_reader.readtext(processed_plate)

        #print("Aditya: ", ocr_results[0])
        for (bbox, text, confidence) in ocr_results:
            print("Detected Text:" ,text, "Confidence:", confidence)

            cv2.putText(frame, text, (x1,y1-1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Cropped", cropped_plate)

    cv2.imshow("Number plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

