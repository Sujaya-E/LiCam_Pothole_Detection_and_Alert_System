import cv2
import pyttsx3
from ultralytics import YOLO

# Load the YOLOv8 model
yolo = YOLO(r"/content/drive/MyDrive/Pothole-codes/yolov10n_train.pt")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize a dictionary to store counted objects and their bounding box coordinates
object_tracker = {}
trackers = {}  # This will hold the individual trackers for each object

# Open webcam
cap = cv2.VideoCapture(1)

# This will keep track of the next object ID
next_object_id = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the current frame
    results = yolo.predict(frame)  # Use the 'predict' method for YOLOv8

    # Extract the bounding boxes and confidence from the results
    bbox_xywh = []
    confidences = []
    new_detections = []

    # Collect detected bounding boxes
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        w = x2 - x1
        h = y2 - y1
        bbox_xywh.append([x1, y1, w, h])
        confidences.append(conf)

        # Create a tuple for the bounding box (top-left and bottom-right coordinates)
        detection = (x1, y1, x2, y2)

        # Check if the object has already been tracked using a tracker
        object_detected = False

        for obj_id, tracker in trackers.items():
            success, tracked_bbox = tracker.update(frame)
            if success:
                # If the tracker successfully tracked the object, check if it's within a valid range
                x1, y1, w, h = [int(v) for v in tracked_bbox]
                detected_bbox = (x1, y1, x1 + w, y1 + h)

                # Check if the current bounding box is similar to the tracked one (allow for small movement)
                if (abs(detection[0] - detected_bbox[0]) < 50 and abs(detection[1] - detected_bbox[1]) < 50):  # Overlap check
                    object_detected = True
                    break

        # If the object is not already being tracked, add a new tracker
        if not object_detected:
            # Use CSRT tracker instead of KCF
            tracker = cv2.TrackerCSRT_create()  # Create a CSRT tracker
            trackers[next_object_id] = tracker
            tracker.init(frame, (x1, y1, w, h))  # Initialize the tracker with the bounding box

            # Store the object with its bounding box coordinates
            object_tracker[next_object_id] = detection
            new_detections.append(detection)

            # Trigger text-to-speech for new detection
            engine.say(f"pothole detected, please slow down")
            engine.runAndWait()

            next_object_id += 1  # Increment the object ID

    # Optional: Display the frame with detections and tracking information
    for obj_id, tracker in trackers.items():
        success, tracked_bbox = tracker.update(frame)
        if success:
            x1, y1, w, h = [int(v) for v in tracked_bbox]
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)  # Draw the bounding box

    # Show the frame with bounding boxes
    cv2.imshow("Webcam Feed", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

print("Final object counts:", object_tracker)