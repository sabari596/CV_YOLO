import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model (n = nano version)
model = YOLO("yolov8n.pt")

# Load video file
cap = cv2.VideoCapture("Bottle_video2.mp4")

# A set to store unique object IDs detected during tracking
unique_ids = set()

# Infinite loop to read video frames
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If no frame is returned, end of video reached
    if not ret:
        break

    # Run tracking on the frame
    # classes=[39] means track only class ID 39 (bottle in COCO dataset)
    # persist=True keeps object IDs consistent across frames
    results = model.track(frame, classes=[39], persist=True, verbose=False)

    # Generate an annotated frame with bounding boxes and IDs
    annotated_frame = results[0].plot()

    # Check if objects are detected and they have assigned track IDs
    if results[0].boxes and results[0].boxes.id is not None:

        # Get the array of IDs for detected objects
        ids = results[0].boxes.id.numpy()

        # Add each detected ID to the unique set
        for oid in ids:
            unique_ids.add(int(oid))

        # Draw the count on the frame
        cv2.putText(
            annotated_frame,
            f"Count: {len(unique_ids)}",   # Number of unique tracked objects
            (10, 30),                      # Position
            cv2.FONT_HERSHEY_SIMPLEX,     # Font
            1,                             # Scale
            (0, 255, 0),                   # Color (Green)
            2                              # Thickness
        )

    # Show the processed frame
    cv2.imshow("Object Tracking", annotated_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and close windows
cap.release()
cv2.destroyAllWindows()