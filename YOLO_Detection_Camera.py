import cv2                       # OpenCV for working with video frames and display windows
from ultralytics import YOLO     # YOLOv8 model from ultralytics package

# Load the YOLOv8 nano model (small and fast; downloads on first use if not present)
model = YOLO("yolov8n.pt")

# Open the default camera (0 = primary webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # If frame capture failed, exit the loop
    if not ret:
        print("Failed to capture frame")
        break

    # Run YOLO inference on the current frame
    results = model(frame)

    # Draw bounding boxes, labels, etc. on a copy of the frame
    annotated_frame = results[0].plot()

    # Show the annotated frame in a window
    cv2.imshow("Live Camera Detection", annotated_frame)

    # Wait 1 ms for a key press; if 'q' is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera resource
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
