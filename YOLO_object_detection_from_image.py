# Note: To run this script, ensure you have the 'ultralytics' package installed:
# pip install ultralytics

import cv2                     # Import OpenCV for image loading and display
from ultralytics import YOLO   # Import YOLO class from the ultralytics package

# Load the YOLOv8 nano model (will download yolov8n.pt the first time if not present)
model = YOLO("yolov8n.pt")

# Read the input image from file
image = cv2.imread("bird.jpg") # Replace "bird.jpg" with your image

# Safety check: if the image failed to load, raise an error
if image is None:
    raise FileNotFoundError("Image not found. Check the path.")

# Run object detection on the image
results = model(image)

# Draw bounding boxes, labels, and confidence scores on a copy of the image
annotated_image = results[0].plot()

# Display the annotated image in a window titled "Annotated Image"
cv2.imshow("Annotated Image", annotated_image)

# Wait indefinitely for a key press (0 means wait forever)
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()

