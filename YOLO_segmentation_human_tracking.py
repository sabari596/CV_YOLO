import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")

# Load video
cap = cv2.VideoCapture("street_walking1.mp4")   # use 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Track objects (class 0 = person)
    results = model.track(
        source=frame,
        classes=[0],        # only persons
        persist=True,       # keep same ID across frames
        verbose=False
    )

    annotated_frame = frame.copy()

    for r in results:
        if (
            r.masks is not None and
            r.boxes is not None and
            r.boxes.id is not None
        ):
            masks = r.masks.data.numpy()     # segmentation masks
            boxes = r.boxes.xyxyn.numpy()    # bounding boxes (normalized)
            ids = r.boxes.id.numpy()         # object IDs

            for i, mask in enumerate(masks):
                person_id = ids[i]

                # Get bounding box
                x1, y1, x2, y2 = boxes[i]
                x1 = int(x1 * frame.shape[1])
                x2 = int(x2 * frame.shape[1])
                y1 = int(y1 * frame.shape[0])
                y2 = int(y2 * frame.shape[0])

                # Resize mask to match frame size
                mask_resized = cv2.resize(mask.astype(np.uint8) * 255,
                                          (frame.shape[1], frame.shape[0]))

                # Find contours
                contours, _ = cv2.findContours(
                    mask_resized,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                # Draw contour
                cv2.drawContours(annotated_frame, contours, -1, (0, 0, 255), 2)

                # Draw ID text
                cv2.putText(
                    annotated_frame,
                    f"ID: {int(person_id)}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )

    # Display output
    cv2.imshow("Object Tracking with Segmentation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
