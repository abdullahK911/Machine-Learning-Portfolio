import tensorflow as tf
import numpy as np
import cv2

# Load the object detection model
model = tf.saved_model.load('path/to/saved_model')

# Define the classes to detect
classes = ['class1', 'class2', 'class3']

# Open the default camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to RGB format
    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Expand the dimensions of the image to match the input shape of the model
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Perform object detection
    detections = model(image_np_expanded)

    # Extract the bounding boxes, classes, and scores from the detections
    boxes = detections['detection_boxes'][0].numpy()
    classes_idx = detections['detection_classes'][0].numpy().astype(int)
    scores = detections['detection_scores'][0].numpy()

    # Loop through the detected objects and draw bounding boxes on the frame
    for i in range(len(boxes)):
        if scores[i] > 0.5: # Only display objects with confidence score above 0.5
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * frame.shape[1])
            xmax = int(xmax * frame.shape[1])
            ymin = int(ymin * frame.shape[0])
            ymax = int(ymax * frame.shape[0])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, classes[classes_idx[i]-1], (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Wait for key press and exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
