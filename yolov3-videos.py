import numpy as np
import cv2
import os
import time


# define the minimum confidence (to filter weak detections), 
# Non-Maximum Suppression (NMS) threshold, and the green color
confidence_thresh = 0.5
NMS_thresh = 0.3
green = (0, 255, 0)

# Initialize the video capture object
video_cap = cv2.VideoCapture("examples/videos/1.mp4")

# load the class labels the model was trained on
classes_path = "yolov3-config/coco.names"
with open(classes_path, "r") as f:
    classes = f.read().strip().split("\n")
    
# load the configuration and weights from disk
yolo_config = "yolov3-config/yolov3.cfg"
yolo_weights = "yolov3-config/yolov3.weights"

# load the pre-trained YOLOv3 network
net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the name of all the layers in the network
layer_names = net.getLayerNames()
# Get the names of the output layers
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

while True:
    # start time to compute the fps
    start = time.time()
    # read the video frame
    success, frame = video_cap.read()
  
    # if there are no more frames to show, break the loop
    if not success:
        break

    # # get the frame dimensions
    h = frame.shape[0]
    w = frame.shape[1]

    # create a blob from the frame
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255, (416, 416), swapRB=True, crop=False)
    # pass the blog through the network and get the output predictions
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    # create empty lists for storing the bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # loop over the output predictions
    for output in outputs:
        # loop over the detections
        for detection in output:
            # get the class ID and confidence of the dected object
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence  = scores[class_id]

            # filter out weak detections by keeping only those with a confidence 
            # above the minimum confidence threshold (0.5 in this case).
            if confidence > confidence_thresh:
                # perform element-wise multiplication to get
                # the coordinates of the bounding box
                box = [int(a * b) for a, b in zip(detection[0:4], [w, h, w, h])]
                center_x, center_y, width, height = box

                # get the top-left corner of the bounding box
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                # append the bounding box, confidence, and class ID to their respective lists
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])

    # apply non-maximum suppression to remove weak bounding boxes that overlap with others.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, NMS_thresh)
    indices = indices.flatten()

    for i in indices:
        (x, y, w, h) = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        cv2.rectangle(frame, (x, y), (x + w, y + h), green, 2)
        text = f"{classes[class_ids[i]]}: {confidences[i] * 100:.2f}%"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
        
    # end time to compute the fps
    end = time.time()
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start):.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
    
    # display the frame
    cv2.imshow("Frame", frame)
    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(30) == ord("q"): 
        break
        
# release the video capture object
video_cap.release()
cv2.destroyAllWindows()
