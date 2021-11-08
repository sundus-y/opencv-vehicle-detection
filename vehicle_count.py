# Import necessary packages

import cv2
import csv
import collections
import numpy as np
from tracker import *

# Initialize Tracker
tracker = EuclideanDistTracker()

# Initialize the videocapture object
cap = cv2.VideoCapture('a.mov')
input_size = 320
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps

# Detection confidence threshold
confThreshold =0.2
nmsThreshold= 0.2

font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Middle cross line position
middle_line = [(350,155),(550,5)]
up_line = [(330,155),(530,5)]
down_line = [(370,155),(570,5)]

# Store Coco Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')
print(classNames)
print(len(classNames))

# class index for our required detection classes
required_class_index = [2, 3, 5, 7]

detected_classNames = []

## Model Files
modelConfiguration = 'yolov3-320.cfg'
modelWeigheights = 'yolov3-320.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backend

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy
    
# List for store vehicle count information
ploting_frequency = 2
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]
up_per_frame = np.zeros(frame_count//ploting_frequency)
down_per_frame = np.zeros(frame_count//ploting_frequency)
# Function for count vehicle
def count_vehicle(box_id, img, frame_number=0):
    
    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center
    
    above_up_line = ((ix-up_line[0][0])*(up_line[1][1]-up_line[0][1]) - (iy-up_line[0][1])*(up_line[1][0]-up_line[0][0])) > 0
    below_up_line = not above_up_line
    above_middle_line = ((ix-middle_line[0][0])*(middle_line[1][1]-middle_line[0][1]) - (iy-middle_line[0][1])*(middle_line[1][0]-middle_line[0][0])) > 0
    below_middle_line = not above_middle_line
    above_down_line = ((ix-down_line[0][0])*(down_line[1][1]-down_line[0][1]) - (iy-down_line[0][1])*(down_line[1][0]-down_line[0][0])) > 0
    below_down_line = not above_down_line

    ploting_index = int(frame_number/ploting_frequency)

    # Find the current position of the vehicle
    if (above_middle_line and below_up_line):
    # if (iy > up_line_position) and (iy < middle_line_position):
        if id not in temp_up_list:
            temp_up_list.append(id)

    elif (above_down_line and below_middle_line):
    # elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)

    elif (above_up_line):            
    # elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index]+1
            up_per_frame[ploting_index] = up_per_frame[ploting_index] + 1

    elif (below_down_line):
    # elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1
            down_per_frame[ploting_index] = down_per_frame[ploting_index] + 1

    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    # print(up_list, down_list)

# Function for finding the detected objects from the network output
def postProcess(outputs,img,frame_number=0):
    global detected_classNames 
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            # print(x,y,w,h)

            color = [int(c) for c in colors[classIds[i]]]
            name = classNames[classIds[i]]
            detected_classNames.append(name)
            # Draw classname and confidence score 
            cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw bounding rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img, frame_number)

def realTime():
    count = 0
    output_video = cv2.VideoWriter('outpy.mov',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))
    
    while True:
        success, img = cap.read()
        if success:
            count += 1 # i.e. at 30 fps, this advances one second
            # cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        else:
            cap.release()
            break
        
        img = cv2.resize(img,(0,0),None,1,1)
        ih, iw, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        # Set the input of the network
        outputs = fetchNetworkOutput(blob)
    
        # Find the objects from the network output
        postProcess(outputs,img,count)

        # Draw the crossing lines
        drawCrossingLines(img)

        # Render counting texts in the frame
        renderCountingStats(img)

        # Plot counting stats
        plotCountingStats(img, count)

        # Show the frames
        # cv2.imshow('Output', img)

        output_video.write(img)
        print("Progress: " + str(int((count/frame_count)*100)) + "%")

        if cv2.waitKey(1) == ord('q'):
            break

    # Write the vehicle counting information in a file and save it

    with open("data.csv", 'w') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
        up_list.insert(0, "Up")
        down_list.insert(0, "Down")
        cwriter.writerow(up_list)
        cwriter.writerow(down_list)
    f1.close()
    # print("Data saved at 'data.csv'")

    with open("per_frame_data.csv", 'w') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Frame Rate:', fps])
        cwriter.writerow(['Frame Count:', frame_count])
        cwriter.writerow(['index', 'up', 'down'])
        for i in range(len(up_per_frame)):
            cwriter.writerow([i, up_per_frame[i], down_per_frame[i]])
    f1.close()

    # Finally realese the capture object and destroy all active windows
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

def drawCrossingLines(img):
    cv2.line(img, middle_line[0], middle_line[1], (255, 0, 255), 1)
    cv2.line(img, up_line[0], up_line[1], (0, 0, 255), 1)
    cv2.line(img, down_line[0], down_line[1], (0, 0, 255), 1)

def plotCountingStats(img, frame_number):
    origin = (20, frame_height-20)
    cv2.line(img, (0,frame_height-20), (frame_width,frame_height-20), (255, 0, 255), 1)
    cv2.line(img, (20,0), (20,frame_height), (255, 0, 255), 1)
    single_rectangle_width = int(frame_width/len(up_per_frame))
    upsum = 0
    if isPlotableFrame(frame_number):
        for i in range(int(frame_number/ploting_frequency)):
            upsum += up_per_frame[i]
            bp = (origin[0] + (i * single_rectangle_width), origin[1])
            bt = (origin[0] + (i * single_rectangle_width), origin[1] - (int(upsum) * 5))
            cv2.rectangle(img, bp , bt, (0, 255, 0), 1);

def isPlotableFrame(frame_number):
    return frame_number%ploting_frequency == 0

def renderCountingStats(img):
    cv2.putText(img, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Car:        "+str(up_list[0])+"     "+ str(down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Motorbike:  "+str(up_list[1])+"     "+ str(down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Bus:        "+str(up_list[2])+"     "+ str(down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Truck:      "+str(up_list[3])+"     "+ str(down_list[3]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

def fetchNetworkOutput(blob):
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i- 1]) for i in net.getUnconnectedOutLayers()]
        # Feed data to the network
    outputs = net.forward(outputNames)
    return outputs

image_file = 'frame100.jpg'
def from_static_image(image):
    img = cv2.imread(image)

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

    # Set the input of the network
    net.setInput(blob)
    layersNames = net.getLayerNames()
    l2 = net.getUnconnectedOutLayers()
    outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    # Feed data to the network
    outputs = net.forward(outputNames)

    # Find the objects from the network output
    postProcess(outputs,img)

    # count the frequency of detected classes
    frequency = collections.Counter(detected_classNames)
    print(frequency)
    # Draw counting texts in the frame
    cv2.putText(img, "Car:        "+str(frequency['car']), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Motorbike:  "+str(frequency['motorbike']), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Bus:        "+str(frequency['bus']), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Truck:      "+str(frequency['truck']), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)


    cv2.imshow("image", img)

    cv2.waitKey(0)

    # save the data to a csv file
    with open("static-data.csv", 'a') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow([image, frequency['car'], frequency['motorbike'], frequency['bus'], frequency['truck']])
    f1.close()

if __name__ == '__main__':
    realTime()
    # from_static_image(image_file)