import cv2
import numpy as np
import time
import subprocess
import Jetson.GPIO as GPIO

# For initializing PiCamera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Setting up GPIO for servo control and Lumos/Nox
SERVO_PIN = 33  # Change this to the GPIO pin connected to the servo
GPIO_PIN_LUMOS = 32  # GPIO pin for Lumos
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(GPIO_PIN_LUMOS, GPIO.OUT)

# Define parameters for the required blob
params = cv2.SimpleBlobDetector_Params()

# setting the thresholds
params.minThreshold = 150
params.maxThreshold = 250

# filter by color
params.filterByColor = 1
params.blobColor = 255

# filter by circularity
params.filterByCircularity = 1
params.minCircularity = 0.68

# filter by area
params.filterByArea = 1
params.minArea = 30
# params.maxArea = 1500

# creating object for SimpleBlobDetector
detector = cv2.SimpleBlobDetector_create(params)

flag = 0
points = []
lower_blue = np.array([255, 255, 0])
upper_blue = np.array([255, 255, 0])

# Function for Pre-processing
def last_frame(img):
    cv2.imwrite("lastframe1.jpg", img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite("lastframe2.jpg", img)
    retval, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    cv2.imwrite("lastframe3.jpg", img)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    cv2.imwrite("lastframe4.jpg", img)
    img = cv2.dilate(img, (3, 3))
    cv2.imwrite("lastframe.jpg", img)
    output = subprocess.check_output(['python3', 'predict.py'])
    print(output.decode('utf-8')[1])
    print("is the prediction")
    if output.decode('utf-8')[1] == '2':
        print("Alohamora!!")
        GPIO.output(SERVO_PIN, GPIO.HIGH)  # Set GPIO 32 to LOW for Nox
        print("Opened!!")
        time.sleep(1.5)
        GPIO.output(SERVO_PIN, GPIO.LOW)
        print("Closed!!")
    elif output.decode('utf-8')[1] == '3':
        print("Closed!!")
        GPIO.output(SERVO_PIN, GPIO.LOW)
        time.sleep(1.5)
    elif output.decode('utf-8')[1] == '1':
        print("Lumos")
        GPIO.output(GPIO_PIN_LUMOS, GPIO.HIGH)  # Set GPIO 32 to HIGH for Lumos
        time.sleep(1.5)
    elif output.decode('utf-8')[1] == '0':
        print("Nox")
        GPIO.output(GPIO_PIN_LUMOS, GPIO.LOW)  # Set GPIO 32 to LOW for Nox
        time.sleep(1.5)

time.sleep(0.1)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Flip the frame 180 degrees about the y-axis
    frame = cv2.flip(frame, 1)

    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecting keypoints
    keypoints = detector.detect(frame_gray)
    frame_with_keypoints = cv2.drawKeypoints(frame_gray, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # starting and ending circle
    frame_with_keypoints = cv2.circle(frame_with_keypoints, (140, 70), 6, (0, 255, 0), 2)
    frame_with_keypoints = cv2.circle(frame_with_keypoints, (190, 140), 6, (0, 0, 255), 2)

    points_array = cv2.KeyPoint_convert(keypoints)

    if flag == 1 and len(points_array) > 0:
        # Get coordinates of the center of blob from keypoints and append them in points list
        points.append(points_array[0])

        # Draw the path by drawing lines between 2 consecutive points in points list
        for i in range(1, len(points)):
            pt1 = tuple(map(int, points[i-1]))
            pt2 = tuple(map(int, points[i]))
            print("Drawing line from", pt1, "to", pt2)
            cv2.line(frame_with_keypoints, pt1, pt2, (255, 255, 0), 3)

    if len(points_array) != 0:
        if flag == 1:
            if int(points_array[0][0]) in range(185, 195) and int(points_array[0][1]) in range(135, 145):
                print("Tracing Done!!")
                # Set the color range for the frame
                frame_with_keypoints = cv2.inRange(frame_with_keypoints, lower_blue, upper_blue)
                last_frame(frame_with_keypoints)
                # Reset flag and points to start fresh
                flag = 0
                points = []
                continue  # Continue to the next iteration to clear keypoints and start from beginning

        if flag == 0:
            if int(points_array[0][0]) in range(135, 145) and int(points_array[0][1]) in range(65, 75):
                time.sleep(0.5)
                print("Start Tracing!!")
                flag = 1

    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("video", 640, 480)
    cv2.imshow("video", frame_with_keypoints)
    # cv2.imshow("video 2", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
