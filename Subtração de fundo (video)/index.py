import cv2
import numpy as np

VIDEO = 'surveillance.mpg'

vrObj = cv2.VideoCapture(VIDEO)
nFrames = int(vrObj.get(cv2.CAP_PROP_FRAME_COUNT))

frame_width = int(vrObj.get(3))
frame_height = int(vrObj.get(4))
fps = vrObj.get(5)

vwObj = cv2.VideoWriter('Background_Subtraction.avi', cv2.VideoWriter_fourcc(*"MJPG"), 30, (frame_width, frame_height), 0)

alpha = 0.95
theta = 0.1
_, background = vrObj.read()
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
while True:
    ret, frame = vrObj.read()
    if not ret:
        break

    currImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    background = alpha * background + (1 - alpha) * currImg
    diffImg = np.abs(currImg - background)
    _, threshImg = cv2.threshold(diffImg, theta, 1, cv2.THRESH_BINARY)
    threshImg = threshImg.astype(np.uint8)

    cv2.imshow('New frame', currImg)
    cv2.imshow('Background frame', background)
    cv2.imshow('Difference image', diffImg)
    cv2.imshow('Thresholded difference image', threshImg.astype(np.uint8))

    vwObj.write((threshImg * 255).astype(np.uint8))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vwObj.release()
vrObj.release()
cv2.destroyAllWindows()

cv2.imwrite('Background_Subtraction_curr.png', (currImg * 255).astype(np.uint8))
cv2.imwrite('Background_Subtraction_background.png', (background * 255).astype(np.uint8))
cv2.imwrite('Background_Subtraction_thresh.png', (threshImg * 255).astype(np.uint8))
