import numpy as np
import cv2 as cv
import time
import sys

cap = cv.VideoCapture(cv.samples.findFile(sys.argv[1]))
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1]= 255

start = time.time()
while True:
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    print(f'{time.time() - start}s1')
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    prvs = next
    print(f'{time.time() - start}s2')
    start = time.time()
