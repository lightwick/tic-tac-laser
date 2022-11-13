import cv2
import numpy as np

while True:
    img = cv2.imread("C:/projects/tictactoe_real/tic-tac-laser/gree.jpg")
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grey,50,150,apertureSize = 3)
    cv2.imshow('hello', edges)
    if cv2.waitKey(1)==ord('q'):
        break