import cv2
import numpy as np

def process_line(img):
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    minLineLength=100
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)

    # To ensure that the returned lines are not None
    # If lines are None, we can't use .any()
    lines = np.array(lines)
    if not lines.any():
        return
    a,b,c = lines.shape
    for i in range(a):
        cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
        #cv2.imwrite('houghlines5.jpg',gray)