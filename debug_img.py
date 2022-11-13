import cv2
import numpy as np

val = 80
toggle = False

while True:
    img = cv2.imread("C:/projects/tictactoe_real/tic-tac-laser/gree.jpg")
    if toggle:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (5,5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
        edges = cv2.Canny(grey,50,150,apertureSize = 3)
        # initially minLineLength 100
        minLineLength=10
        lines = cv2.HoughLinesP(image=edges,rho=1.2,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=val)
        
        lines = np.array(lines)
        if not lines.any():
            continue
        a,b,c = lines.shape
        for i in range(a):
            cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
    else:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # find Harris corners
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        dst = cv2.dilate(dst,None)
        ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        dst = np.uint8(dst)
        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
        # Now draw them
        res = np.hstack((centroids,corners))
        res = np.int0(res)

        for i in range(len(res)):
            cv2.circle(img,(res[i,0],res[i,1]), 4, (255,0,0), 2)
        img[res[:,1],res[:,0]]=[0,0,255]
        img[res[:,3],res[:,2]] = [0,255,0]

    cv2.imshow('hello', img)
    key_input = cv2.waitKey(1)
    if key_input in [ord('q'), ord('d')]:
        break
    elif key_input == ord('w'):
        val+=10
        print(val)
    elif key_input == ord('e'):
        val-=10
        print(val)
    elif key_input == ord(' '):
        toggle = not toggle