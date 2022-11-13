import cv2
import numpy as np


def drawLines(img, lines):
    """
    Draw lines on an image
    """
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 1)

def get_strong_lines(edged):
    strong_lines = np.zeros([4,1,2])

    minLineLength = 2
    maxLineGap = 10

    lines = cv2.HoughLines(edged,1,np.pi/180,10, minLineLength, maxLineGap)

    n2 = 0
    for n1 in range(0,len(lines)):
        for rho,theta in lines[n1]:
            if n1 == 0:
                strong_lines[n2] = lines[n1]
                n2 = n2 + 1
            else:
                if rho < 0:
                    rho*=-1
                    theta-=np.pi
                closeness_rho = np.isclose(rho,strong_lines[0:n2,0,0],atol = 10)
                closeness_theta = np.isclose(theta,strong_lines[0:n2,0,1],atol = np.pi/36)
                closeness = np.all([closeness_rho,closeness_theta],axis=0)
                if not any(closeness) and n2 < 4:
                    strong_lines[n2] = lines[n1]
                    n2 = n2 + 1
    return strong_lines

def process_line(img, frame):
    
    #cv2.imwrite('grid.jpg', img)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(grey, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    edges = cv2.Canny(thresh,50,150,apertureSize = 3)

    #lines = cv2.HoughLines(image=edges, rho = 1.1,theta = 1*np.pi/180, threshold = 135,minLineLength = 100,maxLineGap = 50)
    lines = cv2.HoughLinesP(image=edges, rho = 1,theta = 1*np.pi/180, threshold = 160,minLineLength = 100,maxLineGap = 50)
    #print("p shape: ", lines.shape)
    # To ensure that the returned lines are not None
    # If lines are None, we can't use .any()
    #lines = get_strong_lines(edges)

    lines = np.array(lines)
    if not lines.any():
        return

    a,b,c = lines.shape
    
    for i in range(a):
        cv2.line(frame, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
    
    #drawLines(frame, lines)

    key_input = cv2.waitKey(1)
    if key_input == ord('w'):
        val+=0.1
        print(val)
    elif key_input == ord('e'):
        val-=0.1
        print(val)

def harris_corner_detection(img, frame):
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # find Harris corners
        gray = np.float32(gray)
        
        dst = cv2.cornerHarris(gray,2,3,0.04)
        dst = cv2.dilate(dst,None)

        # This value vary depending on the image and how many corners you want to detect
        # Try changing this free parameter, 0.1, to be larger or smaller and see what happens
        thresh = 0.4*dst.max()

        # Iterate through all the corners and draw them on the image (if they pass the threshold)
        for j in range(0, dst.shape[0]):
            for i in range(0, dst.shape[1]):
                if(dst[j,i] > thresh):
                    # image, center pt, radius, color, thickness
                    cv2.circle(frame, (i, j), 1, (0,255,0), 1)

        '''
        for i in range(len(res)):
            cv2.circle(img,(res[i,0],res[i,1]), 4, (255,0,0), 2)
        
        img[res[:,1],res[:,0]]=[0,0,255]
        img[res[:,3],res[:,2]] = [0,255,0]
        '''

def getBlackOnly(img):
    # Convert to float and divide by 255:
    imgFloat = img.astype(np.float) / 255.

    # Calculate channel K:
    kChannel = 1 - np.max(imgFloat, axis=2)

    # Convert back to uint 8:
    kChannel = (255 * kChannel).astype(np.uint8)
    # Threshold image:
    binaryThresh = 140
    _, binaryImage = cv2.threshold(kChannel, binaryThresh, 255, cv2.THRESH_BINARY)
    return binaryImage