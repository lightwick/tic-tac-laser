import cv2
import numpy as np
import image_processing as ip
import imutils

val = 80
toggle = False

img = cv2.imread("./img/grid.jpg")
##################### IMAGE SPECIFIC CODE #####################
img = img[:800,:]
#img = imutils.rotate(img, 3)
##################### IMAGE SPECIFIC CODE #####################

img = ip.preprocessing(img)
print(img.shape)
res = np.zeros([15,2])

def connect(img, points, idx_1, idx_2):
    cv2.line(img, points[idx_1], points[idx_2], (0, 0, 255), 3, cv2.LINE_AA)

# this is along the presupposition that the camera and the object on which the grid is drawn is stationary
points = []
while len(points)!=16:
    copy = np.copy(img)
    # ip.process_line(img, copy)
    points = ip.harris_corner_detection(img, copy)
    cv2.imshow('hi', copy)
    # TODO: changed parameters until grid is found

print("found 16 points of grid")

# drawing grid
points = points[points[:,1].argsort()]

for i in range(4):
        points[i*4:(i+1)*4] = points[points[i*4:(i+1)*4,0].argsort()+i*4]

for i in range(4):
    for j in range(3):
        connect(copy, points, 4*i+j, 4*i+j+1)
        connect(copy, points, 4*j+i, 4*(j+1)+i)

# grid draw end

# parameters: 4 points of a trapezoid
# returns: 2 points of a rectangle (upper_left, lower_right)
def get_rectangle(upper_left, upper_right, lower_left, lower_right):
    top_left_x = max(upper_left[0], lower_left[0])
    top_left_y = max(upper_left[1], upper_right[1])
    bottom_right_x = min(upper_right[0], lower_right[0])
    bottom_right_y = min(lower_right[1], lower_left[1])

    return ((top_left_x, top_left_y),(bottom_right_x, bottom_right_y))

# 9 squares, 2 coordinates, x and y for each coordinate
grid_rectangle = np.zeros([9,2,2]).astype(int)

for i in range(3):
    for j in range(3):
        grid_rectangle[i*3+j] = get_rectangle(points[i*4+j], points[i*4+j+1], points[(i+1)*4+j], points[(i+1)*4+j+1])

for point1, point2 in grid_rectangle:
    cv2.rectangle(copy, point1, point2, (200, 200, 0), 2)

cv2.imwrite('img/rectangle_in_grid.png', copy)

while True:
    cv2.imshow('hi', copy)
    #cv2.imshow('to hell',img)

    key_input = cv2.waitKey(1)
    if key_input == ord('q'):
        break

cv2.destroyAllWindows()
'''
while True:
    img = cv2.cv2.imread("C:/projects/tictactoe_real/tic-tac-laser/grid.jpg")
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
'''
