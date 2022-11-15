import cv2
import os
import image_processing as ip
from time import sleep
import numpy as np
import detection

# TODO: should figure out a way for url to be fixed, or auto configured
# url = "http://192.168.0.6:4747/mjpegfeed"

def connect(img, points, idx_1, idx_2):
    cv2.line(img, points[idx_1], points[idx_2], (0, 0, 255), 3, cv2.LINE_AA)

def draw_grid(frame, points):
    for i in range(4):
        points[i*4:(i+1)*4] = points[points[i*4:(i+1)*4,0].argsort()+i*4]

    for i in range(4):
        for j in range(3):
            connect(frame, points, 4*i+j, 4*i+j+1)
            connect(frame, points, 4*j+i, 4*(j+1)+i)

# parameters: 4 points of a trapezoid
# returns: 2 points of a rectangle (upper_left, lower_right)
def get_rectangle(upper_left, upper_right, lower_left, lower_right):
    top_left_x = max(upper_left[0], lower_left[0])
    top_left_y = max(upper_left[1], upper_right[1])
    bottom_right_x = min(upper_right[0], lower_right[0])
    bottom_right_y = min(lower_right[1], lower_left[1])

    return ((top_left_x, top_left_y),(bottom_right_x, bottom_right_y))

def main():
    cap = cv2.VideoCapture(1)
    detect_line = False
    if not cap.isOpened():
        print("camera is not open")
        return
    
    # this is along the presupposition that the camera and the object on which the grid is drawn is stationary
    points = []
    copy = None
    while len(points)!=16:
        _, img = cap.read()
        img = ip.preprocessing(img)

        copy = np.copy(img)
        # ip.process_line(img, copy)
        points = ip.harris_corner_detection(img, copy)
        cv2.imshow('hi', copy)
        # TODO: changed parameters until grid is found
        key_input = cv2.waitKey(1000)
        if key_input == ord('q'):
            return
    print("found 16 points of grid")

    '''
    while True:
            # The shape of frame is (qheight: 480, width: 640, 3)
            frameExists, frame = cap.read()
            
            if not frameExists:
                print("no more incoming data.. aborting..")
                break
    '''
    draw_grid(copy, points)
    sleep(2)

    # 9 squares, 2 coordinates, x and y for each coordinate
    grid_rectangle = np.zeros([9,2,2]).astype(int)

    for i in range(3):
        for j in range(3):
            grid_rectangle[i*3+j] = get_rectangle(points[i*4+j], points[i*4+j+1], points[(i+1)*4+j], points[(i+1)*4+j+1])

    while True:
        _, img = cap.read()
        img = ip.preprocessing(img)

        copy = np.copy(img)
        draw_grid(copy, points)

        for i in range(9):
            #name = "grid_"+str(i+1)+".png"
            start_point = grid_rectangle[i][0]
            end_point = grid_rectangle[i][1]
            _img = img[start_point[1]:end_point[1],start_point[0]:end_point[0]]
            #cv2.imwrite(name, _img)
            # prediction of what's in the grid
            prediction = detection.get_prediction(_img)
            mid_point = (int((start_point[0]+end_point[0])/2), int((start_point[1]+end_point[1])/2))

            if prediction=='O':
                cv2.circle(copy, mid_point, 25, (0, 0, 255), 4)
            elif prediction=='X':
                cv2.line(copy, (mid_point[0]-10,mid_point[1]-10), (mid_point[0]+10, mid_point[1]+10), (0, 0, 255), 4)
                cv2.line(copy, (mid_point[0]+10,mid_point[1]-10), (mid_point[0]-10, mid_point[1]+10), (0, 0, 255), 4)

        cv2.imshow('hi',copy)

        key_input = cv2.waitKey(1)
        if key_input == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

main()
'''
if __name__=="__main__":
    main()
'''
