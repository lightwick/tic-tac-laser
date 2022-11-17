import cv2
import os
import image_processing as ip
from time import sleep
import numpy as np
import classification
import time
import tictactoe_ai as ai

# TODO: should figure out a way for url to be fixed, or auto configured
# url = "http://192.168.0.6:4747/mjpegfeed"

def check(grid, frame):
    def fullBoard():
      for i in range(len(grid)):
        for j in range(len(grid[i])):
          if grid[i][j]==' ':
            return False
      return True
    
    isOver = False
    playerWon = ' '
    # Checking rows
    for i in range(3):
        if (grid[i][0] == grid[i][1] == grid[i][2] and grid[i][0] != ' '):
            playerWon = grid[i][0]
            isOver = True
            break
        if (grid[0][i] == grid[1][i] == grid[2][i] and grid[0][i] != ' '):
            playerWon = grid[0][i]
            isOver = True
            break
    if (not isOver and grid[0][0] == grid[1][1] == grid[2][2] and grid[0][0] != ' '):
        playerWon = grid[0][0]
        isOver = True
    if (not isOver and grid[0][2] == grid[1][1] == grid[2][0] and grid[0][2] != ' '):
        playerWon = grid[0][2]
        isOver = True
    if fullBoard():
        playerWon = ' '
        isOver = True
    if isOver:
        text=""
        if playerWon==' ':
            text = "Game Over: Tie"
        else:
            text = "Game Over: {} Wins!".format(playerWon)
        cv2.putText(frame, text, (40,300), cv2.FONT_HERSHEY_SIMPLEX, 4, (102,255,178), 6, cv2.LINE_AA)
    


def connect(img, points, idx_1, idx_2):
    cv2.line(img, points[idx_1], points[idx_2], (0, 0, 255), 3, cv2.LINE_AA)

def draw_grid(frame, points):
    for i in range(4):
        points[i*4:(i+1)*4] = points[points[i*4:(i+1)*4,0].argsort()+i*4]

    for i in range(4):
        for j in range(3):
            connect(frame, points, 4*i+j, 4*i+j+1)
            connect(frame, points, 4*j+i, 4*(j+1)+i)

crop_val = -2
# parameters: 4 points of a trapezoid
# returns: 2 points of a rectangle (upper_left, lower_right)
def get_rectangle(upper_left, upper_right, lower_left, lower_right):
    top_left_x = max(upper_left[0], lower_left[0])
    top_left_y = max(upper_left[1], upper_right[1])
    bottom_right_x = min(upper_right[0], lower_right[0])
    bottom_right_y = min(lower_right[1], lower_left[1])

    return ((top_left_x+crop_val, top_left_y+crop_val),(bottom_right_x-crop_val, bottom_right_y-crop_val))

def main():
    cap = cv2.VideoCapture(0)
    detect_line = False
    if not cap.isOpened():
        print("camera is not open")
        return
    
    # this is along the presupposition that the camera and the object on which the grid is drawn is stationary
    points = []
    copy = None
    
    # this serves no real purpose: just giving me enough time to start recording
    print("initialization ended, entering image processing")
    sleep(1)

    while len(points)!=16:
        _, img = cap.read()
        original = np.copy(img)
        img = ip.preprocessing(img)

        copy = np.copy(img)
        # ip.process_line(img, copy)
        points = ip.harris_corner_detection(img, copy)

        horizontal_concat = np.concatenate((original, copy), axis=1)
        cv2.imshow('hi',horizontal_concat)

        # TODO: changed parameters until grid is found
        key_input = cv2.waitKey(1)
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

    ################### GRID DETECTION END ####################

    recent_send = time.time()
    next_move = [-1,-1]

    prediction = [[' ' for i in range(3)] for j in range(3)] # prediction of markers of each cell
    action_value = ai.get_action_value(prediction)[0]
    
    while True:
        _, img = cap.read()
        original = np.copy(img)

        img = ip.preprocessing(img)

        copy = np.copy(img)
        draw_grid(copy, points)

        for i in range(9):
            #name = "grid_"+str(i+1)+".png"
            start_point = grid_rectangle[i][0]
            end_point = grid_rectangle[i][1]
            _img = img[start_point[1]:end_point[1],start_point[0]:end_point[0]]
            
            # draw recognition rectangle
            # cv2.rectangle(copy,  start_point, end_point, (200, 200, 0), 2)

            #cv2.imwrite(name, _img)
            # prediction of what's in the grid
            cell_prediction = classification.get_prediction(_img)

            prediction[i//3][i%3] = cell_prediction
            mid_point = (int((start_point[0]+end_point[0])/2), int((start_point[1]+end_point[1])/2))

            if cell_prediction=='O':
                cv2.circle(copy, mid_point, 20, (0, 0, 255), 4)
                if (i//3,i%3) == next_move:
                    next_move = (-1,-1)
            elif cell_prediction=='X':
                cv2.line(copy, (mid_point[0]-10,mid_point[1]-10), (mid_point[0]+10, mid_point[1]+10), (0, 0, 255), 4)
                cv2.line(copy, (mid_point[0]+10,mid_point[1]-10), (mid_point[0]-10, mid_point[1]+10), (0, 0, 255), 4)
            elif cell_prediction==' ':
                value = round(float(action_value[i]),2)
                cv2.putText(copy, str(value), (start_point[0]+10,end_point[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102,255,178), 2, cv2.LINE_AA)
                if (i//3,i%3) == next_move:
                    cv2.circle(copy, mid_point, 20, (255, 200, 0), 4)

        number_of_blank = sum(row.count(' ') for row in prediction)
        # time keeping to keep from overflowing buffer of arduino
        if number_of_blank%2==1 and (time.time()-recent_send)>=2:
            next_move = ai.get_action(prediction)
            start_point = grid_rectangle[next_move[0]][0]
            end_point = grid_rectangle[next_move[1]][1]
            mid_point = (int((start_point[0]+end_point[0])/2), int((start_point[1]+end_point[1])/2))
            action_value = ai.get_action_value(prediction)[0]
            recent_send = time.time()
        elif number_of_blank%2==0:
            next_move = (-1,-1)

        horizontal_concat = np.concatenate((original, copy), axis=1)
        check(prediction, horizontal_concat)

        cv2.imshow('hi',horizontal_concat)

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
