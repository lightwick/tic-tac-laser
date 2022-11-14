import cv2
import os
import image_processing
from time import sleep
import numpy as np

# TODO: should figure out a way for url to be fixed, or auto configured
# url = "http://192.168.0.6:4747/mjpegfeed"

def main():
    cap = cv2.VideoCapture(1)
    detect_line = False
    if not cap.isOpened():
        print("camera is not open")
        return
        
    while True:
            # The shape of frame is (qheight: 480, width: 640, 3)
            frameExists, frame = cap.read()

            if not frameExists:
                print("no more incoming data.. aborting..")
                break

            frame = image_processing.preprocessing(frame)
            #frame[blackAndWhiteImage==0] = (0,0,255)

            copy = np.copy(frame)

            image_processing.process_line(copy, frame)
            image_processing.harris_corner_detection(copy, frame)

            #cv2.imshow('Hello', frame)
            key_input = cv2.waitKey(1)
            cv2.imshow('Hello', frame)

            if key_input in [ord('q'), ord('d')]:
                break
            elif key_input == ord(' '):
                detect_line = not detect_line
                print("line detection " + ("on" if detect_line else "off"))

    cap.release()
    cv2.destroyAllWindows()

main()
'''
if __name__=="__main__":
    main()
'''
