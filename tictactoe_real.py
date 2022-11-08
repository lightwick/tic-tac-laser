import cv2
import os
from image_processing import process_line

url = "http://172.30.10.175:4747/mjpegfeed"

def main():
    cap = cv2.VideoCapture(url)
    detect_line = False
    if not cap.isOpened():
        print("camera is not open")
        return
        
    while True:
            # The shape of frame is (height: 480, width: 640, 3)
            frameExists, frame = cap.read()
            frame = frame[50:]
            if not frameExists:
                print("no more incoming data.. aborting..")
                break
            if detect_line == True:
                process_line(frame)
            cv2.imshow('Hello', frame)
            key_input = cv2.waitKey(1)
            if key_input in [ord('q'), ord('d')]:
                break
            elif key_input == ord(' '):
                detect_line = not detect_line
                print("line detection " + ("on" if detect_line else "off"))
            
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
