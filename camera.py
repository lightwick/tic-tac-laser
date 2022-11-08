import cv2

url = "http://172.30.10.175:4747/mjpegfeed"

def main():
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("camera is not open")
        return
        
    while True:
            frameExists, frame = cap.read()
            #frame = cv2.resize(frame,(640,480))
            if not frameExists:
                print("no more incoming data.. aborting..")
                break
            cv2.imshow('Hello', frame)
            if cv2.waitKey(1) == ord('d'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
