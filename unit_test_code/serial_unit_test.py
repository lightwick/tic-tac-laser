import pyautogui
import serial
from time import sleep
from pynput import mouse

def on_click(x, y, button, pressed):
    if button == mouse.Button.left:
        if pressed:
            ser.write(b't')
        return True
    else:
        print("disabled key click")
        return False

listener = mouse.Listener(on_click=on_click)
listener.start()


#ser = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=0.001)
ser = serial.Serial(port='COM8', baudrate=9600, timeout=0.001)

print("serial port opened for " + ser.name)

width, height = pyautogui.size()
x,y = pyautogui.position()
x = int(x/width*180)
y = int(y/height*180)

while True:
    sleep(0.05)
    _x,_y = pyautogui.position()
    _x = 180-int(_x/width*180)
    _y = int(_y/height*180)
    if x!=_x or y!=_y:
        x = _x
        y = _y
        print(x,y)
        data = "{0} {1}".format(x,y)
        ser.write(bytes(data, 'utf-8'))
