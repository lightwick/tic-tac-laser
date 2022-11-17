import serial
from time import sleep

#ser = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=0.001)
ser = serial.Serial(port='COM8', baudrate=9600, timeout=1)

# the state should be either "on" or "off"
def control_laser(state):
    ser.write(bytes(state, 'utf-8'))

sleep(2)
ser.write(bytes("on", 'utf-8'))

print("serial port opened for " + ser.name)

def send_coord(i,j):
    x = 90
    y = 90
    if i==0:
        y=42
    elif i==1:
        y=53
    else:
        y=62

    if j==0:
        x=116
    elif j==1:
        x=108
    elif j==2:
        x=96
    print(x,y)
    data = "{0} {1}".format(x,y)
    ser.write(bytes(data, 'utf-8'))
    print("send data: {}".format(data))
    while True:
        print(ser.readline())