from keras.models import load_model
import cv2
import numpy as np

model = load_model("./model.h5")
grid=[[-1 for i in range(3)] for j in range(3)]

def preprocess_input(img):
    """Preprocess image to match model's input shape for shape detection"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (32, 32))
    # Expand for channel_last and batch size, respectively
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32) / 255

mapping = {0:' ', 1:'X', 2:'O'}

def get_prediction(img):
    img = preprocess_input(img)
    idx = np.argmax(model(img))
    return mapping[idx]
    
'''
for i in range(9):
    name = "./grid_"+str(i+1)+".png"
    img = cv2.imread(name)
    img = preprocess_input(img)
''' 
'''
while True:
    cv2.imshow('hi', img)
    key_input = cv2.waitKey(1)
    if key_input == ord('q'):
        break
'''