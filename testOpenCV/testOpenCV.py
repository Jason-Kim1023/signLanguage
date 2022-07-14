from __future__ import absolute_import, division, print_function
# TensorFlow and tf.keras
import os
import tensorflow as tf
from tensorflow import keras
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Helper libraries
import numpy as np

import cv2
import csv
import matplotlib.pyplot as plt
    

def mouseRGB(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition

        rows,cols = grayFrame.shape
        print(rows)
        print(cols)
        k = []
        for i in range(rows):
            for j in range(cols):
                k.append(grayFrame[i,j])
        print(len(k))
        
        # todo: frame is an matrix. you can look at it with debug breaks. you should get the gray scale value for Each pixel,
        # and once done, you can send it to the training.csv. 
    #with open('training.csv', 'a', newline='') as myFile:
    #    writer = csv.writer(myFile, delimiter=',')
    #    rowToWrite = np.array(['A'])
    #    np.append(rowToWrite,grayFrame.flatten())
    #    writer.writerow(rowToWrite)  

    
def cameraCapture(model):
    while(True):
      
        ret, frame = capture.read()
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        grayFrame = cv2.resize(grayFrame, [128,128], interpolation=cv2.INTER_AREA)
      
        cv2.imshow('mouseRGB', grayFrame)
    
        if cv2.waitKey(1) == 27:
            break
        
        print(model.predict(grayFrame.flatten()))
    capture.release()

def modelTraining():
    model = keras.Sequential([
        keras.layers.Dense(128*128, activation=tf.nn.relu, input_shape = (None, 1)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(26, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              run_eagerly = True)
    
    #Get our training image data
    trainingImageMatrix = []
    trainingLabelsArray = np.empty(0)
    temp = 0
    with open('training.csv', 'r', newline='') as myFile:
        reader = csv.reader(myFile, delimiter=',')
        for row in reader:
            temp+=1
            imageDataToAdd = np.zeros(128*128)
            for i in range (128*128):
                imageDataToAdd[i] = int(row[i])
            trainingLabelsArray = np.append(trainingLabelsArray,) #need to get the first index of every row
            trainingImageMatrix.append(imageDataToAdd)
            print("reading row: "+str(temp))
            if temp >=200:
                break
        trainingImageMatrix = np.array(trainingImageMatrix)
    model.fit(trainingImageMatrix, trainingLabelsArray, epochs = 26)
    return model





model = modelTraining()
cv2.namedWindow('mouseRGB')
capture = cv2.VideoCapture(0)
#cv2.setMouseCallback('mouseRGB',mouseRGB)

cameraCapture(model)
cv2.destroyAllWindows()
