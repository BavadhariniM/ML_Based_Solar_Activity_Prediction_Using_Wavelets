#importing the necessary libraries

import os
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model
import pywt
import tensorflow

from flask import Flask, render_template, request

model=load_model('lstm_sunspots.h5')
modelc=load_model('lstm_with_wav_c.h5')
modelx=load_model('lstm_with_wav_x.h5')
modelm=load_model('lstm_with_wav_m.h5')

def get_model_output(model,input):
    return model.predict(input)



app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    date = request.files['date']
    
    prediction = output_prediction(date)
   
    return render_template('result.html', prediction=prediction)

def output_prediction(date):
    date=date.split()
    d=int(date[0])
    m=int(date[1])
    y=int(date[2])
    sunspots=get_model_output(model,np.array([d,m,y]))
    
    resc=resm=resx=pywt.wavedec([sunspots], 'db2', level=11,mode='smooth')
    
    for i in resc:
        for j in i:
            j=get_model_output(modelc,j)
    
    for i in resm:
        for j in i:
            j=get_model_output(modelm,j)
    
    for i in resx:
        for j in i:
            j=get_model_output(modelx,j)
    
    c=pywt.waverec(resc,'db2',mode="antireflect")
    m=pywt.waverec(resm,'db2',mode="antireflect")
    x=pywt.waverec(resx,'db2',mode="antireflect")
    
    return "No. of C Class Flares : "+c+"\n"+"No. of M Class Flares : "+m+"\n"+"No. of X Class Flares : "+x


if __name__ == '_main_':
    app.run()