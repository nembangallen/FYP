from flask import Flask
import pandas as pd
import numpy as np
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras.models import load_model

app = Flask(__name__)

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

def pre_processing(df1):
    '''Data Processing modeule
    1. take only closing value from dataframe
    2. scale value from 0 to 1 taking minimum and maxixum value from closing value
    3. split to train and test 70-30 %
    4. return train and test data
    '''
    df1=df1.reset_index()['c']   #read only closing value
    ### LSTM are sensitive to the scale of the data. so we apply MinMax scaler 
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
    ##splitting dataset into train and test split
    training_size=int(len(df1)*0.75)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    print('test-data: ', test_data)
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)
    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    return X_train,y_train, X_test,ytest,test_data,scaler



#forcasting
def forcase_data(model,test_data,n_steps=100):
  x_input=test_data[len(test_data)-100:].reshape(1,-1)
  temp_input=list(x_input)
  temp_input=temp_input[0].tolist()
  # demonstrate prediction for next 10 days
  lst_output=[]
  i=0
  while(i<7):
      if(len(temp_input)>100):
          #print(temp_input)
          x_input=np.array(temp_input[1:])
          print("{} day input {}".format(i,x_input))
          x_input=x_input.reshape(1,-1)
          x_input = x_input.reshape((1, n_steps, 1))
          #print(x_input)
          yhat = model.predict(x_input, verbose=0)
          print("{} day output {}".format(i,yhat))
          temp_input.extend(yhat[0].tolist())
          temp_input=temp_input[1:]
          #print(temp_input)
          lst_output.extend(yhat.tolist())
          i=i+1
      else:
          x_input = x_input.reshape((1, n_steps,1))
          yhat = model.predict(x_input, verbose=0)
        #   print(yhat[0])
          temp_input.extend(yhat[0].tolist())
        #   print(len(temp_input))
          lst_output.extend(yhat.tolist())
          i=i+1
  return lst_output

@app.route("/")
def main_fun():
    return {"status": "Running server"}
      

@app.route("/<company_name>", methods=["GET"])
def hello(company_name):
    MODEL_PATH = 'models/'+company_name+'.h5'
    model = load_model(MODEL_PATH)
    df=pd.read_csv('data/' + company_name + '.csv')
    X_train,y_train, X_test,ytest,test_data,scaler = pre_processing(df1=df)
    lst_output = forcase_data(model=model,test_data=test_data,n_steps=100)

    lst_output=scaler.inverse_transform(lst_output).flatten().tolist()
    test_data = scaler.inverse_transform(test_data).flatten().tolist()
    
    return {"test_data": test_data,"forecasted_stocks": lst_output}

if __name__ == "__main__":
  app.run()