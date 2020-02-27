import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from keras.models import Sequential
from keras.layers import Dense,Dropout,MaxPool1D,Convolution1D
from keras.wrappers.scikit_learn import KerasClassifier

# load data
data=pd.read_csv("data/train.csv")
length=len(data)

# visualization
visualization=0
if visualization:
    figure,axes=plt.subplots(nrows=2,ncols=1,figsize=(20,10))
    data['signal'].plot(ax=axes[0])
    data['open_channels'].plot(ax=axes[1])
    plt.show()

# data process

train_data=data[:int(0.8*length)][['signal','open_channels']].values.astype(float)
val_data=data[int(0.8*length):][['signal','open_channels']].values.astype(float)

def baseline_model():
    model=Sequential()
    model.add(Convolution1D(filter=1,kernel_size=10,strides=1,padding='causal',activation='relu'))

