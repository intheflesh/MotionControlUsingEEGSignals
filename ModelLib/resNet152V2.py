from tensorflow.keras.backend import l2_normalize
from tensorflow.keras import activations
from tensorflow.keras import Input
from tensorflow.keras.layers import Softmax,LSTM,GRU,Conv1D,Conv2D, Conv1D, BatchNormalization, Activation, Add, AveragePooling1D, ZeroPadding2D,ZeroPadding1D, MaxPooling2D,MaxPooling1D, AveragePooling2D, Flatten ,Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Reshape
from tensorflow.keras.applications import ResNet152V2
import numpy as np

def GRUAndConv(dim0,dim1,dim2):
  rate = 0.5
  numOfGRUUnits = 1024
  input_im = Input(shape=(dim0,dim1, dim2))
  x = ResNet152V2(input_shape=(dim0,dim1,dim2),include_top=False, weights=None, classes=6)(input_im)
  x = Flatten()(x)
  x = Dense(6,activation='sigmoid')(x)
  model = Model(inputs=input_im, outputs=x, name='resnet152V2')

  return model