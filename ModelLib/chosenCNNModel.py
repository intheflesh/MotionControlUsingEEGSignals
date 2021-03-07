from tensorflow.keras.backend import l2_normalize
from tensorflow.keras import activations
from tensorflow.keras import Input
from tensorflow.keras.layers import Softmax,LSTM,GRU,Conv1D,Conv2D, Conv1D, BatchNormalization, Activation, Add, AveragePooling1D, ZeroPadding2D,ZeroPadding1D, MaxPooling2D,MaxPooling1D, AveragePooling2D, Flatten ,Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Reshape
import numpy as np

def GRUAndConv(dim1,dim2,dim3):
  rate = 0.3
  input_im = Input(shape=(dim1, dim2 ,dim3))
  x = Conv2D(512, kernel_size=(2), strides=(2))(input_im)
  x = Dropout(rate)(x)
  x = MaxPooling2D(pool_size=2, strides=1)(x)
  x = BatchNormalization(axis=(1, 2))(x)
  x = Conv2D(256, kernel_size=(2), strides=(1))(x)
  x = BatchNormalization(axis=(1, 2))(x)
  x = Dropout(rate)(x)
  x = Flatten()(x)
  x = Dense(6)(x)
  model = Model(inputs=input_im, outputs=x, name='multipleGRU')

  return model