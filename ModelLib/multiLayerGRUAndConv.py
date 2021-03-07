from tensorflow.keras.backend import l2_normalize
from tensorflow.keras import activations
from tensorflow.keras import Input
from tensorflow.keras.layers import Softmax,LSTM,GRU,Conv1D,Conv2D, Conv1D, BatchNormalization, Activation, Add, AveragePooling1D, ZeroPadding2D,ZeroPadding1D, MaxPooling2D,MaxPooling1D, AveragePooling2D, Flatten ,Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Reshape
import numpy as np

# this is great only that it's slow and it overfits like crazy - need to see what can be done
def GRUAndConv(dim1,dim2,dim3):
  rate = 0.5
  numOfGRUUnits = 30
  input_im = Input(shape=(dim1, dim2,dim3))
  x = Conv2D(128, kernel_size=(2,2), strides=(2,2))(input_im)
  x = BatchNormalization()(x)
  x = activations.selu(x)
  x = Reshape((x.shape[1], int(x.shape[2]*x.shape[3])))(x)
  # keep in mind that in case of the GRU layers, 
  # if you want recurrent dropout the memory needed increases dramatically, but it yields better results
  x = GRU(units=numOfGRUUnits,dropout=rate,return_sequences=True)(x)
  x = BatchNormalization()(x)
  x = GRU(units=numOfGRUUnits, input_shape=(dim1, dim2), dropout=rate, return_sequences=True)(x)
  x = BatchNormalization()(x)
  x = GRU(units=numOfGRUUnits, input_shape=(dim1, dim2), dropout=rate)(x)
  x = Reshape((x.shape[1],1))(x)
  x = AveragePooling1D(pool_size=2,strides=2)(x)
  x = BatchNormalization()(x)
  x = Dense(64)(x)
  x = Flatten()(x)
  x = Dense(6)(x)
  model = Model(inputs=input_im, outputs=x, name='multipleLayerGRUAndCNN')

  return model