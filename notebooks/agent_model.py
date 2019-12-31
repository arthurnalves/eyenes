import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, LSTM, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Reshape
from keras.layers import Softmax, Input, Concatenate, Embedding, Activation, Lambda, Input
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras.models import model_from_json
from keras.initializers import glorot_uniform

import numpy as np


class AgentModel:
    
    model = None
    input_shape = None
    output_dim = None
    activation = None
    buffer = None
    
    def __init__(self, buffer, input_shape, output_dim, activation = 'relu'):
        self.buffer = buffer
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.activation = activation
        self.get_model()
    
    def get_conv_model(self):
        w, h, _ = self.input_shape
        max_dim = max([w,h])
        activation = self.activation

        conv_model = Sequential()

        conv_model.add(Lambda(lambda x: x/255., input_shape = self.input_shape))        
        conv_model.add(ZeroPadding2D(padding = ((max_dim - w)//2, (max_dim - h)//2)))
        conv_model.add(MaxPooling2D((8,8)))

        conv_model.add(Conv2D(4, 4, strides=(2, 2), padding="same", activation=activation))
        conv_model.add(Conv2D(8, 4, strides=(2, 2), padding="same", activation=activation))
        conv_model.add(Conv2D(16,4, strides=(2, 2), padding="same", activation=activation))

        conv_model.add(Flatten())
        conv_model.add(Dense(64, activation= activation))
        return conv_model
    
    def get_model(self):
        
        input_imgs = []
        for i in range(self.buffer):
            input_imgs.append(Input(self.input_shape))

        output_imgs = []
        for input_img in input_imgs:
            output_img = input_img
            for layer in self.get_conv_model().layers:
                output_img = layer(output_img)
            output_imgs.append(output_img)
    
        output = Concatenate(axis = -1)(output_imgs)
        output = Reshape([self.buffer, 64])(output)
        output = LSTM(self.output_dim, activation = 'softmax')(output)
        
        self.model = Model(inputs = input_imgs, outputs = output)
        
    def mutate(self, freq, intensity):
        mutated_weights = []
        for weight in self.model.get_weights():
            A = np.random.choice([0,1], p=[1-freq, freq], size = np.shape(weight))
            A = A*intensity
            glorot = glorot_uniform().__call__(np.shape(weight))
            mutated_weights.append(A*glorot + weight)
            
        self.model.set_weights(mutated_weights)
            
    def summary(self):
        return self.model.summary()
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        return self.model.set_weights(weights)
    
    def predict(self, inputs):
        return self.model.predict(inputs)
