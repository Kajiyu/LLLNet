#!/usr/bin/env python
# -*- coding: utf-8 -*-


from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, AveragePooling2D, Activation, concatenate, Dropout, warnings, Flatten, ZeroPadding2D, Dense
from keras.models import Model, Sequential
from keras.engine.topology import get_source_inputs
from keras.utils import get_file, plot_model, layer_utils


def fire_module(x, squeeze=16, expand=64):
    x = Convolution2D(squeeze, (1, 1), padding='valid')(x)
    x = Activation('relu')(x)
    left = Convolution2D(expand, (1, 1), padding='valid')(x)
    left = Activation('relu')(left)
    right = Convolution2D(expand, (3, 3), padding='same')(x)
    right = Activation('relu')(right)
    x = concatenate([left, right])
    return x


class LLLNet():
    def __init__(self):
        self.model = None
        self.discriminate_model = None
        self.audio_submodel = self._audio_submodel()
        self.image_submodel = self._image_submodel()
        self.model = self.make_model()
        self.model.summary()
        self.discriminate_model.summary()


    def __call__(self, model_option):
        if model_option == "train":
            return self.model
        elif model_option == "sound":
            return self.audio_submodel
        elif model_option == "image":
            return self.image_submodel
        else:
            return None
    

    def _audio_submodel(self):
        img_input = Input(shape=(199, 257, 1))

        x = ZeroPadding2D((1, 1))(img_input)
        x = Convolution2D(64, (3, 3))(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(64, (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(128, (3, 3))(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(128, (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(256, (3, 3))(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(256, (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(512, (3, 3))(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(512, (3, 3))(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((24, 32), strides=(2, 2))(x)

        x = Flatten()(x)
        _model = Model(inputs=img_input, outputs=x)
        return _model
    

    def _image_submodel(self):
        img_input = Input(shape=(224, 224, 3))
        x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid')(img_input)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        
        x = fire_module(x, squeeze=16, expand=64)
        x = fire_module(x, squeeze=16, expand=64)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        
        x = fire_module(x, squeeze=32, expand=128)
        x = fire_module(x, squeeze=32, expand=128)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        
        x = fire_module(x, squeeze=48, expand=192)
        x = fire_module(x, squeeze=48, expand=192)
        x = fire_module(x, squeeze=64, expand=256)
        x = fire_module(x, squeeze=64, expand=256)

        x = Dropout(0.5)(x)
        x = AveragePooling2D((13, 13))(x)
        x = Flatten()(x)
        _model = Model(inputs=img_input, outputs=x)
        return _model
    

    def make_model(self):
        tmp_input = concatenate([self.audio_submodel.outputs[0], self.image_submodel.outputs[0]], axis=-1)
        input_tensor = Input(shape=(None,1024))
        x = Dense(128)(input_tensor)
        x = Dropout(0.5)(x)
        x = Activation("relu")(x)
        x = Dense(2)(x)
        x = Activation("softmax")(x)
        self.discriminate_model = Model(inputs=input_tensor, outputs=x)
        output = self.discriminate_model(tmp_input)
        _model = Model(inputs=[self.audio_submodel.inputs[0], self.image_submodel.inputs[0]], outputs=output)
        return _model
