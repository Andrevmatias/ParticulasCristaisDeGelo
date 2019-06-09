import keras as keras
from keras.models import Model
from keras.layers import Input, Dense, Activation,Flatten, Conv2D
from keras.layers import Dropout, MaxPooling2D
from keras import callbacks

import tensorflow as tf

from tensorflow.python.lib.io import file_io

import numpy as np

import argparse

## A adaptação para rodar no Google Cloud é baseada em 
# https://medium.com/@natu.neeraj/training-a-keras-model-on-google-cloud-ml-cb831341c196

##Carrega os dados a serem usados para treinamento
def load_dataset():
    return tf.contrib.learn.datasets.load_dataset("mnist")

##Cria o modelo da rede a partir do formato do input
def create_model(input_shape):
    X_input = Input(input_shape)

    ##Convolutional Layer 1
    X = Conv2D(
    filters=32,
    kernel_size=(5, 5),
    strides=(1, 1),
    padding='same',
    name = 'conv1'
    )(X_input)
    X = Activation('relu')(X)

    ##Max pooling layer 1
    X = MaxPooling2D(pool_size=(2, 2), strides =2, name = 'maxpool1')(X)

    ##Convolutional Layer 2
    X = Conv2D(
    filters=64,
    kernel_size=[5,5],
    padding='same',
    name = 'conv2'
    )(X)
    X = Activation('relu')(X)

    ##Max Pooling Layer 2
    X = MaxPooling2D(pool_size=(2, 2), strides =2, name = 'maxpool2')(X)

    ##Flatten
    X = Flatten()(X)

    ##Dense Layer
    X = Dense(1024, activation='relu', name='dense_1')(X)

    ##Dropout layer
    X = Dropout(0.4, name = 'dropout')(X)

    ##dense 2 layer
    X = Dense(10, activation='softmax', name ='dense_2')(X)

    ##The model object
    model = Model(inputs = X_input, outputs = X, name='cnnMINSTModel')

    return model

def save_model(job_dir, model):
  model.save('model.h5')
  with file_io.FileIO('model.h5', mode='r') as input_f:
    with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
      output_f.write(input_f.read())

def main(job_dir,**args):
    logs_path = job_dir + '/logs/'

    ##Usar GPU
    with tf.device('/device:GPU:0'):

        ##Carrega dados
        dados_particulas = load_dataset()

        train_data = dados_particulas.train.images
        train_labels = np.asarray(dados_particulas.train.labels, dtype=np.int32)
        eval_data = dados_particulas.test.images
        eval_labels = np.asarray(dados_particulas.test.labels, dtype=np.int32)

        ##Pré-processamento
        train_labels = keras.utils.np_utils.to_categorical(train_labels, 10)
        eval_labels = keras.utils.np_utils.to_categorical(eval_labels, 10)
        train_data = np.reshape(train_data, [-1, 28, 28, 1])
        eval_data = np.reshape(eval_data, [-1,28,28,1])

        ## Gera o modelo
        model = create_model(train_data.shape[1:])

        ## Compila o modelo
        model.compile(optimizer = "Adam" , loss = "binary_crossentropy", metrics = ["accuracy"])

        ## Gera o log de execução da rede para o Tensorboard
        tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

        ## Treina o modelo
        model.fit(x = train_data, y = train_labels, epochs = 4,verbose = 1, batch_size=100, callbacks=[tensorboard], validation_data=(eval_data,eval_labels) )

        ## Salva o modelo
        save_model(job_dir, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='Pasta de trabalho',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

main(**arguments)