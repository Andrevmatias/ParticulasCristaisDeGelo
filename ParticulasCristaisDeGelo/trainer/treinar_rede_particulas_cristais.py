import os

import matplotlib
matplotlib.use("agg") 
import matplotlib.pyplot as plt

import keras as keras
from keras.models import Model
from keras.layers import Input, Dense, Activation,Flatten, Conv2D
from keras.layers import Dropout, MaxPooling2D
from keras import callbacks
from keras import backend as K
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import FunctionTransformer

import pandas as pd

import tensorflow as tf

from tensorflow.python.lib.io import file_io

import h5py
import numpy as np
from numpy import genfromtxt

import argparse

## A adaptação para rodar no Google Cloud é baseada em 
# https://medium.com/@natu.neeraj/training-a-keras-model-on-google-cloud-ml-cb831341c196

## Retorna todos os arquivos h5 de um diretório
def get_all_h5_files():
  files = []

  for r, _, f in os.walk("data"):
      for file in f:
          if '.h5' in file:
              files.append(os.path.join(r, file))

  return files

## Carrega os dados de um arquivo h5 em formato compatível com a rede
def get_data(h5):
  features = []
  target = h5['size'].value

  #TODO: Carregar features para rodar a rede

  return [features, target]

## Carrega os dados a serem usados para treinamento
def load_dataset():
    h5_files = get_all_h5_files()

    raw_data = map(lambda f: h5py.File(f, 'r') , h5_files)
    features = []
    targets = []

    for h5 in raw_data:
      ftr, tgt = get_data(h5)
      features.append(ftr)
      targets.append(tgt)

    return [features, targets]

## Carrega os dados a serem usados para treinamento
def load_dataset_csv(job_dir):
  with file_io.FileIO(job_dir + '/data/DadosIntensidade.csv', 'r') as f:
    dataset = pd.read_csv(f, header=None).values
    #transformer = FunctionTransformer(np.log1p, validate=True)
    #transformer.transform(dataset)

    features = dataset[:,0:-1]
    targets = dataset[:,-1]

    return [features, targets]

##Cria o modelo da rede a partir do formato do input
def create_model(input_shape):
    X_input = Input(input_shape)

    ##Dense Layer
    X = Dense(1024, activation='relu', name='dense_1')(X_input)
    X = Dense(1, activation='linear', name='output')(X)

    ##The model object
    model = Model(inputs = X_input, outputs = X, name='particulasModel')

    return model

def save_model(job_dir, model):
  model.save('model.h5')
  with file_io.FileIO('model.h5', mode='r') as input_f:
    with file_io.FileIO(job_dir + 'model.h5', mode='w+') as output_f:
      output_f.write(input_f.read())

#Roda uma NN com uma camada escondida de 1024 neurônios
def run_nn(features_train, features_test, targets_train, targets_test, job_dir):
  logs_path = job_dir + '/logs/'

  ## Gera o modelo
  model = create_model((features_train.shape[1],))

  ## Compila o modelo
  model.compile(optimizer = "adam" , loss = "mean_squared_error", metrics = ["mean_squared_error", r2_keras], )

  ## Gera o log de execução da rede para o Tensorboard
  tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

  ## Treina o modelo
  model.fit(x = features_train, 
            y = targets_train, 
            epochs = 250,
            verbose = 1, 
            batch_size = 100, 
            callbacks = [tensorboard], 
            validation_split = .1)

  ## Salva o modelo
  save_model(job_dir, model)

  #model.test_on_batch(features_test, targets_test)

  predicted = model.predict(features_test)

  plot_scatter(targets_test, predicted, job_dir)

  scores = model.evaluate(features_test, targets_test, verbose=True)

  print(model.metrics_names)
  print(scores)

#Roda uma SVN conforme os parâmetros descritos no artigo
def run_svm(features_train, features_test, targets_train, targets_test, job_dir):
  regr = SVR(C=1000.0, coef0=0.001, degree=3, epsilon=0.1, gamma=0.1,
    kernel='rbf', shrinking=True, tol=0.001, verbose=True)
  regr.fit(features_train, targets_train)
  r2 = regr.score(features_test, targets_test)
  print('R2: {0}'.format(r2))

  predict = regr.predict(features_test)
  plot_scatter(targets_test, predict, job_dir)

#Plota real x predito
def plot_scatter(target, predict, job_dir):
  plt.scatter(target, predict, c='red')
  plt.xlabel('Real Size')
  plt.ylabel('Predicted Size')

  with file_io.FileIO(job_dir + '/scatter.png', mode='w+') as output_f:
    plt.savefig(output_f, format='png')

  #plt.show()

def test_best_nn(test_features, test_targets, job_dir):
  with file_io.FileIO(job_dir + 'model.h5', mode='r') as input_f:
    model = load_model(input_f)

    scores = model.evaluate(test_features, test_targets, verbose=True)

    print(model.metrics_names)
    print(scores)
  

def main(job_dir,**args):

    ##Usar GPU
    with tf.device('/device:GPU:0'):
      ##Carrega dados
      features, targets = load_dataset_csv(job_dir)

      features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size = .3)

      run_nn(features_train, features_test, targets_train, targets_test, job_dir)
      #run_svm(features_train, features_test, targets_train, targets_test, job_dir)

      #test_best_nn(features_test, targets_test, job_dir)

#Calcula o R-quadrado como score para o keras
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

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