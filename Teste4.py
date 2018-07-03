import os from glob import glob
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import random
import cv2
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sns
import zlib
import itertools
import sklearn
import itertools
import scipy
import skimage
from skimage.transform import resize
import csv
from tqdm import tqdm
from sklearn import model_selection
from sklearn.model_selection import train_test_split, learning_curve, KFold, cross_val_score, StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import keras
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, \
    BatchNormalization
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import class_weight
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta, RMSprop
from keras.models import Sequential, model_from_json
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D, AveragePooling2D, \
    BatchNormalization
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from keras.layers.noise import AlphaDropout
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# from keras.applications.mobilenet import MobileNet
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc


keras.callbacks.TensorBoard(log_dir='/home/patrik/jogos_mortais/',
                            histogram_freq=0,
                            batch_size=32,
                            write_graph=True,
                            write_grads=False,
                            write_images=False,
                            embeddings_freq=0,
                            embeddings_layer_names=None,
                            embeddings_metadata=None)

# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib inline

import warnings

warnings.filterwarnings("ignore")

epocas = 2
img_width = 224
img_height = 224
batch_size = 64

train_dir = "/home/patrik/jogos_mortais/chest_xray/train/"
test_dir = "/home/patrik/jogos_mortais/chest_xray/test/"

# Helper Functions  Learning Curves and Confusion Matrix

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

weight_path1 = '/home/patrik/jogos_mortais/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
weight_path2 = '/home/patrik/jogos_mortais/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
weight_path3 = '/home/patrik/jogos_mortais/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
weight_path4 = '/home/patrik/jogos_mortais/resnet152_weights_tf.h5'
# Patrik = tambem pegar os pesos da ResNet para colocar aqui em outro pretreino
# tamanho padrao VGG: 224,224,3 variar
pretrained_model_1 = VGG16(weights=weight_path1, include_top=False, input_shape=(img_width, img_height, 3))
# tamanho padrao inception 299,299,3
pretrained_model_2 = InceptionV3(weights=weight_path2, include_top=False, input_shape=(img_width, img_height, 3))

# tamaho padrao resnet 224x224x3
pretrained_model_3 = ResNet50(weights=weight_path3, include_top=False, input_shape=(img_width, img_height, 3))

# pretrained_model_4 = ResNet152(weights = weight_path4, include_top=False, input_shape=(img_width, img_height, 3))

# mudar taxa de aprendizado e incluir decaimento
# optimizer1 = keras.optimizers.RMSprop(lr=0.0001)
optimizer1 = SGD(lr=0.0001, momentum=0.9, decay=0.0005, nesterov=True)

dropout = AlphaDropout
dropout_rate = 0.05


def pretrainedNetwork(pretrainedmodel, pretrainedweights, numclasses, numepochs, optimizer):
    base_model = pretrainedmodel  # Topless

    # Add top layer
    x = base_model.output
    x = dropout(dropout_rate)(x)
    x = Flatten()(x)
    x = Dense(4096)(x)
    x = dropout(dropout_rate)(x)
    x = Dense(4096)(x)
    predictions = Dense(numclasses, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    # Train top layer
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]

    #################################################################################################################
    # This is the augmentantion configuration we will for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        shear_range=0.2,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # olhar a documentacao de flow_from_directory
    # o train_generator precisa saber a porcentagem de desfolha de cada imagem
    # pesquisar na internet qual o data generator para problemas de regressao
    # arquivo txt com o nome da imagem e a porcentagem de desfolha
    train_generator = train_datagen.flow_from_directory(
        'chest_xray/train',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    # passar a porcentagem de desfolha para o treinamento saber.

    validation_generator = test_datagen.flow_from_directory(
        'chest_xray/test',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    # Fit the model on the batches generated by datagen.flow().
    # total de imagens de treino = 5232
    # total de imagens de teste = 624
    model.fit_generator(
        train_generator,
        steps_per_epoch=167904 // batch_size,
        epochs=epocas,
        validation_data=validation_generator,
        validation_steps=167904 // batch_size)
    # workers=4)

    # Score trained model.
    # scores = model.evaluate(x_test, y_test, verbose=1)
    scores = model.evaluate(validation_generator, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # Score trained model on validation data
    validation_scores = model.evaluate_generator(
        validation_generator,
        steps=int(num_validation_images / batch_size))

    print('Validation Accuracy:', validation_scores[1])

    # model.save('jogos_mortais'+str(pretrainedmodel).h5py)
    # model.save('jogos_mortais'+str(pretrainedmodel).h5)
    return model

    #################################################################################################################


pretrainedNetwork(pretrained_model_1, weight_path1, 2, epocas, optimizer1)

pretrainedNetwork(pretrained_model_2, weight_path2, 2, epocas, optimizer1)

pretrainedNetwork(pretrained_model_3, weight_path3, 2, epocas, optimizer1)

print('CNN VGG - accuracy:', vgg, '\n')
print('CNN INCEPTION - accuracy:', inception, '\n')
print('CNN RESNET50 - accuracy:', resnet50, '\n')