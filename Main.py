import os
from glob import glob
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
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.xception import Xception, preprocess_input
# from keras.applications.mobilenet import MobileNet
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc


# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib inline

import warnings

warnings.filterwarnings("ignore")

epocas = 7
img_width = 1000
img_height = 1000
batch_size = 32

train_dir = "/home/patrik/jogos_mortais/chest_xray/train/"
test_dir = "/home/patrik/jogos_mortais/chest_xray/test/"


def get_data(folder):
    X = []
    y = []
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['NORMAL']:
                label = 0
            elif folderName in ['PNEUMONIA']:
                label = 1
            else:
                label = 2
            for image_filename in tqdm(os.listdir(folder + folderName)):
                img_file = cv2.imread(folder + folderName + '/' + image_filename)
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, (img_width, img_height, 3))
                    # img_file = scipy.misc.imresize(arr=img_file, size=(150, 150, 3))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y


X_train, y_train = get_data(train_dir)
X_test, y_test = get_data(test_dir)
# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from keras.utils.np_utils import to_categorical

y_trainHot = to_categorical(y_train, num_classes=2)
y_testHot = to_categorical(y_test, num_classes=2)


def plotHistogram(a):
    """
    Plot histogram of RGB Pixel Intensities
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(a)
    plt.axis('off')
    histo = plt.subplot(1, 2, 2)
    histo.set_ylabel('Count')
    histo.set_xlabel('Pixel Intensity')
    n_bins = 30
    plt.hist(a[:, :, 0].flatten(), bins=n_bins, lw=0, color='r', alpha=0.5);
    plt.hist(a[:, :, 1].flatten(), bins=n_bins, lw=0, color='g', alpha=0.5);
    plt.hist(a[:, :, 2].flatten(), bins=n_bins, lw=0, color='b', alpha=0.5);


plotHistogram(X_train[1])

multipleImages = glob('/home/patrik/jogos_mortais/chest_xray/train/NORMAL/**')


def plotThreeImages(images):
    r = random.sample(images, 3)
    plt.figure(figsize=(16, 16))
    plt.subplot(131)
    plt.imshow(cv2.imread(r[0]))
    plt.subplot(132)
    plt.imshow(cv2.imread(r[1]))
    plt.subplot(133)
    plt.imshow(cv2.imread(r[2]));


plotThreeImages(multipleImages)

print("No Pneumonia")
multipleImages = glob('/home/patrik/jogos_mortais/chest_xray/train/NORMAL/**')
i_ = 0
plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.subplots_adjust(wspace=0, hspace=0)
for l in multipleImages[:25]:
    im = cv2.imread(l)
    im = cv2.resize(im, (128, 128))
    plt.subplot(5, 5, i_ + 1)  # .set_title(l)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB));
    plt.axis('off')
    i_ += 1

print("Yes Pneumonia")
multipleImages = glob('/home/patrik/jogos_mortais/chest_xray/train/PNEUMONIA/**')

i_ = 0
plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.subplots_adjust(wspace=0, hspace=0)
for l in multipleImages[:25]:
    im = cv2.imread(l)
    im = cv2.resize(im, (128, 128))
    plt.subplot(5, 5, i_ + 1)  # .set_title(l)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB));
    plt.axis('off')
    i_ += 1

map_characters = {0: 'No Pneumonia', 1: 'Yes Pneumonia'}
dict_characters = map_characters
import seaborn as sns

df = pd.DataFrame()
df["labels"] = y_train
lab = df['labels']
dist = lab.value_counts()
sns.countplot(lab)
print(dict_characters)

# Helper Functions  Learning Curves and Confusion Matrix

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class MetricsCheckpoint(Callback):
    """Callback that saves metrics after each epoch"""

    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)


def plotKerasLearningCurve():
    plt.figure(figsize=(10, 5))
    metrics = np.load('logs.npy')[()]
    filt = ['acc']  # try to add 'loss' to see the loss learning curve
    for k in filter(lambda x: np.any([kk in x for kk in filt]), metrics.keys()):
        l = np.array(metrics[k])
        plt.plot(l, c='r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')
        x = np.argmin(l) if 'loss' in k else np.argmax(l)
        y = l[x]
        plt.scatter(x, y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')
        plt.text(x, y, '{} = {:.4f}'.format(x, y), size='15', color='r' if 'val' not in k else 'b')
    plt.legend(loc=4)
    plt.axis([0, None, None, None]);
    plt.grid()
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_learning_curve(history):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./accuracy_curve.png')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./loss_curve.png')


map_characters1 = {0: 'No Pneumonia', 1: 'Yes Pneumonia'}
class_weight1 = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
weight_path1 = '/home/patrik/jogos_mortais/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
weight_path2 = '/home/patrik/jogos_mortais/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
weight_path3 = '/home/patrik/jogos_mortais/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
weight_path4 = '/home/patrik/jogos_mortais/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
weight_path5 = '/home/patrik/jogos_mortais/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'

# Patrik = tambem pegar os pesos da ResNet para colocar aqui em outro pretreino
# tamanho padrao VGG: 224,224,3 variar
pretrained_model_1 = VGG16(weights=weight_path1, include_top=False, input_shape=(img_width, img_height, 3))
# tamanho padrao inception 299,299,3
pretrained_model_2 = InceptionV3(weights=weight_path2, include_top=False, input_shape=(img_width, img_height, 3))

# tamaho padrao resnet 224x224x3
pretrained_model_3 = ResNet50(weights=weight_path3, include_top=False, input_shape=(img_width, img_height, 3))

# tamanho padrao 299x299
pretrained_model_4 = Xception(weights=weight_path5, include_top=False, input_shape=(img_width, img_height, 3))

# tamanho padrao 299x299
pretrained_model_5 = InceptionResNetV2(weights=weight_path6, include_top=False, input_shape=(img_width, img_height, 3))

# pretrained_model_4 = ResNet152(weights = weight_path4, include_top=False, input_shape=(img_width, img_height, 3))

# mudar taxa de aprendizado e incluir decaimento
optimizer1 = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.01)
# optimizer2 = Adadelta(lr=0.01, rho=0.95, epsilon=None, decay=0.0)
# optimizer3 = SGD(lr=0.0001, momentum=0.9, decay=0.001, nesterov=True)

dropout = AlphaDropout
dropout_rate = 0.5


def pretrainedNetwork(xtrain, ytrain, xtest, ytest, pretrainedmodel, pretrainedweights, classweight, numclasses,
                      numepochs, optimizer, labels):
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

    model.summary()
    # Fit model
    history = model.fit(xtrain, ytrain, epochs=numepochs, class_weight=classweight, validation_data=(xtest, ytest),
                        verbose=1, callbacks=[MetricsCheckpoint('logs')])
    # Evaluate model
    score = model.evaluate(xtest, ytest, verbose=0)
    print('\nKeras CNN - accuracy:', score[1], '\n')
    y_pred = model.predict(xtest)
    # print('\n', sklearn.metrics.classification_report(np.where(ytest > 0)[1], np.argmax(y_pred, axis=1), target_names=list(labels.values())), sep='')
    Y_pred_classes = np.argmax(y_pred, axis=1)
    Y_true = np.argmax(ytest, axis=1)
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    plotKerasLearningCurve()
    plt.show()
    plt.savefig('Curva_1_inception_resnet_v2.png')
    plot_learning_curve(history)
    plt.show()
    plt.savefig('Curva_2_inception_resnet_v2.png')
    plot_confusion_matrix(confusion_mtx, classes=list(labels.values()))
    plt.show()
    plt.savefig('Matriz_inception_resnet_v2.png')
    return model


# pretrainedNetwork(X_train, y_trainHot, X_test, y_testHot,pretrained_model_1,weight_path1,class_weight1,2,3,optimizer1,map_characters1)

# pretrainedNetwork(X_train, y_trainHot, X_test, y_testHot,pretrained_model_2,weight_path2,class_weight1,2,3,optimizer1,map_characters1)


# Deal with imbalanced class sizes below
# Make Data 1D for compatability upsampling methods
X_trainShape = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
X_testShape = X_test.shape[1] * X_test.shape[2] * X_test.shape[3]
X_trainFlat = X_train.reshape(X_train.shape[0], X_trainShape)
X_testFlat = X_test.reshape(X_test.shape[0], X_testShape)
Y_train = y_train
Y_test = y_test
# ros = RandomOverSampler(ratio='auto')
ros = RandomUnderSampler(ratio='auto')
X_trainRos, Y_trainRos = ros.fit_sample(X_trainFlat, Y_train)
X_testRos, Y_testRos = ros.fit_sample(X_testFlat, Y_test)
# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_trainRosHot = to_categorical(Y_trainRos, num_classes=2)
Y_testRosHot = to_categorical(Y_testRos, num_classes=2)
# Make Data 2D again
for i in range(len(X_trainRos)):
    height, width, channels = img_width, img_height, 3
    X_trainRosReshaped = X_trainRos.reshape(len(X_trainRos), height, width, channels)
for i in range(len(X_testRos)):
    height, width, channels = img_width, img_height, 3
    X_testRosReshaped = X_testRos.reshape(len(X_testRos), height, width, channels)
# Plot Label Distribution
dfRos = pd.DataFrame()
dfRos["labels"] = Y_trainRos
labRos = dfRos['labels']
distRos = lab.value_counts()
sns.countplot(labRos)
print(dict_characters)

class_weight1 = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
print("Old Class Weights: ", class_weight1)

class_weight2 = class_weight.compute_class_weight('balanced', np.unique(Y_trainRos), Y_trainRos)
print("New Class Weights: ", class_weight2)

class_weight3 = class_weight.compute_class_weight('balanced', np.unique(Y_trainRos), Y_trainRos)
print("ResNet50 Class Weights: ", class_weight3)

class_weight4 = class_weight.compute_class_weight('balanced', np.unique(Y_trainRos), Y_trainRos)
print("ResNet50 Class Weights: ", class_weight4)

# pretrainedNetwork(X_trainRosReshaped, Y_trainRosHot, X_testRosReshaped, Y_testRosHot,pretrained_model_1,weight_path1,class_weight2,2,epocas,optimizer1,map_characters1)

# pretrainedNetwork(X_trainRosReshaped, Y_trainRosHot, X_testRosReshaped, Y_testRosHot,pretrained_model_2,weight_path2,class_weight2,2,epocas,optimizer1,map_characters1)

# Pesos 4 - xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
# pretrainedNetwork(X_trainRosReshaped, Y_trainRosHot, X_testRosReshaped, Y_testRosHot,pretrained_model_4,weight_path3,class_weight2,2,epocas,optimizer1,map_characters1)

# Pesos 5 - inception_resnet_v2
pretrainedNetwork(X_trainRosReshaped, Y_trainRosHot, X_testRosReshaped, Y_testRosHot, pretrained_model_5, weight_path3,
                  class_weight2, 2, epocas, optimizer1, map_characters1)

# pretrainedNetwork(X_trainRosReshaped, Y_trainRosHot, X_testRosReshaped, Y_testRosHot,pretrained_model_4,weight_path4,class_weight2,2,epocas,optimizer1,map_characters1)
# resnet152 = score[1]
