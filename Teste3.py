from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception, preprocess_input
from keras.models import load_model
import keras.callbacks as kcall
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage.transform import resize
from tqdm import tqdm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

img_width = 299
img_height = 299

import os

# print(os.listdir("/home/patrik/jogos_mortais"))

# Any results you write to the current directory are saved as output.
# train_NORMAL    = !find ../input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL/ -type f  -exec file {} \+ | grep -c -i 'image'
# train_PNEUMONIA = !find ../input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA/ -type f  -exec file {} \+ | grep -c -i 'image'
# val_NORMAL      = !find ../input/chest-xray-pneumonia/chest_xray/chest_xray/val/NORMAL/ -type f  -exec file {} \+ | grep -c -i 'image'
# val_PNEUMONIA   = !find ../input/chest-xray-pneumonia/chest_xray/chest_xray/val/PNEUMONIA/ -type f  -exec file {} \+ | grep -c -i 'image'
# test_NORMAL     = !find ../input/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL/ -type f  -exec file {} \+ | grep -c -i 'image'
# test_PNEUMONIA  = !find ../input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA/ -type f  -exec file {} \+ | grep -c -i 'image'

# rain_dir = "/home/patrik/jogos_mortais/chest_xray/train/"
# test_dir =  "/home/patrik/jogos_mortais/chest_xray/test/"
train_dir = "/home/patrik/jogos_mortais/chest_xray/train/"
test_dir = "/home/patrik/jogos_mortais/chest_xray/test/"

# def get_data(folder):
#     X_NORMAL = []
#     X_PNEUMONIA = []
#     for folderName in os.listdir(folder):
#         if not folderName.startswith('.'):
#             if folderName in ['NORMAL']:
#             	for image_filename in tqdm(os.listdir(folder + folderName)):
#                   img_file = cv2.imread(folder + folderName + '/' + image_filename)
#                   if img_file is not None:
#                     img_file = skimage.transform.resize(img_file, (img_width, img_height, 3))
#                     X_NORMAL.append(img_file)

#             elif folderName in ['PNEUMONIA']:
#               for image_filename in tqdm(os.listdir(folder + folderName)):
#                 img_file = cv2.imread(folder + folderName + '/' + image_filename)
#                 if img_file is not None:
#                   img_file = skimage.transform.resize(img_file, (img_width, img_height, 3))
#                   X_PNEUMONIA.append(img_file)

#     return X_NORMAL, X_PNEUMONIA
# X_train_normal, X_train_pneumonia = get_data(train_dir)

# os.mkdir("/home/patrik/jogos_mortais/chest_xray/train/val")
# os.mkdir("/home/patrik/jogos_mortais/chest_xray/train/val/NORMAL")
# os.mkdir("/home/patrik/jogos_mortais/chest_xray/train/val/PNEUMONIA")

# #dividir imagens normal para treino e validacao
# num = len(X_train_normal)
# ind = np.random.permutation(num)
# num_train_normal = int(len(ind)*0.8)
# num_val_normal = num-num_train_normal
# X_train_normal = ind[0:num_train_normal]
# X_val_normal = ind[num_train_normal:num]

# #dividir imagens pneumonia para treino e validacao
# num = len(X_train_pneumonia)
# ind = np.random.permutation(num)
# num_train_pneumonia = int(len(ind)*0.8)
# num_val_pneumonia = num-num_train_pneumonia
# X_train_pneumonia = ind[0:num_train_pneumonia]
# X_val_pneumonia = ind[num_train_pneumonia:num]

# salvar imagens em val/normal e val/pneumonia
# i = 0
# for image in list(X_val_normal):
#     nameimg = '/home/patrik/jogos_mortais/chest_xray/val/NORMAL/' + str(i) + '.jpeg'
#     cv2.imwrite(nameimg, image)
#     i+=1
# for image in list(X_val_pneumonia):
#     nameimg = '/home/patrik/jogos_mortais/chest_xray/val/PNEUMONIA/'+ str(i) + '.jpeg'
#     cv2.imwrite(nameimg, image)
#     i+=1


# X_test_normal, X_test_pneumonia = get_data(test_dir)

# num = len(X_test_normal)
# ind = np.random.permutation(num)
# num_test_normal = int(len(ind)*1)
# X_test_normal = ind[0:num_test_normal]

# num = len(X_test_pneumonia)
# ind = np.random.permutation(num)
# num_test_pneumonia = int(len(ind)*1)
# num_test_pneumonia = num-num_test_pneumonia
# X_test_pneumonia = ind[0:num_test_pneumonia]


# train_data = np.array([int(X_train_normal),int(X_train_pneumonia)])
# val_data = np.array([int(X_val_normal),int(X_val_pneumonia)])
# test_data = np.array([int(X_test_normal),int(X_test_pneumonia)])


# index = np.arange(2)
# bar_width = 0.25
# opacity = 0.7

# rects1 = plt.bar(index, train_data, bar_width,
#                 alpha=opacity, color='b',
#                 label='Train')
# rects2 = plt.bar(index + bar_width, val_data, bar_width,
#                 alpha=opacity, color='r', tick_label = ('Normal', 'Pneumonia'),
#                 label='Val')
# rects3 = plt.bar(index + 2*bar_width, test_data, bar_width,
#                 alpha=opacity, color='g', tick_label = ('Normal', 'Pneumonia'),
#                 label='test')

# plt.xlabel('Class')
# plt.ylabel('Number of examples')
# plt.title('Total examples per set')
# plt.xticks(index + bar_width)
# plt.legend()

# plt.show()

## Intilizing variables
output_classes = 2
learning_rate = 0.0001
channel = 3
training_examples = 5216
batch_size = 30
epochs = 5
resume_model = False
training_data_dir = '/home/patrik/jogos_mortais/chest_xray/train'
val_data_dir = '/home/patrik/jogos_mortais/chest_xray/val'
test_data_dir = '/home/patrik/jogos_mortais/chest_xray/test'
trained_model_dir = '/home/patrik/jogos_mortais/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'

if resume_model == False:
    ## Model Defination
    model = Sequential()
    model.add(
        Xception(weights=trained_model_dir, include_top=False, pooling='avg', input_shape=(img_width, img_height, 3)))
    # model.add(Dense(units = 100 , activation = 'relu'))
    model.add(Dense(units=output_classes, activation='softmax'))

    model.layers[0].trainable = True

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy'])

    ## model.load_weights('xception_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)

    ## Image generator function for training and validation
    img_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_img_generator = img_generator.flow_from_directory(
        training_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    val_img_generator = img_generator.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        class_mode='categorical')

    for i, layer in enumerate(model.layers):
        print('Layer: ', i + 1, ' Name: ', layer.name)

## Callbacks for model training
early_stop = kcall.EarlyStopping(monitor='acc', min_delta=0.0001)
tensorboard = kcall.TensorBoard(log_dir='./tensorboard-logs', write_grads=1, batch_size=batch_size)


class LossHistory(kcall.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))


history = LossHistory()

## Training only the newly added layer
if resume_model:
    model = load_model('chest_xray.h5')
else:
    model.fit_generator(train_img_generator,
                        steps_per_epoch=training_examples // batch_size,
                        epochs=epochs,
                        validation_data=val_img_generator,
                        validation_steps=1,
                        callbacks=[early_stop, history])

    ## saving model
    model.save('chest_xray.h5')

## Image generator function for testing
test_img_generator = img_generator.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False)
test_accu = model.evaluate_generator(test_img_generator, steps=624 // batch_size)

print('Accuracy on test data is:', test_accu[1])
print('Loss on test data is:', test_accu[0])

plt.plot(history.losses, 'b--', label='Training')
plt.plot(len(history.losses) - 1, test_accu[0], 'go', label='Test')

plt.xlabel('# of batches trained')
plt.ylabel('Training loss')

plt.title('Training loss vs batches trained')

plt.legend()

plt.ylim(0, 1.2)
plt.show()
plt.savefig('./accuracy_loss_teste3.png')

plt.plot(history.acc, '--', label='Training')
plt.plot(len(history.acc) - 1, test_accu[1], 'go', label='Test')

plt.xlabel('# of batches trained')
plt.ylabel('Training accuracy')

plt.title('Training accuracy vs batches trained')

plt.legend(loc=4)
plt.ylim(0, 1.1)
plt.show()
plt.savefig('./accuracy_curve_teste3.png')