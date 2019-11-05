
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import utils
from tensorflow.keras.optimizers import SGD
from numpy import loadtxt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import multi_gpu_model


print("All packages loaded")

DATADIR = "/data/rkatyal/Images/ImageExport"
os.chdir(DATADIR)
CATEGORIES = list(filter(os.path.isdir, os.listdir(os.getcwd())))
CLASS_MODE = 'categorical'
LR = 0.001
IMG_SIZE = 200
BATCH_SIZE = 32
EPOCH = 20
COLOR_MODE = 'grayscale'
CATEGORIES

# modified vgg model
model = Sequential()

model.add(ZeroPadding2D((1, 1), input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.0001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'],
              )
model.summary()

NAME = "project"
tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))

aug = ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")

#model.fit(X,np.array(y), batch_size=32, epochs=40, validation_split=0.3, callbacks=[tensorboard])


train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        '/data/rkatyal/Images/dataset/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        color_mode=COLOR_MODE)

validation_generator = test_datagen.flow_from_directory(
        '/data/rkatyal/Images/dataset/validation',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        color_mode=COLOR_MODE)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size

# Creating multi-GPU model

parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer= sgd , metrics=['accuracy'])


history= model.fit_generator(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=EPOCH,
        validation_data=validation_generator,
        validation_steps=STEP_SIZE_VALID, callbacks=[tensorboard])


#with open('/data/rkatyal/Images/dataset/HistoryDict', 'wb') as file_pi:
#        pickle.dump(history.history, file_pi)

parallel_model.save("/data/rkatyal/Images/dataset/par_model_2.h5")
print("Saved model to disk")

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy for 10 Classes ')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('accuracy.png')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss for 10 Classes')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('loss.png')


# load model
model = load_model('/data/rkatyal/Images/dataset/par_model_2.h5')

# summarize model.
model.summary()

test_generator = test_datagen.flow_from_directory(
    directory=r"/data/rkatyal/Images/dataset/test",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)

# Evaluate the model
model.evaluate_generator(generator=validation_generator,
steps=STEP_SIZE_VALID)
