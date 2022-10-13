
# -*- coding: utf-8 -*-
"""
@author: mdrafsunsheikh
"""

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from os import listdir
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import random
import os


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import random
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from itertools import cycle
import os

import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
 
from sklearn import svm, datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix
print("Load Successful")




data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

from os import listdir
person_01 = os.listdir(r"../Datasets/Train/Hrittik")
person_02 = os.listdir(r"../Datasets/Train/Salman")
person_03 = os.listdir(r"../Datasets/Train/Shahrukh")

dataset_dir = r"../Datasets/preprocess"
image_size = 224
labels = []
dataset = []

def create_dataset(image_category, label):
    for img in tqdm(image_category):
        image_path = os.path.join(dataset_dir, img)
        try:
            image = cv2.imread(image_path,cv2.IMREAD_COLOR)
            image = cv2.resize(image,(image_size,image_size))
            for i in range(20):
                augmented_image = data_augmentation(image).numpy().astype("float32")
                dataset.append([np.array(augmented_image),np.array(label)])
        except:
            continue
        
#         dataset.append([np.array(augmented_image),np.array(label)])
    random.shuffle(dataset)
    return dataset

dataset = create_dataset(person_01,1)
dataset = create_dataset(person_02,2)
dataset = create_dataset(person_03,3)
print(len(dataset))


X = np.array([i[0] for i in dataset]).reshape(-1,image_size,image_size,3)
y = np.array([i[1] for i in dataset])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)




from tensorflow.keras.utils import to_categorical
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)



print((X_train.shape,y_train1.shape))
print((X_test.shape,y_test1.shape))


from keras.applications.vgg16 import VGG16, preprocess_input
vgg16_weight_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg = VGG16(
    weights=vgg16_weight_path,
    include_top=False, 
    input_shape=(224,224,3)
)


for layer in vgg.layers:
    layer.trainable = False


from tensorflow.keras import Sequential
from keras import layers
from tensorflow.keras.layers import Flatten,Dense
model = Sequential()
model.add(vgg)
model.add(Dense(256, activation='relu'))
model.add(layers.Dropout(rate=0.5))
model.add(Dense(128, activation='sigmoid'))
model.add(layers.Dropout(rate=0.2))
model.add(Dense(128, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(Flatten())
model.add(Dense(4,activation="sigmoid"))


model.summary()


model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


r = model.fit(X_train,y_train1,batch_size=32,epochs=10,validation_data=(X_test,y_test1))


model.evaluate(X_test,y_test1)


# # re-size all the images to this
# IMAGE_SIZE = [224, 224]

# train_path = 'Datasets/Train'
# valid_path = 'Datasets/Test'

# # add preprocessing layer to the front of VGG
# vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# # don't train existing weights
# for layer in vgg.layers:
#   layer.trainable = False
  

  
#   # useful for getting number of classes
# folders = glob('Datasets/Train/*')
  

# # our layers - you can add more if you want
# x = Flatten()(vgg.output)
# # x = Dense(1000, activation='relu')(x)
# prediction = Dense(len(folders), activation='softmax')(x)

# # create a model object
# model = Model(inputs=vgg.input, outputs=prediction)

# # view the structure of the model
# model.summary()

# # tell the model what cost and optimization method to use
# model.compile(
#   loss='categorical_crossentropy',
#   optimizer='adam',
#   metrics=['accuracy']
# )


# from keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(rescale = 1./255,
#                                    shear_range = 0.2,
#                                    zoom_range = 0.2,
#                                    horizontal_flip = True)

# test_datagen = ImageDataGenerator(rescale = 1./255)

# training_set = train_datagen.flow_from_directory('Datasets/Train',
#                                                  target_size = (224, 224),
#                                                  batch_size = 32,
#                                                  class_mode = 'categorical')

# test_set = test_datagen.flow_from_directory('Datasets/Test',
#                                             target_size = (224, 224),
#                                             batch_size = 32,
#                                             class_mode = 'categorical')

# '''r=model.fit_generator(training_set,
#                          samples_per_epoch = 8000,
#                          nb_epoch = 5,
#                          validation_data = test_set,
#                          nb_val_samples = 2000)'''

# # fit the model
# r = model.fit_generator(
#   training_set,
#   validation_data=test_set,
#   epochs=50,
#   steps_per_epoch=len(training_set),
#   validation_steps=len(test_set)
# )
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

# import tensorflow as tf

# from keras.models import load_model

model.save(r'facefeatures_new_model.h5')

