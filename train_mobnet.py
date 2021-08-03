import os, cv2, random
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from random import shuffle 
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import ResNet50, DenseNet121, MobileNet
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.callbacks import Callback
import dvclive
import json
import yaml
class MetricsCallback(Callback):
    def on_epoch_end(self, epoch: int, logs: dict = None):
        logs = logs or {}
        for metric, value in logs.items():
            dvclive.log(metric, value)
        dvclive.next_step()

TEST_SIZE = 0.5
RANDOM_STATE = 2018
BATCH_SIZE = 64
NO_EPOCHS = 2
NUM_CLASSES = 2
SAMPLE_SIZE = 20000
PATH = '/home/shruthi/Shruthi_Tasks/s3_flask'
TRAIN_FOLDER = './train/'
TEST_FOLDER =  './test1/'
IMG_SIZE = 128
params = yaml.safe_load(open("/home/shruthi/Shruthi_Tasks/s3_flask/params.yaml"))["train"]
model_name=params["model"]
train_image_list = os.listdir("./train/")[0:SAMPLE_SIZE]
test_image_list = os.listdir("./test1/")

def label_pet_image_one_hot_encoder(img):
    pet = img.split('.')[-3]
    if pet == 'cat': return [1,0]
    elif pet == 'dog': return [0,1]

def process_data(data_image_list, DATA_FOLDER, isTrain=True):
    data_df = []
    for img in tqdm(data_image_list):
        path = os.path.join(DATA_FOLDER,img)
        if(isTrain):
            label = label_pet_image_one_hot_encoder(img)
        else:
            label = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        data_df.append([np.array(img),np.array(label)])
    shuffle(data_df)
    return data_df

train = process_data(train_image_list, TRAIN_FOLDER)

def show_images(data, isTest=False):
    f, ax = plt.subplots(5,5, figsize=(15,15))
    for i,data in enumerate(data[:25]):
        img_num = data[1]
        img_data = data[0]
        label = np.argmax(img_num)
        if label  == 1: 
            str_label='Dog'
        elif label == 0: 
            str_label='Cat'
        if(isTest):
            str_label="None"
        ax[i//5, i%5].imshow(img_data)
        ax[i//5, i%5].axis('off')
        ax[i//5, i%5].set_title("Label: {}".format(str_label))
    plt.show()

show_images(train)
test = process_data(test_image_list, TEST_FOLDER, False)

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
y = np.array([i[1] for i in train])
if model_name=='MobileNet':
    model = Sequential()
    model.add(MobileNet(include_top=False, pooling='avg', weights='imagenet'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    # ResNet-50 model is already trained, should not be trained
    model.layers[0].trainable = True

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_model = model.fit(X_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=NO_EPOCHS,
                  verbose=1,
                  validation_data=(X_val, y_val),
                  callbacks=[MetricsCallback()])

    model.save("mob_model_catdog.h5")

    score = model.evaluate(X_val, y_val, verbose=0)
    print('Validation loss:', score[0])
    print('Validation accuracy:', score[1])
    with open("score.txt", "w") as fp:
        json.dump(score, fp)

elif model_name=='ResNet50':
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    # ResNet-50 model is already trained, should not be trained
    model.layers[0].trainable = True

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_model = model.fit(X_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=NO_EPOCHS,
                  verbose=1,
                  validation_data=(X_val, y_val),
                  callbacks=[MetricsCallback()])

    model.save("res_model_catdog.h5")

    score = model.evaluate(X_val, y_val, verbose=0)
    print('Validation loss:', score[0])
    print('Validation accuracy:', score[1])
    with open("score_res.txt", "w") as fp:
        json.dump(score, fp)
