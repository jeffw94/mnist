import numpy as np
import pandas as pd 
from keras.utils import np_utils
np.random.seed(10)
from keras.models import Sequential
from keras.layers import Dense

from keras.datasets import mnist

(x_train_image, y_train_label), \
(x_test_image, y_test_label) = mnist.load_data()


import matplotlib.pyplot as plt
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(4, 4)
    plt.imshow(image, cmap='binary')
    plt.show()


print('train data=', len(x_train_image))
print('test data=',len(x_test_image))
print('x_train_image:', x_train_image.shape)
print('y_train_label:', y_train_label.shape)

plot_image(x_train_image[0])
print(y_train_label[0])


#import matplotlib.pyplot as plt
def plot_image_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num > 25: num = 25
    
    for i in range(0,num):
        ax = plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title = "label=" + str(labels[idx])
        if len(prediction) > 0:
            title+="predict="+str(prediction[idx])

        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        idx+=1
    plt.show()

plot_image_labels_prediction(x_train_image, y_train_label,[],0,10)

print('x_test_image:', x_test_image.shape)
print('y_test_label:', y_test_label.shape)

plot_image_labels_prediction(x_test_image, y_test_label,[],0,10)

print('x_train_image:', x_train_image.shape)
print('y_train_label:', y_train_label.shape)

x_Train = x_train_image.reshape(60000, 784).astype('float')
x_Test = x_test_image.reshape(10000, 784).astype('float')

print('x_train:',x_Train.shape)
print('x_test:', x_Test.shape)

print(x_train_image[0])

x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255

print(x_Train_normalize[0])

y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)

y_TrainOneHot[:5]


#-------------------------------------------------pg.78
model = Sequential()
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x=x_Train_normalize, 
                            y = y_TrainOneHot, 
                            validation_split=0.2,
                            epochs=10,
                            batch_size=200,
                            verbose=2)

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Traiin History')
    plt.ylabel('train')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

scores = model.evaluate(x_Test_normalize, y_TestOneHot)
print()
print('accuracy=', scores[1])

prediction = model.predict_classes(x_Test)

plot_image_labels_prediction(x_test_image, y_test_label, prediction, idx=340)