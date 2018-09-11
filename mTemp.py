# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.pyplot as plt
import time

from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from keras.preprocessing import image
print("Initialising the Convolutional neural network...")

classifier = Sequential()
print("Convolution...")
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
print("Pooling...")
classifier.add(MaxPooling2D(pool_size = (2, 2)))
print("Adding a second convolutional layer...")
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
print("Pooling...")
classifier.add(MaxPooling2D(pool_size = (2, 2)))
print("Flattening...")
classifier.add(Flatten())
print("Full connection...")
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
print("Compiling the CNN...")
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print("Fitting the CNN to the images...")
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
print("fetching images from test set...")
training_set = train_datagen.flow_from_directory('test_set',
classes=['benign','malignant'],
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
print("fetching images from training set...")
test_set = test_datagen.flow_from_directory('training_set',
classes=['benign','malignant'],
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
label_map=training_set.class_indices;

print("applying generator on training set...: ")
history=classifier.fit_generator(training_set,
steps_per_epoch =350,
epochs =50,
validation_data = test_set,
validation_steps = 20)
def plotgraph():
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('valacc1.png')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('valloss1.png')
def prediction(filename):
    print("starting prediction please wait...")
    time.sleep(3)
    test_image = image.load_img(filename, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    outcome=classifier.predict_classes(test_image)
    print("analyzing outcome of predictions ...")
    time.sleep(3)
    print("preparing result...")
    time.sleep(3)
    print("final outcome: ")
    print(outcome[0][0])
    if outcome[0][0]==0:
        print("benign")
    elif outcome[0][0]==1:
        print("malignant")
    else:
        print("sab golmaal hai bhai sab golmaal hai")
    
plotgraph()
prediction('malignant.png')

