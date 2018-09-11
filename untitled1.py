# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 02:19:17 2018

@author: keshav
"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import threading
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from keras.preprocessing import image

class MainThread(threading.Thread):
    filename=''
    
    def prediction(self,filename):
        # Initialising the CNN
        classifier = Sequential()
        # Step 1 - Convolution
        classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
        # Step 2 - Pooling
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        # Adding a second convolutional layer
        classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        # Step 3 - Flattening
        classifier.add(Flatten())
        # Step 4 - Full connection
        classifier.add(Dense(units = 128, activation = 'relu'))
        classifier.add(Dense(units = 1, activation = 'sigmoid'))
        # Compiling the CNN
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        # Part 2 - Fitting the CNN to the images

        train_datagen = ImageDataGenerator(rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)
        test_datagen = ImageDataGenerator(rescale = 1./255)
        training_set = train_datagen.flow_from_directory('test_set',
        target_size = (64, 64),
        batch_size = 32,
        class_mode = 'binary')
        test_set = test_datagen.flow_from_directory('training_set',
        target_size = (64, 64),
        batch_size = 32,
        class_mode = 'binary')
        classifier.fit_generator(training_set,
        steps_per_epoch =75,
        epochs =1,
        validation_data = test_set,
        validation_steps = 20)
        print("from prediction")

        test_image = image.load_img(filename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict(test_image)
        #print(result.getfield(dtype='str'))
        #print(result.view())
        print(result.tolist())
        #print(result.count())
        #print(classifier.predict(test_image))
        #print(classifier.predict_proba(test_image))
        #print(classifier.output())
        #target_names[classifier.predict(test_image)]
        if result[0][0] == 1:
        #pd.DataFrame(classifier.predict_proba(test_image), columns=classifier.__class__)
            print("perfect match ")
            #predict = 'dog'
        elif result[0][0]==0:
            print("no match with:")
            #predict = 'cat'
        else:
            print("intermediate match ")
        #predict='mouse'
        #print(predict)
        #print(result)
        #prediction('y.png')
    def run(self):
       
       self.prediction('y.png')
       return
def main():
    
    
    
            
    t=MainThread()
            
                
                
    t.start()
                
    time.sleep(1)
            
        
if __name__ == '__main__':
    main()