
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential

'''Creating a sequential model and creating 9 hidden layers with neurons(64, 64 and 32) each of Convolution layer
and activation funtion used is Relu with kernel size 3x3'''
model = Sequential()
model.add(Convolution2D(filters=64, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(filters=64,
                        kernel_size=(3,3), 
                        activation='relu',
                       ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


'''Training the model with the data provided, and also generating more data from existing data
by performing minor alterations'''
from keras_preprocessing.image import ImageDataGenerator
import PIL
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'Corona_dataset/chest_xray/chest_xray/train/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'Corona_dataset/chest_xray/chest_xray/test/',
        target_size=(64, 64),
        batch_size=32,
       class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=8000,
        epochs=2,
        validation_data=test_set,
        )

#Testing the model with affected and Unaffected images
from keras.preprocessing import image
test_image = image.load_img('03BF7561-A9BA-4C3C-B8A0-D3E585F73F3C.jpeg', 
               target_size=(64,64))
test_image.shape
import numpy as np 
test_image = np.expand_dims(test_image, axis=0)
test_image.shape
result = model.predict(test_image)
r = training_set.class_indices
r
test_image = image.load_img('39EE8E69-5801-48DE-B6E3-BE7D1BCF3092.jpeg', 
               target_size=(64,64))
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
print(result)

#Save a the trained model with name Chest_Xray_prediction.pk1
model.save("Chest_Xray_prediction.pk1")

#Testing a new image by importing the model (pre-trained weight)
from keras.models import load_model
m = load_model('Chest_Xray_prediction.pk1')
result = m.predict(test_image)
print(result)