# TensorFlow and tf.keras

import tensorflow as tf
from tensorflow import keras



# Helper libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import matplotlib.pyplot as plt

EAGER = True



fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images.shape,train_labels.shape)
print(test_images.shape,test_labels.shape)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

train_images = train_images.reshape([-1,28,28,1]) / 255.0
test_images = test_images.reshape([-1,28,28,1]) / 255.0
class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat',
             'Sandal','Shirt','Sneaker','Bag','Ankle boot']




model = keras.Sequential([
    #(-1,28,28,1)->(-1,28,28,32)
    keras.layers.Conv2D(input_shape=(28, 28, 1),filters=32,kernel_size=5,strides=1,padding='same'),     # Padding method),
    #(-1,28,28,32)->(-1,14,14,32)
    keras.layers.MaxPool2D(pool_size=2,strides=2,padding='same'),
    #(-1,14,14,32)->(-1,14,14,64)
    keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same'),     # Padding method),
    #(-1,14,14,64)->(-1,7,7,64)
    keras.layers.MaxPool2D(pool_size=2,strides=2,padding='same'),
    #(-1,7,7,64)->(-1,7*7*64)
    keras.layers.Flatten(),
    #(-1,7*7*64)->(-1,512)
    keras.layers.Dense(256, activation=tf.nn.relu),
    #(-1,256)->(-1,10)
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

print(model.summary())

lr = 0.0005
epochs = 20
model.compile(optimizer=tf.train.AdamOptimizer(lr),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size = 128, epochs=epochs,validation_data=[test_images[:10000],test_labels[:10000]])

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(np.argmax(model.predict(test_images[:1100]),1),test_labels[:1100])
predictions=model.predict(test_images)
sample=[]
for i in range(len(test_labels)):

    if int(np.argmax(predictions[i])) == int(test_labels[i]):
        sample.append(i)

print(sample[:1000])
print("The first picture's prediction is:{},so the result is:{}".format(predictions[0],np.argmax(predictions[0])))
print("The first picture is ",test_labels[0])

model.save('model_weight.h5')
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

