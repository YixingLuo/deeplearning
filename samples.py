import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat',
             'Sandal','Shirt','Sneaker','Bag','Ankle boot']
print("The shape of train_images is ",train_images.shape)
print("The shape of train_labels is ",train_labels.shape)
print("The shape of test_images is ",test_images.shape)
print("The length of test_labels is ",len(test_labels))
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)
plt.show()

train_images=train_images/255.0
test_images=test_images/255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

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

model.fit(train_images,train_labels,epochs=20)


model = keras.models.load_model('model-keras.h5')
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Acc:',test_acc)
"""
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=20)
test_loss,test_acc=model.evaluate(test_images,test_labels)
print('Test Acc:',test_acc)"""

predictions=model.predict(test_images)
print("The first picture's prediction is:{},so the result is:{}".format(predictions[0],np.argmax(predictions[0])))
print("The first picture is ",test_labels[0])
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label],
                                class_names[true_label]),
               color=color)
plt.show()

img = test_images[0]
print(img.shape)

img = (np.expand_dims(img, 0))
print(img.shape)

predictions = model.predict(img)
print(predictions)

prediction = predictions[0]
print("You will find that the ans is same to the former res", np.argmax(prediction))
