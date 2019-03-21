import foolbox
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

keras.backend.set_learning_phase(0)
kmodel = keras.models.load_model('model-keras.h5')
fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255),preprocess_fn=preprocess_input)


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.reshape([-1,28,28,1]) / 255.0
test_images = test_images.reshape([-1,28,28,1]) / 255.0


#image, label = foolbox.utils.imagenet_example()
attack  = foolbox.attacks.FGSM(fmodel)
adversarial = attack(test_images, test_labels)

plt.subplot(1, 2, 1)
plt.imshow(test_images)

plt.subplot(1, 3, 2)
plt.imshow(adversarial)

plt.subplot(1, 3, 3)
plt.imshow(adversarial - test_images)