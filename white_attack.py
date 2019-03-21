import foolbox
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from foolbox.criteria import TargetClassProbability
from foolbox.attacks import LBFGSAttack

def samples(dataset='imagenet', index=0, batchsize=1, shape=(224, 224),
            data_format='channels_last'):
    from PIL import Image

    images, labels = [], []
    basepath = os.path.dirname(__file__)
    samplepath = os.path.join(basepath, 'data')
    files = os.listdir(samplepath)

    for idx in range(index, index + batchsize):
        i = idx % 20

        # get filename and label
        file = [n for n in files if '{}_{:02d}_'.format(dataset, i) in n][0]
        label = int(file.split('.')[0].split('_')[-1])

        # open file
        path = os.path.join(samplepath, file)
        image = Image.open(path)

        if dataset == 'imagenet':
            image = image.resize(shape)

        image = np.asarray(image, dtype=np.float32)

        if dataset != 'mnist' and data_format == 'channels_first':
            image = np.transpose(image, (2, 0, 1))

        images.append(image)
        labels.append(label)

    labels = np.array(labels)
    images = np.stack(images)
    return images, labels


if __name__ == "__main__":
    keras.backend.set_learning_phase(0)
    #kmodel = keras.models.load_model('model-keras.h5')
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    kmodel = keras.models.model_from_json(loaded_model_json)
    preprocessing = (np.array([28,28,1]),1)
    fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 28), preprocessing=preprocessing)
    #image, label = samples(dataset='fashionMNIST', index=0, batchsize=1, shape=(224, 224), data_format='channels_last')
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.reshape([-1, 28, 28, 1]) / 255.0
    test_images = test_images.reshape([-1, 28, 28, 1]) / 255.0
    print(train_images.shape, train_labels.shape)
    print(test_images.shape, test_labels.shape)

#    print(np.argmax(fmodel.predictions(test_images)), int(test_labels))

    """
    # run the attack
    attack = LBFGSAttack(model=fmodel, criterion=TargetClassProbability(781, p=.5))
    adversarial = attack(image, label)

    # show results
    print(np.argmax(fmodel.predictions(adversarial)))
    print(foolbox.utils.softmax(fmodel.predictions(adversarial))[781])
    adversarial_rgb = adversarial[np.newaxis, :, :, :]
    preds = kmodel.predict(preprocess_input(adversarial_rgb.copy()))
    print("Top 5 predictions (adversarial: ", decode_predictions(preds, top=5))
    """

    #run the attack
    attack  = foolbox.attacks.FGSM(model=fmodel,criterion=TargetClassProbability(10, p=.99))
    adversarial = attack(test_images, test_labels)
    print(np.argmax(fmodel.predictions(adversarial)))
    print(foolbox.utils.softmax(fmodel.predictions(adversarial))[10])



    plt.subplot(1, 2, 1)
    plt.imshow(test_images)

    plt.subplot(1, 3, 2)
    plt.imshow(adversarial)

    plt.subplot(1, 3, 3)
    plt.imshow(adversarial - test_images)