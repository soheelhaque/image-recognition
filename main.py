import numpy as np
# import the models for further classification experiments
from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3
)
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions


def initialise_models():
    # init the models
    vgg_model = vgg16.VGG16(weights='imagenet')
    inception_model = inception_v3.InceptionV3(weights='imagenet')
    resnet_model = resnet50.ResNet50(weights='imagenet')
    mobilenet_model = mobilenet.MobileNet(weights='imagenet')
    models = [vgg_model, inception_model, resnet_model, mobilenet_model]
    return models

def load_image(filename):


    # load an image in PIL format
    original = load_img(filename, target_size=(224, 224))
    print('PIL image size', original.size)
    plt.imshow(original)
    plt.show()
    return original


def preprocess_image(image):
    # convert the PIL image to a numpy array
    # IN PIL - image is in (width, height, channel)
    # In Numpy - image is in (height, width, channel)
    numpy_image = img_to_array(image)
    plt.imshow(np.uint8(numpy_image))
    plt.show()
    print('numpy array size', numpy_image.shape)

    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.expand_dims(numpy_image, axis=0)
    print('image batch size', image_batch.shape)
    plt.imshow(np.uint8(image_batch[0]))
    return image_batch


def identify_objects(image_batch, models):
    # prepare the image for the VGG model
    processed_image = vgg16.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = models[0].predict(processed_image)
    # print predictions
    # convert the probabilities to class labels
    # we will get top 5 predictions which is the default
    label_vgg = decode_predictions(predictions)
    # print VGG16 predictions
    for prediction_id in range(len(label_vgg[0])):
        print(label_vgg[0][prediction_id])


if __name__ == "__main__":

    models = initialise_models()

    # assign the image path for the classification experiments
    filename = 'images/cat.jpg'
    image = load_image(filename)

    image_batch = preprocess_image(image)

    identify_objects(image_batch, models)

