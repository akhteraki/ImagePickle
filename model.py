import tensorflow as tf
import tensorflow_hub as hub
import pickle

def create_model():
    IMAGE_SHAPE = (224, 224)
    classifier = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE+(3,))
    ])
    return classifier

def save_model(model, filename):
    with open(filename, 'wb') as model_file:
        pickle.dump(model, model_file)

def load_model(filename):
    with open(filename, 'rb') as model_file:
        model = pickle.load(model_file)
    return model
