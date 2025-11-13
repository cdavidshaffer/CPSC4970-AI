from keras import Input, activations
from keras.callbacks import ModelCheckpoint
from keras.layers import SimpleRNN, Dense
from keras.losses import CategoricalCrossentropy
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
import numpy as np

PRINT_DEBUG = False
PRINT_INFO = True


def debug(*args):
    if PRINT_DEBUG:
        print(*args)


def info(*args):
    if PRINT_INFO:
        print(*args)


def time_delayed(seq, delay):
    features = []
    targets = []
    for target_index in range(delay, len(seq)):
        features.append(seq[target_index - delay:target_index])
        targets.append(seq[target_index])
    return np.array(features), np.array(targets)


def predict_from_seed(seed, model, output_encoder, input_encoder, prediction_count):
    result = seed
    new_seed = seed
    for i in range(prediction_count):
        inp = input_encoder.transform(np.array([list(new_seed)]).reshape(-1, 1))
        inp = np.array([inp])
        p = output_encoder.inverse_transform(model.predict(inp))
        result = result + p[0, 0]
        new_seed = result[-len(seed):]
    return result


def encode_input(input_sequence):
    ohe = OneHotEncoder(sparse=False)
    result = ohe.fit_transform(np.array(input_sequence).reshape(-1, 1))
    return result, ohe


class RecurrentNetworkModel:
    def __init__(self, training_string, time_steps):
        info("Training string length:", len(training_string))
        debug("Training string:", training_string)
        info("Number of distinct letters", len(set(training_string)))
        info("Encoding input...")
        encoded_input, self.input_encoder = encode_input(list(training_string))
        info("Encoded input shape", encoded_input.shape)
        debug("Encoded input:", encoded_input)
        info("Time delaying input...")
        self.X, self.y = time_delayed(encoded_input, time_steps)
        info("X shape:", self.X.shape)
        debug("X:", self.X)
        info("y shape:", self.y.shape)
        debug("y:", self.y)
        self.model = self.create_model(self.X.shape, self.y.shape)

    @staticmethod
    def create_model(input_shape, output_shape):
        model = Sequential(
            [Input(shape=input_shape[1:]),
             SimpleRNN(200, activation=activations.sigmoid, return_sequences=True),
             SimpleRNN(200, activation=activations.sigmoid),
             Dense(output_shape[1], activation=activations.softmax)]
        )
        model.summary()
        model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=['categorical_accuracy'])
        return model

    def fit(self, epochs=2, batch_size=35, prefix=None):
        callbacks = []
        if prefix:
            callback = ModelCheckpoint(prefix+"-{epoch:04d}-{loss:.4f}.hdf5", monitor='loss', save_best_only=True,
                                       verbose=True, mode='min')
            callbacks.append(callback)
        self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    def predict(self, seed, steps):
        return predict_from_seed(seed, self.model, self.input_encoder, self.input_encoder, steps)

    def load_weights(self, filename):
        info("Loading weights from", filename)
        self.model.load_weights(filename)


def main_kafka():
    time_steps = 100
    m = RecurrentNetworkModel(open("data/kafka_english_the_trial.txt.cleaned").read(), time_steps)
    #m.load_weights("kafka_weights/kafka-0002-2.9242.hdf5")
    m.fit(epochs=2, batch_size=5000, prefix='kafka_weights/kafka')
    seed = """someone must have been telling lies about josef k., he knew he had\ndone nothing wrong but, one morning, he was arrested.  every day at"""[:time_steps]
    print("Seed:", seed)
    print(m.predict(seed, 50))


def main_debug():
    m = RecurrentNetworkModel("abcbab", 2)
    m.load_weights("debug_weights/debug-0010-1.0417.hdf5")
    #m.fit(epochs=10, batch_size=2,  prefix='debug_weights/debug')
    print(m.predict("bc", 10))


if __name__ == '__main__':
    main_kafka()
