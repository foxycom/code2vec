import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_model():
    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model = keras.models.Sequential([
        layers.Dense(10, activation='relu', input_shape=(384,)),
        layers.Dense(10, activation='relu', input_shape=(10,)),
        layers.Dense(2, activation='softmax', input_shape=(10,), )
    ])
    model.compile(optimizer=opt, loss='binary_crossentropy')
    x = tf.ones((1, 384))
    y = model(x)
    print(model.summary())


if __name__ == '__main__':
    create_model()
