import tensorflow as tf
import numpy as np


def main():
    W, b = [], []
    model = tf_simple()
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if weights:
            W.append(weights[0])
            b.append(weights[1])

    np.save('weights.npy', np.array(W))
    np.save('biases.npy', np.array(b))


def tf_simple():

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([

        # input layer with 784 nodes
        tf.keras.layers.Flatten(input_shape=(28, 28)),

        # densely connected layer witht 256 hidden units
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        # tf.keras.layers.Dropout(0.2),

        # densely connected layer with 10 hidden units
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)

    return model


if __name__ == '__main__':
    main()
