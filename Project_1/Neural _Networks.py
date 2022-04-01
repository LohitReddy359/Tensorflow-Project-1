import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

# x_train = tf.convert_to_tensor(x_train)

# Sequential API (Very convenient, not very flexible)
model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu', name='my_layer'),

        layers.Dense(10, activation='softmax'),
    ]
)

# model = keras.Model(inputs=model.inputs, outputs=[model.layers[-2].output])

# model = keras.Model(inputs=model.inputs, outputs=[model.get_layer('my_layer').output])

# model = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

# features = model.predict(x_train)

# for feature in features:
# print(feature.shape)


# Another way:
# model = keras.Sequential()
# model.add(keras.Input(shape=(28*28))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(256, activation='relu'))


# Functional API (A bit more flexible)
inputs = keras.Input(28*28)
x = layers.Dense(512, activation='relu', name='first_layer')(inputs)
x = layers.Dense(256, activation='relu', name='second_layer')(x)
outputs = layers.Dense(10, activation='softmax')(x)
# model = keras.Model(inputs=inputs, outputs=outputs)

# Lets you get more information about the model itself before doing model.fit
# print(model.summary())  # Common debugging tool also

print(model.summary())


# Specifies the network configurations
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

# Specify the training of the network
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)

# Testing the model
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
