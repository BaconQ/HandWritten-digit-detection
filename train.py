from tensorflow import keras


#load digit training data
(x_train, y_train) , (x_test, y_test) = keras.datasets.mnist.load_data()

#scaling
x_train = x_train / 255
x_test = x_test / 255

#2D array flatten to 1D array
x_train_flatten = x_train.reshape(len(x_train), 28 * 28)
x_test_flatten = x_test.reshape(len(x_test), 28 * 28)

#Model
model = keras.Sequential([
    keras.layers.Dense(100, activation= 'relu'),
    keras.layers.Dense(10, input_shape=(784,), activation= 'sigmoid')
])

model.compile(
    optimizer ='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'],
)

#model fit
model.fit(x_train_flatten, y_train, epochs=5)

#test
model.evaluate(x_test_flatten, y_test)

model.save("digits")
