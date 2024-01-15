import tensorflow as tf
import matplotlib.pyplot as plt

# Check TensorFlow version
print(tf.__version__)

# Load Fashion MNIST dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess training and test images
training_images = training_images.reshape(60000, 28, 28, 1) / 255.0
test_images = test_images.reshape(10000, 28, 28, 1) / 255.0

# Build the convolutional neural network (CNN) model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Train the model
model.fit(training_images, training_labels, epochs=5)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('Test loss: {}, Test accuracy: {:.2f}%'.format(test_loss, test_accuracy * 100))

# Display the true labels of the first 100 test images
print(test_labels[:100])

# Visualize convolutional outputs for specific images
f, axarr = plt.subplots(3, 4)
FIRST_IMAGE = 0
SECOND_IMAGE = 23
THIRD_IMAGE = 28
CONVOLUTION_NUMBER = 6

# Create an activation model for visualization
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

# Iterate through different layers and display feature maps
for x in range(4):
    f1, f2, f3 = (
        activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x],
        activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x],
        activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
    )
    axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[0, x].grid(False)
    axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[1, x].grid(False)
    axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[2, x].grid(False)

# Display the visualizations
plt.show()
