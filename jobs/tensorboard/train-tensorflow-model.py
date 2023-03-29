# import dependencies
import tensorflow as tf
import datetime

# load your dataset and split it (here the basic MNIST dataset)
def load_dataset():

    # load data with Keras
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # scale images to the [0, 1] range
    X_train = X_train.astype("float32")/255
    X_test = X_test.astype("float32")/255

    # images must have shape (28,28,1)
    X_train = X_train.reshape(-1 ,28 ,28 ,1)
    X_test = X_test.reshape(-1 ,28 ,28 ,1)

    # display test and train data size
    print("X_train shape: ", X_train.shape) # number of train images, size
    print("X_test shape: ", X_test.shape)   # number of test images, size
    print("y_train shape: ", y_train.shape) # number of train labels (= number of train images)
    print("y_test shape: ", y_test.shape)   # number of test labels (= number of test images)

    return X_train, y_train, X_test, y_test

# build the ML model architecture
def build_model():

    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# main
if __name__ == '__main__':

    X_train, y_train, X_test, y_test = load_dataset()

    # define the labels of the dataset (for the MNIST dataset, 10 labels)
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

    # parameters
    num_classes = len(class_names)
    input_shape = (28,28,1)

    # display the model summary
    model = build_model()
    model.summary()

    # compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # add the TensorBoard callback
    log_dir = "/workspace/runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # take a subset of images (you don't need to display all of them)
    val_images, val_labels = X_test[:32], y_test[:32]

    # launch the training
    _ = model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test), callbacks = [tensorboard_callback])
