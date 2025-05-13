
import os
import numpy
import matplotlib.pyplot as plt
import png
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import pandas


def set_environment():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run_dnn_train(
    filename,
    dnn_size,
    label,
):
    # load data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    images = numpy.concatenate((train_images, test_images))
    labels = numpy.concatenate((train_labels, test_labels))

    # permute dataset
    permutation_index = numpy.random.permutation(len(labels))

    permute_images = images[permutation_index]
    permute_labels = labels[permutation_index]

    # split dataset into training, test and validation samples
    new_train_images = permute_images[0:50000]
    new_train_labels = permute_labels[0:50000]

    new_test_images = permute_images[50000:60000]
    new_test_labels = permute_labels[50000:60000]

    new_validation_images = permute_images[60000:]
    new_validation_labels = permute_labels[60000:]

    # invert image colors
    new_train_images = 255 - new_train_images
    new_test_images = 255 - new_test_images
    new_validation_images = 255 - new_validation_images

    # rescale image data
    new_train_images_float = new_train_images.astype('float32') / 255.0
    new_test_images_float = new_test_images.astype('float32') / 255.0
    new_validation_images_float = new_validation_images.astype('float32') / 255.0

    # reshape image data
    new_train_images_float = new_train_images_float.reshape(
        (new_train_images_float.shape[0], new_train_images_float.shape[1]*new_train_images_float.shape[2])
    )
    new_test_images_float = new_test_images_float.reshape(
        (new_test_images_float.shape[0], new_test_images_float.shape[1]*new_test_images_float.shape[2])
    )
    new_validation_images_float = new_validation_images_float.reshape(
        (new_validation_images_float.shape[0], new_validation_images_float.shape[1]*new_validation_images_float.shape[2])
    )

    # create and train DNN
    print(f'layer size: {dnn_size}, label: {label}')

    model = keras.Sequential([
        layers.Dense(dnn_size, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])

    model.compile(
        optimizer='rmsprop',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    with tensorflow.device('/GPU:0'):
        model_history = model.fit(new_train_images_float, new_train_labels, epochs=200, batch_size=512, verbose=0)

    accuracy = model_history.history['accuracy'][0:]
    loss = model_history.history['loss'][0:]

    if os.path.isfile(filename):
        df = pandas.read_csv(filename)
    else:
        df = pandas.DataFrame()

    column_name_accuracy = f'accuracy_{label}'
    column_name_loss = f'loss_{label}'

    df[column_name_accuracy] = accuracy
    df[column_name_loss] = loss

    df.to_csv(filename, index=False)


def plot_accuracy_loss(filename, output_filename):
    df = pandas.read_csv(filename)

    fig, ax = plt.subplots(1, 1)
    ax2 = ax.twinx()
    for column in df.columns:

        if 'accuracy' in column:
            column_number = column.split('_')[1]
            accuracy = df[column]
            ax.plot(accuracy, linestyle='solid', label=f'{column_number}')

        elif 'loss' in column:
            loss = df[column]
            #ax2.plot(loss, linestyle='dotted', label=f'{column}')
            ax2.plot(loss, linestyle='dotted')

        else:
            raise RuntimeError(f'unknown column name {column}')

    #fig.legend()
    #fig.tight_layout()
    #plt.subplots_adjust(right=0.8)

    fig.savefig(output_filename)

