# Python 3.5.5
# tensorflow 1.10.0
# Keras 2.2.2

import os
import os.path as path
import sys

import keras
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from numpy.random import rand

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework import graph_util

MODEL_NAME = 'example'
EPOCHS = 1
BATCH_SIZE = 5


MODEL_JSON = 'model_'+MODEL_NAME+'.json'
WEIGHTS = 'weights_'+MODEL_NAME+'.h5'

CHKP = 'out/' + MODEL_NAME + '.chkp'
GRAPH_PBTXT = 'out/' + MODEL_NAME + '_graph.pbtxt'
FROZEN_PB = 'out/frozen_' + MODEL_NAME + '.pb'
OPTIMIZED_PB = 'out/opt_' + MODEL_NAME + '.pb'

def load_data():
    # Really shitty training and test data, just placeholder
    x_train = rand(10,160,160,3).astype('float32')
    y_train = rand(10,  5,  5,8).astype('float32')
    x_test  = rand(10,160,160,3).astype('float32')
    y_test  = rand(10,  5,  5,8).astype('float32')

    return x_train, y_train, x_test, y_test



def build_model():
    print('Building model...')

    input_shape=(160,160,3)
    padding = 'same'

    inputs = Input(shape=input_shape)
    x = inputs
    x = Conv2D(64, (3, 3), strides=(2, 2), padding=padding, name='conv1',activation='relu', kernel_initializer='he_normal', input_shape=input_shape)(x)
    x = BatchNormalization(axis=1, name='conv1_bn')(x)
    x = Conv2D(32, (3, 3), strides=(2, 2), padding=padding, name='conv2',activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=1, name='conv2_bn')(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding=padding, name='conv3',activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=1, name='conv3_bn')(x)
    x = Conv2D(256, (3, 3), strides=(2, 2), padding=padding, name='conv4',activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=1, name='conv4_bn')(x)
    x = Conv2D(8, (3, 3), strides=(2, 2), padding=padding, name='conv5',activation='relu', kernel_initializer='he_normal')(x)
    outputs = x

    model = Model(inputs=inputs, outputs=outputs)
    return model



def train(model, x_train, y_train, x_test, y_test):
    print('Training...')
    
    model.compile(loss=keras.losses.categorical_crossentropy, \
                  optimizer=keras.optimizers.Adadelta(), \
                  metrics=['accuracy'])

    model.fit(x_train, y_train, \
              batch_size=BATCH_SIZE, \
              epochs=EPOCHS, \
              verbose=1, \
              validation_data=(x_test, y_test))


def save_trained(model):
    print('Saving...')

    with open(MODEL_JSON, 'w') as file:
        file.write(model.to_json())
    model.save_weights(WEIGHTS)
    
    print('Saved model and weights')


def export_model(saver, model, input_node_names, output_node_name):
    print('Exporting...')
    tf.train.write_graph(K.get_session().graph_def, '.', GRAPH_PBTXT)

    saver.save(K.get_session(), CHKP)

    freeze_graph.freeze_graph(GRAPH_PBTXT, None, False, CHKP, output_node_name, "save/restore_all", "save/Const:0", FROZEN_PB, True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(FROZEN_PB, "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile(OPTIMIZED_PB, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")
    print('input_names:')
    print(input_node_names)
    print('output_name:')
    print(output_node_name)



def main():
    if not path.exists('out'):
        os.mkdir('out')

    do_train = False;
    do_export = False;

    for arg in sys.argv[1:]:
        if arg == 'train':
            do_train = True
        if arg == 'export':
            do_export = True

    if do_train:
        x_train, y_train, x_test, y_test = load_data()

        model = build_model()

        from tensorflow.keras.utils import plot_model
        plot_model(model, to_file='plotmodel_'+MODEL_NAME+'.png', show_shapes=True)

        train(model, x_train, y_train, x_test, y_test)

        save_trained(model)


    if do_export:
        K.set_learning_phase(0)
        with open(MODEL_JSON, "r") as file:
            config = file.read()

        model = model_from_json(config)
        model.load_weights(WEIGHTS)

        input_names = [model.input.name.split(':')[0]]
        output_name = model.output.name.split(':')[0]

        export_model(tf.train.Saver(), model, input_names, output_name)


if __name__ == '__main__':
    main()
