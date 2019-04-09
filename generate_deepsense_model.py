import tensorflow as tf
from keras.layers import BatchNormalization, Activation
from keras.layers import Input
from keras.layers import Conv2D, RNN, concatenate, Flatten, Dense, GRU, Dropout, GRUCell, Lambda
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
import generate_fake_data
import numpy as np
import matplotlib.pyplot as plt
from keras import losses
import data_generator
from keras.callbacks import CSVLogger
import csv
from keras.utils import plot_model

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 收集individual conv层需要输入的张量
inputs_list = []
dataset_size = 512
batch_size = 16

def flatten_individual_conv_output(inner):
    inner_shape = inner.get_shape().as_list()
    inner = tf.reshape(tensor=inner, shape=[-1, 1, 1, inner_shape[1] * inner_shape[2] * inner_shape[3]]) # shape = [batch_size, 1, ]
    return inner


def get_individual_conv_model(time_interval_t, sensor_k, dim_of_filters):
    drop_rate = 0.2
    # time_interval_t starts index at 0
    # sensor_k starts index at 0
    # dim_of_filters is a list, contains the second dim of filter in each individual covolution layer, e.g. [2, 2, 2]
    cov1, cov2, cov3 = dim_of_filters

    f = 6;
    d_k = 2;
    # define the dimension of X^(k), as described in paper
    # 这里第一个维度应该是batch_size??
    input_shape = (1, d_k, 2 * f)  # (1, 2, 12)
    # input_shape = (None, d_k, 2 * f)  # (batch_size, 2, 12)

    inputs = Input(name='timeInterval%i_Sensor%i' % (time_interval_t, sensor_k), shape=input_shape,
                   dtype='float32')

    inputs_list.append(inputs)

    # Individual convolution layers 1-3
    inner = Conv2D(64, (2, cov1), padding='same', data_format='channels_first',
                   name='individual_conv1_timeInterval%i_sensor%i' % (time_interval_t, sensor_k),
                   kernel_initializer='he_normal')(inputs)  # TBD(None, 128, 64, 64)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Dropout(drop_rate)(inner)

    inner = Conv2D(64, (1, cov2), padding='same', data_format='channels_first',
                   name='individual_conv2_timeInterval%i_sensor%i' % (time_interval_t, sensor_k),
                   kernel_initializer='he_normal')(inner)  # TBD(None, 32, 16, 256)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Dropout(drop_rate)(inner)

    inner = Conv2D(64, (1, cov3), padding='same', data_format='channels_first',
                   name='individual_conv3_timeInterval%i_sensor%i' % (time_interval_t, sensor_k),
                   kernel_initializer='he_normal')(inner)  # TBD(None, 32, 16, 256)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)

    # Flatten
    inner = Lambda(flatten_individual_conv_output)(inner)
    inner = Dropout(drop_rate)(inner)

    return inner

def flatten_merge_conv_output(inner):
    inner_shape = inner.get_shape().as_list()
    inner = tf.reshape(tensor=inner, shape=[-1, 1, inner_shape[1] * inner_shape[2] * inner_shape[3]])
    return inner

def concat_individual_conv_output(concat_individual_conv_output_input_list):
    individual_conv_0, individual_conv_1 = concat_individual_conv_output_input_list
    inputs = tf.concat([individual_conv_0, individual_conv_1], 2)
    return inputs

def concat_tau(inner):
    # In our data, tau = 30
    # tau = tf.placeholder(tf.float32, shape=(None, 1, 1))
    # t = tf.fill(tf.shape(tau), 30.)
    # 在merge cov层输出的tensor最右边加一列值tau
    t = tf.constant(30., shape=(batch_size, 1, 1))
    inner = tf.concat([inner, t], 2)
    return inner

def get_merge_conv_model(time_interval_t, dim_of_filters):
    # dim_of_filters is a list, contains the second dim of filter in each merge covolution layer, e.g. [2, 2, 2]
    cov4, cov5, cov6 = dim_of_filters
    drop_rate = 0.2

    individual_cov_dim_of_filters = [2, 2, 2]
    individual_conv_0 = get_individual_conv_model(time_interval_t, 0, individual_cov_dim_of_filters)
    individual_conv_1 = get_individual_conv_model(time_interval_t, 1, individual_cov_dim_of_filters)

    concat_individual_conv_output_input_list = [individual_conv_0, individual_conv_1]
    inputs = Lambda(concat_individual_conv_output)(concat_individual_conv_output_input_list)

    # K = 2, since our data have two sensors
    K = 2

    # merge convolution layers 1-3
    inner = Conv2D(64, (K, cov4), padding='same', name='merge_conv1_timeInterval%i' % time_interval_t, data_format='channels_first',
                   kernel_initializer='he_normal')(inputs)  # (None, 64, 2, 24)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Dropout(drop_rate)(inner)

    inner = Conv2D(64, (1, cov5), padding='same', name='merge_conv2_timeInterval%i' % time_interval_t, data_format='channels_first',
                   kernel_initializer='he_normal')(inner)  # TBD(None, 32, 16, 256)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Dropout(drop_rate)(inner)

    inner = Conv2D(64, (1, cov6), padding='same', name='merge_conv13_timeInterval%i' % time_interval_t, data_format='channels_first',
                   kernel_initializer='he_normal')(inner)  # TBD(None, 32, 16, 256)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)

    # Flatten
    inner = Lambda(flatten_merge_conv_output)(inner)

    # concatenate x and time interval tau(t)
    inner = Lambda(concat_tau)(inner)
    inner = Dropout(drop_rate)(inner)

    return inner

def concat_merge_conv_output(merge_conv):
    #input = concatenate(inputs=merge_conv, axis=-1)
    input = tf.concat(merge_conv, 1)
    return input

def reshape_merge_conv_output(inner):
    return tf.reshape(inner, shape=[-1, 1, 40])

'''
input_length: the len of sample before padding
'''
def get_GRU_and_output_layer_model(input_length):
    dropout_rate = 0.3
    dim_of_filters = [2, 2, 2]

    # since our max num of time interval is 20, we need to generate 20 merge convolution layers
    merge_conv = []
    for i in range(20):
        merge_conv.append(get_merge_conv_model(i, dim_of_filters))

    input = Lambda(concat_merge_conv_output)(merge_conv)

    # GRU layer, not specifying input length
    inner = GRU(20, return_sequences=True, kernel_initializer='he_normal', name='gru1')(input)
    inner = Dropout(dropout_rate)(inner)
    inner = BatchNormalization()(inner)
    inner = GRU(20, return_sequences=True, kernel_initializer='he_normal', name='gru2')(inner)  # TBD(None, 32, 512)

    # GRU layer, specifying input length
    # gru_cell_1 = GRUCell(units=20, kernel_initializer='he_normal', name='gru_cell_1')
    # inner = RNN(cell=gru_cell_1, return_sequences=True, input_length=input_length)(input)
    # inner = Dropout(dropout_rate)(inner)
    # inner = BatchNormalization()(inner)
    # gru_cell_2 = GRUCell(units=20, kernel_initializer='he_normal', name='gru_cell_2')
    # inner = RNN(cell=gru_cell_2, return_sequences=True, input_length=input_length)(inner)

    # inner = Lambda(reshape_merge_conv_output)(inner)

    output = Dense(2, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros', name='output_FC',
                   use_bias=True)(inner)  # TBD(None, 32, 64)

    return output



'''
return inputs: a np.array contians 40 3-dim np.arrays
                 e.g. [timeInterval0_Sensor0, timeInterval0_Sensor1, timeInterval1_Sensor0, timeInterval1_Sensor1, ...]
                 timeInterval%i_Sensor%i, where 1st i = range(0, 20), 2nd i = range(0, 1)
'''
def get_data_and_label():
    inputs = []
    labels = []
    for i in range(0, batch_size):
        # generate fake data here
        acc, speed, displacement = generate_fake_data.generate_fake_data()
        # acc和speed数据交错连接
        inputs.append(
            generate_fake_data.merge_f(acc, speed))  # inputs: [acc0, speed0, acc1, speed1, ...] shape = [batch_size, 40, 2, 12]
        labels.append(displacement)
    inputs = np.asarray(inputs)  # shape = [batch_size, 40, 2, 12]
    # 这里错了，应该把dataset_size行40列的数据展开，把1-40列stack起来（从上到下拍扁），得到40个np.array
    inputs = inputs.transpose((1, 0, 2, 3))
    inputs = np.reshape(inputs, newshape=[40, -1, 1, 2, 12])
    inputs = list(inputs)
    labels = np.asarray(labels)
    labels = np.reshape(labels, newshape=[dataset_size, 20, 2])
    yield (inputs, {'output_FC': labels})


if __name__ == '__main__':
    # inputs, labels = get_data_and_label() # inputs' shape = [40, dataset_size, 2, 12]  labels' shape = [dataset_size, 20, 2]
    # inputs, labels = reshape_feature.get_feature_label()

    predictions = get_GRU_and_output_layer_model([])
    model = Model(inputs=inputs_list, outputs=predictions)
    # 画出模型图，保存到文件
    # plot_model(model, to_file='model.png')
    # print(model.summary())
    model.compile(optimizer='rmsprop', loss=losses.logcosh, metrics=['accuracy'], sample_weight_mode = "temporal")

    csv_logger = CSVLogger('training.log', append=True, separator=';')

    # history = model.fit(input_data_list, input_label_list, validation_split=0.33, epochs=150, batch_size=10, verbose=0)  # starts training
    # history = model.fit(inputs, labels, epochs=2, verbose=1, batch_size=dataset_size)  # starts training
    sample_length_dict = {}
    with open('lenList.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            sample_length_dict[row["filename"]] = row["T"]
            line_count += 1
    history = model.fit_generator(data_generator.generate_arrays_from_file(sample_length_dict), epochs=1, steps_per_epoch=int(dataset_size/batch_size), verbose=1, callbacks=[csv_logger])

    model.save('my_model.h5')

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()


