import copy
import pickle

import numpy as np
from keras.models import load_model
from kerassurgeon.operations import delete_channels
from keras.applications.vgg16 import preprocess_input
from keras.datasets import mnist, cifar10
from cifar10vgg import cifar10vgg

def imagenet_preprocessing(input_img_data):
    temp = np.copy(input_img_data)
    temp = np.float32(temp)
    qq = preprocess_input(temp)
    return qq


def mnist_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(temp.shape[0], 28, 28, 1)
    temp = temp.astype('float32')
    temp /= 255
    return temp


def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp

critical_profile_path = {
    'lenet1': "./profile/mnist/profiling/lenet1/t_0.5_p_0.75.pickle",
    'lenet4': "./profile/mnist/profiling/lenet4/t_0.2_p_0.25.pickle",
    'lenet5': "./profile/mnist/profiling/lenet5/t_0.5_p_0.75.pickle",
    'resnet20': "./profile/cifar10/profiling/resnet20/t_0.5_p_0.75.pickle",
    'vgg16': "./profile/cifar10/profiling/vgg16/t_0.5_p_0.75.pickle"
}

model_weight_path = {
    'vgg16': "./profile/cifar10/models/cifar10vgg.h5",
    'resnet20': "./profile/cifar10/models/resnet.h5",
    'lenet1': "./profile/mnist/models/lenet1.h5",
    'lenet4': "./profile/mnist/models/lenet4.h5",
    'lenet5': "./profile/mnist/models/lenet5.h5"
}

preprocess_dic = {
    'vgg16': cifar_preprocessing,
    'resnet20': cifar_preprocessing,
    'lenet1': mnist_preprocessing,
    'lenet4': mnist_preprocessing,
    'lenet5': mnist_preprocessing,
    'mobilenet': imagenet_preprocessing,
    'vgg19': imagenet_preprocessing,
    'resnet50': imagenet_preprocessing
}

dataset_dic = {
    'vgg16': 'cifar',
    'resnet20': 'cifar',
    'lenet1': 'mnist',
    'lenet4': 'mnist',
    'lenet5': 'mnist',
    'mobilenet': 'imagenet',
    'vgg19': 'imagenet',
    'resnet50': 'imagenet'
}


execlude_layer_dic = {
    'vgg16': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'resnet20': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet1': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet4': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet5': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'mobilenet': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout',
                  'bn', 'reshape', 'relu', 'pool', 'concat', 'softmax', 'fc'],
    'vgg19': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout', 'bn',
              'reshape', 'relu', 'pool', 'concat', 'softmax', 'fc'],
    'resnet50': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout', 'bn',
                 'reshape', 'relu', 'pool', 'concat', 'add', 'res4', 'res5']
}

def get_training_data(data_name, preprocess):
    if 'mnist' in data_name:
        (x_train, train_label), (_, _) = mnist.load_data()
    elif 'cifar' in data_name:
        (x_train, train_label), (_, _) = cifar10.load_data()
        train_label = train_label[:, 0]
    else:
        print('Please extend the new train data here!')
    x_train = preprocess(x_train)
    return x_train, train_label

def get_testing_data(data_name, preprocess):
    if 'mnist' in data_name:
        (x_train, train_label), (x_test, test_label) = mnist.load_data()
    elif 'cifar' in data_name:
        (x_train, train_label), (x_test, test_label) = cifar10.load_data()
        test_label = test_label[:, 0]
    else:
        print('Please extend the new train data here!')
    x_test = preprocess(x_test)
    return x_test, test_label

def get_critical_neurons_profile(model_name):
    critical_neuron = pickle.load(open(critical_profile_path[model_name], 'rb'), encoding='bytes')
    # print(critical_neuron[0].keys())
    # for va in critical_neuron[0].values():
    #     print(va)
    #critical_ne_ls = dict2list(critical_neuron[int(metadata_list[0][0])].values())
    return critical_neuron

def delete_neurons(model, layer):
    model = delete_channels(model, layer, [3,4])
    return model

def set_model_weight(weights, critical_neuron):
    '''set weight of model'''
    w = weights[0]
    b = weights[1]
    for neuron_idx, cri_lev in enumerate(critical_neuron):
        if cri_lev == 1:
            # print('neuron_id',neuron_idx)
            w[..., neuron_idx] = np.zeros(w.shape[0:-1])
            b[neuron_idx] = 0
    return [w, b]


if __name__ == '__main__':
    model_name = 'lenet4'
    if model_name == 'vgg16':
        model = cifar10vgg(model_weight_path[model_name], False).model
        # ori_model = copy.copy(model)
    else:
        model = load_model(model_weight_path[model_name])
    critical_neuron = get_critical_neurons_profile(model_name)
    total_critical_neuron_dict = critical_neuron[0]
    for cl_name, layer_neurons in critical_neuron.items():
        cl_critical_num = 0
        cl_all_num = 0
        for layer_name, critical_ls in layer_neurons.items():
            if all(ex not in layer_name for ex in execlude_layer_dic[model_name]):
                for num in critical_ls:
                    if num == 1:
                        cl_critical_num += 1
                cl_all_num += len(critical_ls)
            total_critical_neuron_ls = total_critical_neuron_dict[layer_name]
            total_critical_neuron_ls = [cri_i + total_critical_neuron_ls[index] for index, cri_i in enumerate(critical_ls)]
            total_critical_neuron_dict[layer_name] = total_critical_neuron_ls
        print('class %d, all_neurons %d, critical_neurons %d' % (cl_name, cl_all_num, cl_critical_num))
    total_critical_num = 0
    total_neurons_num = 0
    for layer, layer_cri in total_critical_neuron_dict.items():
        for tag in layer_cri:
            if tag > 0:
                total_critical_num += 1
            total_neurons_num += 1

    print('model %s, total_neurons_num %d, total_critical_num %d' % (model_name, total_neurons_num, total_critical_num))