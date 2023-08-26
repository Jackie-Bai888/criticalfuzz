import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #指定GPU
import pickle

import numpy as np
from keras.models import load_model
from kerassurgeon.operations import delete_channels
from keras.applications.vgg16 import preprocess_input
from keras.datasets import mnist, cifar10
from cifar10vgg import cifar10vgg
import random

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
    'lenet1': "./profile/mnist/profiling/lenet1/t_0.5_p_0.25.pickle",
    'lenet4': "./profile/mnist/profiling/lenet4/t_0.2_p_0.75.pickle",
    'lenet5': "./profile/mnist/profiling/lenet5/t_0.2_p_0.75.pickle",
    'resnet20': "./profile/cifar10/profiling/resnet20/t_0.5_p_0.25.pickle",
    'vgg16': "./profile/cifar10/profiling/vgg16/t_0.5_p_0.25.pickle"
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

def set_model_weight(weights, critical_neuron, label_critical):
    '''set weight of model'''
    w = weights[0]
    b = weights[1]
    for neuron_idx, cri_lev in enumerate(critical_neuron):
        if cri_lev == label_critical:
            # print('neuron_id',neuron_idx)
            w[..., neuron_idx] = np.zeros(w.shape[0:-1])
            b[neuron_idx] = 0
    return [w, b]

def get_correct_data(img, label, model, model_name):
    model_correct_data = {}
    for index, x in enumerate(img):
        ori_predict_x = model.predict(np.array([x, ]))
        ori_classes_x = np.argmax(ori_predict_x, axis=1)
        predict_label = int(ori_classes_x[0])
        correct_label = int(label[index])
        if predict_label == correct_label:
            if predict_label in model_correct_data:
                model_correct_data[predict_label].append(x)
            else:
                model_correct_data[predict_label] = []
                model_correct_data[predict_label].append(x)
    np.save(model_name+'_data.npy', model_correct_data)


def mask_all_critical_neurons(model_name, model):
    data = np.load(model_name + '_data.npy', allow_pickle=True).item()
    total_cri_error_num = 0
    total_non_cri_error_num = 0
    num = 0
    critical_error_dict = {}
    non_critical_error_dict = {}
    for label in data:
        for re in range(10):
            if re == 0:
                critical_error_dict[label] = 0
                non_critical_error_dict[label] = 0
            x_train = random.sample(data[label], 1000)
            critical_neuron = get_critical_neurons_profile(model_name)
            for i in range(2):
                # i==0表示mask不重要的神经元；i==1表示mask重要的神经元
                for layer in model.layers:
                    name = layer.name
                    if all(ex not in name for ex in execlude_layer_dic[model_name]):
                        # print(name)
                        layer_weight = model.get_layer(name).get_weights()
                        if len(layer_weight) == 0:
                            continue
                        else:
                            # print('layer_name', name)
                            layer_critical_neu = critical_neuron[label][name]
                            new_weight = set_model_weight(layer_weight, layer_critical_neu, i)
                            model.get_layer(name).set_weights(new_weight)
                for x in x_train:
                    # ori_predict_x = model.predict(np.array([x,]))
                    # ori_classes_x = np.argmax(ori_predict_x, axis=1)
                    change_predict_x = model.predict(np.array([x, ]))
                    change_classes_x = int(np.argmax(change_predict_x, axis=1))
                    # print(label, change_classes_x)
                    if label != change_classes_x:
                        num += 1
                if i == 0:
                    print('mask non-critical neurons', label, num)
                    total_non_cri_error_num += num
                    non_critical_error_dict[label] += num
                else:
                    print('mask critical neurons', label, num)
                    total_cri_error_num += num
                    critical_error_dict[label] += num
                if model_name == 'vgg16':
                    model = cifar10vgg(model_weight_path[model_name], False).model
                else:
                    model = load_model(model_weight_path[model_name])
                num = 0
    print('-----final result-----')
    for i in range(10):
        print(i, end='\t')
    print('\n')
    for i in range(10):
        print(critical_error_dict[i] / 10, end='\t')
    print('\n')
    for i in range(10):
        print(non_critical_error_dict[i] / 10, end='\t')
    print('\n')
    print('%s total 10000; non cri error num %d; cri error num %d' % (
    model_name, total_non_cri_error_num / 10, total_cri_error_num / 10))


def ver_critical_neurons_error(model_name, model):
    data = np.load(model_name + '_data.npy', allow_pickle=True).item()
    total_cri_error_num = 0
    total_non_cri_error_num = 0
    error_num = 0
    critical_error_dict = {}
    for label in data:
        # for re in range(10):
        x_train = random.sample(data[label], 1000)
        critical_neuron = get_critical_neurons_profile(model_name)
        for layer_i in range(len(model.layers)):
            critical_neuron_num = 0
            for layer in model.layers[0:layer_i]:
                name = layer.name
                if all(ex not in name for ex in execlude_layer_dic[model_name]):
                    # print(name)
                    layer_weight = model.get_layer(name).get_weights()
                    layer_critical_neu = critical_neuron[label][name]
                    for cr_ne in layer_critical_neu:
                        if cr_ne == 1:
                            critical_neuron_num += 1
                    if len(layer_weight) == 0:
                        continue
                    else:
                        # print('layer_name', name)
                        new_weight = set_model_weight(layer_weight, layer_critical_neu, 1)
                        model.get_layer(name).set_weights(new_weight)
            for x in x_train:
                change_predict_x = model.predict(np.array([x, ]))
                change_classes_x = int(np.argmax(change_predict_x, axis=1))
                # print(label, change_classes_x)
                if label != change_classes_x:
                    error_num += 1
            # print(label, layer_i, critical_neuron_num)
            # if re == 0:
            critical_error_dict[critical_neuron_num] = error_num
            # else:
            #     critical_error_dict[critical_neuron_num] += error_num
            error_num = 0
            if model_name == 'vgg16':
                model = cifar10vgg(model_weight_path[model_name], False).model
            else:
                model = load_model(model_weight_path[model_name])
        print('class %s' % label)
        for cri_neu_num, er_num in critical_error_dict.items():
            print(cri_neu_num, er_num)
        critical_error_dict = {}
        print('--------')

if __name__ == '__main__':
    model_name = 'resnet20'
    dataset_name = dataset_dic[model_name]
    # Get the preprocess function based on different dataset
    preprocess = preprocess_dic[model_name]
    x_train, train_label = get_training_data(dataset_name, preprocess)
    x_test, test_label = get_testing_data(dataset_name, preprocess)
    if model_name == 'vgg16':
        model = cifar10vgg(model_weight_path[model_name], False).model
        # ori_model = copy.copy(model)
    else:
        model = load_model(model_weight_path[model_name])
    # print(np.array([x_train[0],]).shape)
    # print(model.summary())
    # ori_predict_x = model.predict(x_test)
    # ori_classes_x = np.argmax(ori_predict_x, axis=1)
    # print(np.sum(ori_classes_x == test_label)/ori_classes_x.shape)
    correct_data = get_correct_data(x_train, train_label, model, model_name)
    ver_critical_neurons_error(model_name, model)


