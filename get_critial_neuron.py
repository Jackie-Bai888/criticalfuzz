# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.compat.v1.Session(config=config))

import sys
#import importlib
#importlib.reload(sys)
#reload(sys)
#sys.setdefaultencoding('utf8')

import argparse, pickle

from keras.models import load_model

#import os

sys.path.append('../')

from keras import Input

#from keras.applications import MobileNet, VGG19, ResNet50
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50

from keras.applications.vgg16 import preprocess_input
from cifar10vgg import cifar10vgg

import random
import time
import numpy as np

from lib.queue import Seed

from deephunter.mutators import Mutators
from deephunter.dav2_model import load_model as dav2_load
from deephunter.dav1_model import get_model as dav1_load
from deephunter.dav3_model import get_model as dav3_load

from keras.datasets import mnist, cifar10
from keras.models import Model
import cv2


#import importlib
#importlib.reload(sys)


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

def dav2_preprocessing(x_test):
    temp = np.copy(x_test)
    new_temp = []
    for i in temp:
        new_temp.append(cv2.resize(i, (200, 66)))
    new_temp = np.array(new_temp)
    new_temp = new_temp.astype('float32')
    return new_temp

model_weight_path = {
    'vgg16': "./profile/cifar10/models/cifar10vgg.h5",
    'resnet20': "./profile/cifar10/models/resnet.h5",
    'lenet1': "./profile/mnist/models/lenet1.h5",
    'lenet4': "./profile/mnist/models/lenet4.h5",
    'lenet5': "./profile/mnist/models/lenet5.h5",
    'dav2': "/common/hahabai/code/dav2/weights.hdf5",
    'dav1': "/common/hahabai/code/dav1/weights/model_basic_weight.hdf5",
    'dav3': "/common/hahabai/code/deepfuzzer/deephunter/model/weights/model_basic_weight.hdf5"
}

model_profile_path = {
    'vgg16': "./profile/cifar10/profiling/vgg16/0_50000.pickle",
    'resnet20': "./profile/cifar10/profiling/resnet20/0_50000.pickle",
    'lenet1': "./profile/mnist/profiling/lenet1/0_60000.pickle",
    'lenet4': "./profile/mnist/profiling/lenet4/0_60000.pickle",
    'lenet5': "./profile/mnist/profiling/lenet5/0_60000.pickle",
    'mobilenet': "./profile/imagenet/profiling/mobilenet_merged.pickle",
    'vgg19': "./profile/imagenet/profiling/vgg19_merged.pickle",
    'resnet50': "./profile/imagenet/profiling/resnet50_merged.pickle"
}

preprocess_dic = {
    'vgg16': cifar_preprocessing,
    'resnet20': cifar_preprocessing,
    'lenet1': mnist_preprocessing,
    'lenet4': mnist_preprocessing,
    'lenet5': mnist_preprocessing,
    'mobilenet': imagenet_preprocessing,
    'vgg19': imagenet_preprocessing,
    'resnet50': imagenet_preprocessing,
    'dav2': dav2_preprocessing,
    'dav1': None,
    'dav3': None
}

dataset_dic = {
    'vgg16': 'cifar',
    'resnet20': 'cifar',
    'lenet1': 'mnist',
    'lenet4': 'mnist',
    'lenet5': 'mnist',
    'mobilenet': 'imagenet',
    'vgg19': 'imagenet',
    'resnet50': 'imagenet',
    'dav2': 'dav2',
    'dav1': 'dav1',
    'dav3': 'dav3'
}

shape_dic = {
    'vgg16': (32, 32, 3),
    'resnet20': (32, 32, 3),
    'lenet1': (28, 28, 1),
    'lenet4': (28, 28, 1),
    'lenet5': (28, 28, 1),
    'mobilenet': (224, 224, 3),
    'vgg19': (224, 224, 3),
    'resnet50': (224, 224, 3)
}
metrics_para = {
    'kmnc': 1000,
    'bknc': 10,
    'tknc': 10,
    'nbc': 10,
    'newnc': 10,
    'nc': 0.2,
    'fann': 1.0,
    'snac': 10
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
                 'reshape', 'relu', 'pool', 'concat', 'add', 'res4', 'res5'],
    'dav2': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'dav1': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'dav3': ['input', 'flatten', 'activation', 'batch', 'dropout']
}


def metadata_function(meta_batches):
    return meta_batches


def image_mutation_function(batch_num):
    # Given a seed, randomly generate a batch of mutants
    def func(seed):
        return Mutators.image_random_mutate(seed, batch_num)

    return func


def objective_function(seed, names):
    metadata = seed.metadata
    ground_truth = seed.ground_truth
    assert (names is not None)
    results = []
    if len(metadata) == 1:
        # To check whether it is an adversarial sample
        if metadata[0] != ground_truth:
            #print('---**only compare predict result and truth result**---')
            results.append('')
    else:
        # To check whether it has different results between original model and quantized model
        # metadata[0] is the result of original model while metadata[1:] is the results of other models.
        # We use the string adv to represent different types;
        # adv = '' means the seed is not an adversarial sample in original model but has a different result in the
        # quantized version.  adv = 'a' means the seed is adversarial sample and has a different results in quan models.
        if metadata[0] == ground_truth:
            adv = ''
        else:
            adv = 'a'
        count = 1
        while count < len(metadata):
            if metadata[count] != metadata[0]:
                #print('---**compare multiple models predict result and truth result**---')
                results.append(names[count] + adv)
            count += 1

    # results records the suffix for the name of the failed tests
    return results


def iterate_function(names, training_center_coverage):
    def func(queue, root_seed, parent, mutated_coverage_list, mutated_data_batches, mutated_metadata_list,
             objective_function):

        ref_batches, batches, cl_batches, l0_batches, linf_batches = mutated_data_batches

        successed = False
        bug_found = False
        # For each mutant in the batch, we will check the coverage and whether it is a failed test
        for idx in range(len(mutated_coverage_list)):
            seed_distance = np.sqrt(
                np.sum(np.square(training_center_coverage[int(parent.ground_truth)] - mutated_coverage_list[idx])))
            input = Seed(cl_batches[idx], mutated_coverage_list[idx], root_seed, parent, mutated_metadata_list[:, idx],
                         parent.ground_truth, seed_distance, l0_batches[idx], linf_batches[idx])

            # The implementation for the isFailedTest() in Algorithm 1 of the paper
            results = objective_function(input, names)

            if len(results) > 0:
                # We have find the failed test and save it in the crash dir.
                for i in results:
                    queue.save_if_interesting(input, batches[idx], True, suffix=i)
                bug_found = True
            else:
                new_img = np.append(ref_batches[idx:idx + 1], batches[idx:idx + 1], axis=0)
                # If it is not a failed test, we will check whether it has a coverage gain
                result = queue.save_if_interesting(input, new_img, False)
                successed = successed or result
        return bug_found, successed

    return func


def dry_run(indir, fetch_function, coverage_function, queue, training_center_coverage):
    seed_lis = os.listdir(indir)
    # Read each initial seed and analyze the coverage
    for seed_name in seed_lis:
        tf.compat.v1.logging.info("Attempting dry run with '%s'...", seed_name)
        path = os.path.join(indir, seed_name)
        img = np.load(path)
        # Each seed will contain two images, i.e., the reference image and mutant (see the paper)
        input_batches = img[1:2]
        # Predict the mutant and obtain the outputs
        # coverage_batches is the output of internal layers and metadata_batches is the output of the prediction result
        coverage_batches, metadata_batches = fetch_function((0, input_batches, 0, 0, 0))
        # Based on the output, compute the coverage information

        coverage_list = coverage_function(coverage_batches)
        metadata_list = metadata_function(metadata_batches)
        seed_distance = np.sqrt(np.sum(np.square(training_center_coverage[int(metadata_list[0][0])] - coverage_list[0])))
        # Create a new seed
        input = Seed(0, coverage_list[0], seed_name, None, metadata_list[0][0], metadata_list[0][0], seed_distance)
        new_img = np.append(input_batches, input_batches, axis=0)
        # Put the seed in the queue and save the npy file in the queue dir
        queue.save_if_interesting(input, new_img, False, True, seed_name)

def scale(layer_outputs, rmax=1, rmin=0):
    '''
    scale the intermediate layer's output between 0 and 1
    :param layer_outputs: the layer's output tensor
    :param rmax: the upper bound of scale
    :param rmin: the lower bound of scale
    :return:
    '''
    divider = (layer_outputs.max() - layer_outputs.min())
    if divider == 0:
        return np.zeros(shape=layer_outputs.shape)
    X_std = (layer_outputs - layer_outputs.min()) / divider
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled

def get_training_data(data_name, preprocess):
    if 'mnist' in data_name:
        (x_train, train_label), (_, _) = mnist.load_data()
        x_train = preprocess(x_train)
        return x_train, train_label
    elif 'cifar' in data_name:
        (x_train, train_label), (_, _) = cifar10.load_data()
        train_label = train_label[:, 0]
        x_train = preprocess(x_train)
        return x_train, train_label
    elif 'dav2' in data_name:
        print('--start get img')
        train_path = '/common/hahabai/data/autonomous_driving_dataset/dav2_data/train/'
        img_ls = []
        for name in os.listdir(train_path)[:6000]:
            img = cv2.imread(os.path.join(train_path,name), 1)
            img_ls.append(img)
        x_train = preprocess(np.array(img_ls))
        print('--finish get img')
        return x_train, None
    elif 'dav1' in data_name:
        train_path = '/common/hahabai/code/dav1/'
        x_train = np.load(train_path+'uda_imgs.npy')
        targets = np.load(train_path+"uda_labels.npy")
        return x_train, targets
    elif 'dav3' in data_name:
        train_path = '/common/hahabai/code/deepfuzzer/deephunter/model/'
        x_train = np.load(train_path+'uda_imgs.npy')
        targets = np.load(train_path+"uda_labels.npy")
        return x_train, targets
    else:
        print('Please extend the new train data here!')
        return None


def get_neuron_coverage(model, exclude_layer_list, inputs, labels, t , p):
    layer_to_compute = []
    for layer in model.layers:
        if all(ex not in layer.name for ex in exclude_layer_list):
            layer_to_compute.append(layer.name)
    # category class label
    set_labels = set(labels)
    class_layer_critical_neurons = {}
    # for index in set_labels:
    #     center_neuron_cov_path[index] = np.array(False)
    # num=0
    for tag_label in set_labels:
        tag_label_index = np.where(tag_label == labels)[0]
        class_input = []
        for index in tag_label_index:
            class_input.append(inputs[index])
        class_input = np.array(class_input)
        layer_critical_neurons = {}
        for layer_name in layer_to_compute:
            layer_model = Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)
            layer_outputs = layer_model.predict(class_input)
            activate_neuron = np.zeros(layer_outputs.shape[-1])
            #layer_outputs包含了多张图片对应的输出
            for layer_output in layer_outputs:
                scaled = scale(layer_output)
                for neuron_idx in range(scaled.shape[-1]):
                    tag = np.mean(scaled[..., neuron_idx])
                    if tag > t:
                        activate_neuron[neuron_idx] += 1
            critical_neuron = np.zeros(layer_outputs.shape[-1])
            for index, activate_number in enumerate(activate_neuron):
                if (activate_number/class_input.shape[0]) > p:
                    print('layer_name', layer_name, index, activate_number/class_input.shape[0])
                    #critical_neuron[index] += 1  #这里不应该是加1，而是直接设置为1
                    critical_neuron[index] = 1  # 这里不应该是加1，而是直接设置为1
            layer_critical_neurons[layer_name] = critical_neuron
        class_layer_critical_neurons[tag_label] = layer_critical_neurons
    return class_layer_critical_neurons

def get_driving_neuron_coverage(model, exclude_layer_list, inputs, t , p):
    layer_to_compute = []
    for layer in model.layers:
        if all(ex not in layer.name for ex in exclude_layer_list):
            layer_to_compute.append(layer.name)
    all_critical_neurons = {}
    # for index in set_labels:
    #     center_neuron_cov_path[index] = np.array(False)
    # num=0
    for layer_name in layer_to_compute:
        layer_model = Model(inputs=model.inputs, outputs=model.get_layer(layer_name).output)
        layer_outputs = layer_model.predict(inputs)
        activate_neuron = np.zeros(layer_outputs.shape[-1])
        act_num = 0
        #layer_outputs包含了多张图片对应的输出
        for layer_output in layer_outputs:
            scaled = scale(layer_output)
            for neuron_idx in range(scaled.shape[-1]):
                tag = np.mean(scaled[..., neuron_idx])
                if tag > t:
                    activate_neuron[neuron_idx] += 1
                    act_num += 1
        print('layer_name activate neuron', layer_name, act_num, layer_outputs.shape[-1], act_num / layer_outputs.shape[-1])
        critical_neuron = np.zeros(layer_outputs.shape[-1])
        criti_num = 0
        for index, activate_number in enumerate(activate_neuron):
            if (activate_number/inputs.shape[0]) > p:
                # print('layer_name', layer_name, index, activate_number/inputs.shape[0])
                #critical_neuron[index] += 1  #这里不应该是加1，而是直接设置为1
                critical_neuron[index] = 1  # 这里不应该是加1，而是直接设置为1
                criti_num +=1
        print('layer_name critical neuron', layer_name, criti_num, layer_outputs.shape[-1],criti_num/layer_outputs.shape[-1])
        all_critical_neurons[layer_name] = critical_neuron
    return all_critical_neurons

def save_critical_neurons(model_name, critical_neurons, pickle_name):
    if model_name == 'vgg16' or model_name == 'resnet20':
        data_name = 'cifar10'
    elif model_name == 'mobilenet' or model_name == 'vgg19' or model_name == 'resnet50':
        data_name = 'imagenet'
    elif model_name == 'dav2':
        data_name = 'dav2'
    elif model_name == 'dav1':
        data_name = 'dav1'
    elif model_name == 'dav3':
        data_name = 'dav3'
    else:
        data_name = 'mnist'
    with open("./profile/"+data_name+"/profiling/" + model_name + "/"+pickle_name+".pickle", "wb") as fp:  # Pickling
        pickle.dump(critical_neurons, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    start_time = time.time()

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    random.seed(time.time())

    parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')

    parser.add_argument('-model', help="target model to fuzz", choices=['vgg16', 'resnet20', 'mobilenet', 'vgg19',
                                                                        'resnet50', 'lenet1', 'lenet4', 'lenet5',
                                                                        'dav2', 'dav1', 'dav3'], default='dav3')
    parser.add_argument('-t', help="threshold of activate neuron", default=0.1, type=float)
    parser.add_argument('-p', help="threshold of critical neuron", default=0.75, type=float)
    parser.add_argument('-name', help="name of output file", default="t_0.1_p_0.75")


    args = parser.parse_args()
    img_rows, img_cols = 224, 224
    input_shape = (img_rows, img_cols, 3)
    input_tensor = Input(shape=input_shape)

    # Get the layers which will be excluded during the coverage computation
    exclude_layer_list = execlude_layer_dic[args.model]

    # Load model. For ImageNet, we use the default models from Keras framework.
    # For other models, we load the model from the h5 file.
    model = None
    if args.model == 'mobilenet':
        model = MobileNet(input_tensor=input_tensor)
    elif args.model == 'vgg19':
        model = VGG19(input_tensor=input_tensor)
    elif args.model == 'resnet50':
        model = ResNet50(input_tensor=input_tensor)
    elif args.model == 'vgg16':
        model = cifar10vgg(model_weight_path[args.model],False).model
    elif args.model == 'dav2':
        model = dav2_load(model_weight_path[args.model])
    elif args.model == 'dav1':
        model = dav1_load(model_weight_path[args.model])
    elif args.model == 'dav3':
        model = dav3_load(model_weight_path[args.model])
    else:
        model = load_model(model_weight_path[args.model])

    #Get dataset name
    dataset_name = dataset_dic[args.model]
    # Get the preprocess function based on different dataset
    preprocess = preprocess_dic[args.model]
    x_train, train_label = get_training_data(dataset_name, preprocess)
    # get the center neuron coverage path of training data
    if args.model == 'dav2' or args.model == 'dav1' or args.model == 'dav3':
        critical_neurons = get_driving_neuron_coverage(model, exclude_layer_list, x_train, args.t, args.p)
    else:
        critical_neurons = get_neuron_coverage(model, exclude_layer_list, x_train, train_label, args.t, args.p)
    # store critical neurons
    save_critical_neurons(args.model, critical_neurons, args.name)



