# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
import shutil

from keras.models import load_model

#import os

sys.path.append('../')

from keras import Input
from deephunter.coverage import Coverage

#from keras.applications import MobileNet, VGG19, ResNet50
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50

from keras.applications.vgg16 import preprocess_input

import random
import time
import numpy as np
from deephunter.image_queue import ImageInputCorpus, TensorInputCorpus
from deephunter.fuzzone import build_fetch_function

from lib.queue import Seed
from lib.fuzzer import Fuzzer

from deephunter.mutators import Mutators
from keras.utils.generic_utils import CustomObjectScope

from keras.datasets import mnist, cifar10
import copy
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


model_weight_path = {
    'vgg16': "./profile/cifar10/models/vgg16.h5",
    'resnet20': "./profile/cifar10/models/resnet.h5",
    'lenet1': "./profile/mnist/models/lenet1.h5",
    'lenet4': "./profile/mnist/models/lenet4.h5",
    'lenet5': "./profile/mnist/models/lenet5.h5"
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
    'resnet50': imagenet_preprocessing
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
    'nc': 0.75,
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
                 'reshape', 'relu', 'pool', 'concat', 'add', 'res4', 'res5']
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

def get_training_data_converage(indir, fetch_function, coverage_function, model_name):
    data_set_name = indir.split('/')[2]
    if 'mnist' in data_set_name:
        (x_train, train_label), (x_test, test_label) = mnist.load_data()
        x_train = mnist_preprocessing(x_train)
    elif 'cifar' in data_set_name:
        (x_train, train_label), (x_test, test_label) = cifar10.load_data()
        x_train = cifar_preprocessing(x_train)
    else:
        print('Please extend the new train data here!')
    if 'mnist' in data_set_name:
        result = train_label
    else:
        result = []
        for label in train_label:
            result.append(label[0])
    set_result = set(result)
    center_neuron_cov_path = {}
    for index in set_result:
        center_neuron_cov_path[index] = np.array(False)
    if 'cifar' in data_set_name:
        for i in range(x_train.shape[0]//1000):
            if i == (x_train.shape[0]//1000)-1:
                tag_x_train = x_train[i*1000::]
                tag_result = result[i*1000::]
            else:
                tag_x_train = x_train[i * 1000: (i+1) * 1000]
                tag_result = result[i * 1000: (i+1) * 1000]
            tag_set_result = set(tag_result)
            tag_coverage_batches, _ = fetch_function((0, tag_x_train, 0, 0, 0))
            print('----calculation converage start----')
            tag_coverage_list = coverage_function(tag_coverage_batches)
            print('----calculation converage end----')
            for la in tag_set_result:
                tag_list_index = np.where(tag_result == la)[0]
                # print(center_neuron_cov_path[la])
                if not center_neuron_cov_path[la].shape:
                    tag_class_np = np.zeros(tag_coverage_list.shape[1])
                else:
                    tag_class_np = center_neuron_cov_path[la]
                for i in tag_list_index:
                    tag_class_np += tag_coverage_list[i]
                center_neuron_cov_path[la] = tag_class_np
        with open("./profile/cifar10/profiling/"+model_name+"/training_coverage.pickle", "wb") as fp:  # Pickling
            pickle.dump(center_neuron_cov_path, fp, protocol=pickle.HIGHEST_PROTOCOL)

    elif 'mnist' in data_set_name:
        coverage_batches, _ = fetch_function((0, x_train, 0, 0, 0))
        coverage_list = coverage_function(coverage_batches)
        center_neuron_cov_path = {}  # 存放类别与训练数据激活神经元的对应关系
        for la in set_result:
            list_index = np.where(result == la)[0]
            class_np = np.zeros(coverage_list.shape[1])
            print(coverage_list.shape)
            print(list_index)
            for i in list_index:
                class_np += coverage_list[i]
            center_neuron_cov_path[la] = class_np
        with open("./profile/mnist/profiling/" + model_name + "/training_coverage.pickle", "wb") as fp:  # Pickling
            pickle.dump(center_neuron_cov_path, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('Please extend the new train data here!')

    return center_neuron_cov_path





if __name__ == '__main__':

    start_time = time.time()

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    random.seed(time.time())

    parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')

    parser.add_argument('-i', help='input seed directory', default='../test_seeds/mnist_seeds')
    parser.add_argument('-o', help='output directory', default='lenet5_out/random/nc/0')

    parser.add_argument('-model', help="target model to fuzz", choices=['vgg16', 'resnet20', 'mobilenet', 'vgg19',
                                                                        'resnet50', 'lenet1', 'lenet4', 'lenet5'], default='lenet5')
    parser.add_argument('-criteria', help="set the criteria to guide the fuzzing",
                        choices=['nc', 'kmnc', 'nbc', 'snac', 'bknc', 'tknc', 'fann'], default='nc')
    parser.add_argument('-batch_num', help="the number of mutants generated for each seed", type=int, default=20)
    parser.add_argument('-max_iteration', help="maximum number of fuzz iterations", type=int, default=200)
    parser.add_argument('-metric_para', help="set the parameter for different metrics", type=float)
    parser.add_argument('-quantize_test', help="fuzzer for quantization", default=0, type=int)
    # parser.add_argument('-ann_threshold', help="Distance below which we consider something new coverage.", type=float,
    #                     default=1.0)
    # parser.add_argument('-quan_model_dir', help="directory including the quantized models for testing")
    parser.add_argument('-random', help="whether to adopt random testing strategy", type=int, default=1)
    parser.add_argument('-select', help="test selection strategy",
                        choices=['uniform', 'tensorfuzz', 'deeptest', 'prob'], default='prob')

    args = parser.parse_args()
    img_rows, img_cols = 224, 224
    input_shape = (img_rows, img_cols, 3)
    input_tensor = Input(shape=input_shape)

    # Get the layers which will be excluded during the coverage computation
    exclude_layer_list = execlude_layer_dic[args.model]

    # Create the output directory including seed queue and crash dir, it is like AFL
    if os.path.exists(args.o):
        shutil.rmtree(args.o)
    os.makedirs(os.path.join(args.o, 'queue'))
    os.makedirs(os.path.join(args.o, 'crashes'))

    # Load model. For ImageNet, we use the default models from Keras framework.
    # For other models, we load the model from the h5 file.
    model = None
    if args.model == 'mobilenet':
        model = MobileNet(input_tensor=input_tensor)
    elif args.model == 'vgg19':
        model = VGG19(input_tensor=input_tensor)
    elif args.model == 'resnet50':
        model = ResNet50(input_tensor=input_tensor)
    else:
        model = load_model(model_weight_path[args.model])

    # Get the preprocess function based on different dataset
    preprocess = preprocess_dic[args.model]

    # Load the profiling information which is needed by the metrics in DeepGauge
    # load the neurons output of training data to find the upper bound and lower bound of output of neurons
    print(model_profile_path[args.model])
    profile_dict = pickle.load(open(model_profile_path[args.model], 'rb'), encoding='bytes')

    # Load the configuration for the selected metrics.
    if args.metric_para is None:
        cri = metrics_para[args.criteria]
    elif args.criteria == 'nc':
        cri = args.metric_para
    else:
        cri = int(args.metric_para)

    # The coverage computer
    coverage_handler = Coverage(model=model, criteria=args.criteria, k=cri,
                                profiling_dict=profile_dict, exclude_layer=exclude_layer_list)

    # If testing for quantization, we will load the quantized versions
    # fetch_function is to perform the prediction and obtain the outputs of each layers
    fetch_function = build_fetch_function(coverage_handler, preprocess)

    # The function to update coverage
    coverage_function = coverage_handler.update_coverage

    # get the center neuron coverage path of training data
    center_neuron_cov_path = get_training_data_converage(args.i, fetch_function, coverage_function, args.model)




