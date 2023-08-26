# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.backend import set_session
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
from deephunter.dav2_model import load_model as dav2_load
from deephunter.dav1_model import get_model as dav1_load
from deephunter.dav3_model import get_model as dav3_load

from lib.queue import Seed
from lib.fuzzer import Fuzzer

from deephunter.mutators import Mutators
from keras.utils.generic_utils import CustomObjectScope

from cifar10vgg import cifar10vgg
import cv2
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

def dav2_preprocessing(x_test):
    temp = np.copy(x_test)
    new_temp = []
    for i in temp:
        new_temp.append(cv2.resize(i, (200, 66)))
    new_temp = np.array(new_temp)
    new_temp = new_temp.astype('float32')
    return new_temp

def dav1_preprocessing(x_test):
    temp = np.copy(x_test)
    new_temp = []
    for i in temp:
        new_temp.append(cv2.resize(i, (128, 128)))
    new_temp = np.array(new_temp)
    new_temp = new_temp.astype('float32')
    return new_temp

def dav3_preprocessing(x_test):
    temp = np.copy(x_test)
    new_temp = []
    for i in temp:
        new_temp.append(cv2.resize(i, (128, 32)))
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
    # 'dav2': "/common/hahabai/code/dav2/weights.hdf5"
}

critical_profile_path = {
    'lenet1': "./profile/mnist/profiling/lenet1/t_0.5_p_0.75.pickle",
    'lenet4': "./profile/mnist/profiling/lenet4/t_0.2_p_0.75.pickle",
    'lenet5': "./profile/mnist/profiling/lenet5/t_0.2_p_0.5.pickle",
    'resnet20': "./profile/cifar10/profiling/resnet20/t_0.5_p_0.75.pickle",
    'vgg16': "./profile/cifar10/profiling/vgg16/t_0.5_p_0.75.pickle",
    'dav2': "./profile/dav2/profiling/dav2/t_0.5_p_0.75.pickle",
    'dav1': "./profile/dav1/profiling/dav1/t_0.1_p_0.25.pickle",
    'dav3': "./profile/dav3/profiling/dav3/t_0.1_p_0.5.pickle"
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
    'dav1': dav1_preprocessing,
    'dav3': dav3_preprocessing
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
    'fann': 0.75,
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
    # if len(metadata) == 1:
        # To check whether it is an adversarial sample
    # if abs(metadata - ground_truth)>0.1:
    #     #print('---**only compare predict result and truth result**---')
    #     results.append('')
    if abs(metadata - ground_truth)>0.1:
        #print('---**only compare predict result and truth result**---')
        results.append('')
    '''
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
    '''
    # results records the suffix for the name of the failed tests
    return results

def dict2list(values):
    '''
    make dict to list of one dimension
    :return:
    '''
    ls = []
    for va in values:
        ls.extend(va)
    return ls

def iterate_function(names, critical_neuron, criteria):
    def func(queue, root_seed, parent, mutated_coverage_list, mutated_data_batches, mutated_metadata_list,
             objective_function):

        ref_batches, batches, cl_batches, l0_batches, linf_batches = mutated_data_batches

        successed = False
        bug_found = False
        # For each mutant in the batch, we will check the coverage and whether it is a failed test
        for idx in range(len(mutated_coverage_list)):
            # if criteria == 'fann':
            #     critical_ne_ls = list(critical_neuron[int(parent.ground_truth)].values())[-1]
            # else:
            critical_ne_ls = dict2list(critical_neuron.values())
            # seed_distance = np.sqrt(
            #     np.sum(np.square(critical_ne_ls - mutated_coverage_list[idx])))/len(critical_ne_ls)
            seed_distance = np.sum(np.square(critical_ne_ls - mutated_coverage_list[idx])) / len(critical_ne_ls)
            input = Seed(cl_batches[idx], mutated_coverage_list[idx], root_seed, parent, mutated_metadata_list[idx],
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


def dry_run(indir, fetch_function, coverage_function, queue, critical_neuron, criteria):
    seed_lis = []
    tag_num = 0
    for i in os.listdir(indir):
        if tag_num == 10:
            break
        if 'center' in i:
            seed_lis.append(i)
            tag_num += 1
    # Read each initial seed and analyze the coverage
    for seed_name in seed_lis:
        tf.compat.v1.logging.info("Attempting dry run with '%s'...", seed_name)
        path = os.path.join(indir, seed_name)
        img = cv2.imread(path)#np.load(path)
        # Each seed will contain two images, i.e., the reference image and mutant (see the paper)
        input_batches = [img,]#img[1:2]
        # Predict the mutant and obtain the outputs
        # coverage_batches is the output of internal layers and metadata_batches is the output of the prediction result
        coverage_batches, metadata_batches = fetch_function((0, input_batches, 0, 0, 0))
        # Based on the output, compute the coverage information
        coverage_list = coverage_function(coverage_batches)
        metadata_list = metadata_function(metadata_batches)
        #metadata_list[0][0] is predict class
        # if criteria == 'fann':
        #     critical_ne_ls = list(critical_neuron[int(metadata_list[0][0])].values())[-1]
        # else:
        critical_ne_ls = dict2list(critical_neuron.values())
        # seed_distance = np.sqrt(np.sum(np.square(critical_ne_ls - coverage_list[0])))/len(critical_ne_ls)
        seed_distance = np.sum(np.square(critical_ne_ls - coverage_list[0]))/len(critical_ne_ls)
        # Create a new seed
        input = Seed(0, coverage_list[0], seed_name, None, metadata_list, metadata_list, seed_distance)
        new_img = np.append(input_batches, input_batches, axis=0)
        # Put the seed in the queue and save the npy file in the queue dir
        queue.save_if_interesting(input, new_img, False, True, seed_name)

def out_critical_coverage(queue):
    all_cov = queue.coverage_critical_neuron.keys()
    all_cove_2_num, all_cove_1_num = 0, 0
    for key in all_cov:
        cove_2_num, cove_1_num = 0, 0
        layer_cov = queue.coverage_critical_neuron[key]
        for index, cv in enumerate(layer_cov):
            if cv == 2:
                cove_2_num += 1
            if cv == 1:
                cove_1_num += 1
        all_cove_2_num += cove_2_num
        all_cove_1_num += cove_1_num
        if cove_2_num+cove_1_num==0:
            print('**** %s has %f critical neurons of all networks****' % (
            key, 0))
        else:
            print('**** %s has coveraged %f critical neurons of all networks****' % (key, cove_2_num/(cove_2_num+cove_1_num)))
    print(
        '**** %s has coveraged %f critical neurons of all networks****' % ('all', all_cove_2_num / (all_cove_2_num + all_cove_1_num)))

if __name__ == '__main__':

    start_time = time.time()

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    random.seed(time.time())

    parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')

    parser.add_argument('-i', help='input seed directory', default='/common/hahabai/data/autonomous_driving_dataset/dav1_data/track2data/track2data/IMG/')
    parser.add_argument('-o', help='output directory', default='lenet5_out/random/kmnc/0')

    parser.add_argument('-model', help="target model to fuzz", choices=['vgg16', 'resnet20', 'mobilenet', 'vgg19',
                                                                        'resnet50', 'lenet1', 'lenet4', 'lenet5', 'dav2',
                                                                        'dav1', 'chauffe', 'dav3'], default='dav3')
    parser.add_argument('-criteria', help="set the criteria to guide the fuzzing",
                        choices=['nc', 'kmnc', 'nbc', 'snac', 'bknc', 'tknc', 'fann'], default='nc')
    parser.add_argument('-batch_num', help="the number of mutants generated for each seed", type=int, default=200)
    parser.add_argument('-max_iteration', help="maximum number of fuzz iterations", type=int, default=200000)
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
    elif args.model == 'vgg16':
        model = cifar10vgg(model_weight_path[args.model], False).model
    elif args.model == 'dav2':
        model = dav2_load(model_weight_path[args.model])
    elif args.model == 'dav1':
        model = dav1_load(model_weight_path[args.model])
    elif args.model == 'dav3':
        model = dav3_load(model_weight_path[args.model])

    elif args.model == 'chauffe':
        root_path = '/common/hahabai/code/deepTest/guided/chauffeur_model/'
        cnn_json_path = root_path + "cnn.json"
        cnn_weights_path = root_path + "cnn.weights"
        lstm_json_path = root_path + "lstm.json"
        lstm_weights_path = root_path + "lstm.weights"
        model = ChauffeurModel(
            cnn_json_path,
            cnn_weights_path,
            lstm_json_path,
            lstm_weights_path)
    else:
        model = load_model(model_weight_path[args.model])

    # Get the preprocess function based on different dataset
    preprocess = preprocess_dic[args.model]

    # Load the profiling information which is needed by the metrics in DeepGauge
    # load the neurons output of training data to find the upper bound and lower bound of output of neurons
    # profile_dict = pickle.load(open(model_profile_path[args.model], 'rb'), encoding='bytes')

    # Load the configuration for the selected metrics.
    if args.metric_para is None:
        cri = metrics_para[args.criteria]
    elif args.criteria == 'nc' or args.criteria == 'fann':
        cri = args.metric_para
    else:
        cri = int(args.metric_para)

    # The coverage computer
    coverage_handler = Coverage(model=model, criteria=args.criteria, k=cri,
                                profiling_dict=None, exclude_layer=exclude_layer_list)

    # The log file which records the plot data after each iteration of the fuzzing
    plot_file = open(os.path.join(args.o, 'plot.log'), 'a+')

    # If testing for quantization, we will load the quantized versions
    # fetch_function is to perform the prediction and obtain the outputs of each layers
    if args.quantize_test == 1:
        model_names = os.listdir(args.quan_model_dir)
        model_paths = [os.path.join(args.quan_model_dir, name) for name in model_names]
        if args.model == 'mobilenet':
            import keras

            with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
                                    'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
                models = [load_model(m) for m in model_paths]
        else:
            models = [load_model(m) for m in model_paths]
        fetch_function = build_fetch_function(coverage_handler, preprocess, models)
        model_names.insert(0, args.model)
    else:
        fetch_function = build_fetch_function(coverage_handler, preprocess)
        model_names = [args.model]

    # Like AFL, dry_run will run all initial seeds and keep all initial seeds in the seed queue
    dry_run_fetch = build_fetch_function(coverage_handler, preprocess)
    # The function to update coverage
    coverage_function = coverage_handler.update_coverage

    # The function to perform the mutation from one seed
    mutation_function = image_mutation_function(args.batch_num)

    # get the center neuron coverage path of training data
    # critical_neuron = pickle.load(open(critical_profile_path[args.model], 'rb'), encoding='bytes')
    # The seed queue
    if args.criteria == 'fann':
        queue = TensorInputCorpus(args.o, args.random, args.select, coverage_handler.total_size, 1.0, "kdtree")
    else:
        queue = ImageInputCorpus(args.o, args.random, args.select, coverage_handler.total_size, args.criteria)

    # Perform the dry_run process from the initial seeds.
    dry_run(args.i, dry_run_fetch, coverage_function, queue, args.criteria)
    out_critical_coverage(queue)
    # For each seed, compute the coverage and check whether it is a "bug", i.e., adversarial example
    image_iterate_function = iterate_function(model_names, args.criteria)

    # The main fuzzer class
    fuzzer = Fuzzer(queue, coverage_function, metadata_function, objective_function, mutation_function, fetch_function,
                    image_iterate_function, args.select)

    # The fuzzing process
    fuzzer.loop(args.max_iteration)
    # out_critical_coverage(queue)

    # f = open('process_time.txt', 'a')
    # print(str(args.model) + ',' + str(args.criteria) + 'finish', time.time() - start_time, file=f)


