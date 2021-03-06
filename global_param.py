import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import utils_new as utils

##############################################################################
###                            GENERAL CONSTANTS                           ###
##############################################################################

classes = ('plane', 'car', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck')

# cifar_early_exit_names = ['exit_1.a', 'exit_1.c', 'exit_1.n1', 'exit_1.n2', 'exit_1.codebook.weight', 
#                           'exit_1.codebook.bias', 'exit_1_fc.weight', 'exit_1_fc.bias', 'exit_2.a', 
#                           'exit_2.c', 'exit_2.n1', 'exit_2.n2', 'exit_2.codebook.weight', 'exit_2.codebook.bias']

cifar_normal_layer_names = [
    'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias', 
    'conv4.weight', 'conv4.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'
]

vgg_normal_layer_names = [
    'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias',
    'conv4.weight', 'conv4.bias', 'conv5.weight', 'conv5.bias', 'conv6.weight', 'conv6.bias',
    'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'
]

vgg19_normal_layer_names = [
    'conv1.weight', 'conv1.bias', 'bn1.weight', 'bn1.bias', 'conv2.weight', 'conv2.bias', 'bn2.weight', 
    'bn2.bias', 'conv3.weight', 'conv3.bias', 'bn3.weight', 'bn3.bias', 'conv4.weight', 'conv4.bias', 
    'bn4.weight', 'bn4.bias', 'conv5.weight', 'conv5.bias', 'bn5.weight', 'bn5.bias', 'conv6.weight', 
    'conv6.bias', 'bn6.weight', 'bn6.bias', 'conv7.weight', 'conv7.bias', 'bn7.weight', 'bn7.bias', 
    'conv8.weight', 'conv8.bias', 'bn8.weight', 'bn8.bias', 'conv9.weight', 'conv9.bias', 'bn9.weight', 
    'bn9.bias', 'conv10.weight', 'conv10.bias', 'bn10.weight', 'bn10.bias', 'conv11.weight', 'conv11.bias', 
    'bn11.weight', 'bn11.bias', 'conv12.weight', 'conv12.bias', 'bn12.weight', 'bn12.bias', 'conv13.weight', 
    'conv13.bias', 'bn13.weight', 'bn13.bias', 'conv14.weight', 'conv14.bias', 'bn14.weight', 'bn14.bias', 
    'conv15.weight', 'conv15.bias', 'bn15.weight', 'bn15.bias', 'conv16.weight', 'conv16.bias', 'bn16.weight', 
    'bn16.bias', 'fc.weight', 'fc.bias'
]

average_activation_train_size = 5000

cpu_freq_levels = {
    4: 806400,      # level 4 out of 12
    8: 1420800,     # level 8 out of 12
    12: 2035200     # level 12 out of 12
}

gpu_freq_levels = {
    2: 216750000,   # level 2 out of 13
    5: 522750000,   # level 5 out of 13
    8: 854250000    # level 8 out of 13
}

cpu_max_freq = 2035200

gpu_max_freq = 1300500000

num_testcase_continuous = 5

num_testcase_periodical = 3

##############################################################################
###                                 METHOD                                 ###
##############################################################################

criterion = nn.CrossEntropyLoss()

def get_cpu_target( cpu_idx, cpu_online=True, min_freq=345600, max_freq=2035200, governer='schdeutil' ):
    return {
        "cpu_idx": cpu_idx,
        "cpu_online": cpu_online,
        "min_freq": min_freq,
        "max_freq": max_freq,
        "governor": governer
    }

def get_gpu_target( min_freq=345600, max_freq=2035200, governer='nvhost_podgov' ):
    return {
        "min_freq": min_freq,
        "max_freq": max_freq,
        "governor": governer
    }

def get_optimizer( params, lr, op_type ):
    if op_type == 'adam':
        return optim.Adam( params, lr=lr )
    elif op_type == 'sgd':
        return optim.SGD( params, lr=lr )
    else:
        print( f'Error: the optimizer type ({op_type}) is not valid. Should be adam or sgd' )
        raise NotImplementedError

def get_dataloader( args, task ):
    if task == 'test':
        return torch.utils.data.DataLoader( cifar_test_dataset, 
                                            batch_size=train_test_batch_size if args.task=='train' else evaluate_test_batch_size, 
                                            shuffle=True, 
                                            num_workers=4)
    elif task == 'train':
        if args.model_name == 'cifar':
            batch_size_dict = {
                "normal": cifar_normal_train_hyper.batch_size, 
                "original": cifar_original_train_hyper.batch_size,
                "exits": cifar_exits_train_hyper.batch_size
            }
            return torch.utils.data.DataLoader( cifar_train_dataset, 
                                                batch_size=batch_size_dict[args.train_mode], 
                                                shuffle=True, 
                                                num_workers=4 )
        elif args.model_name == 'vgg':
            batch_size_dict = {
                "normal": vgg_normal_train_hyper.batch_size, 
                "original": vgg_original_train_hyper.batch_size,
                "exits": vgg_exits_train_hyper.batch_size
            }
            return torch.utils.data.DataLoader( cifar_train_dataset, 
                                                batch_size=batch_size_dict[args.train_mode], 
                                                shuffle=True, 
                                                num_workers=4 )
        elif args.model_name == 'vgg19':
            batch_size_dict = {
                "normal": vgg19_normal_train_hyper.batch_size, 
                "original": vgg19_original_train_hyper.batch_size,
                "exits": vgg19_exits_train_hyper.batch_size
            }
            return torch.utils.data.DataLoader( cifar_train_dataset, 
                                                batch_size=batch_size_dict[args.train_mode], 
                                                shuffle=True, 
                                                num_workers=4 )
        else:
            print( f'Error: dataset name ({args.model_name}) is not valid. Should be cifar or vgg or vgg19' )
            raise NotImplementedError
    else:
        print( f'Error: task ({task}) is not valid. Should be train or test' )
        raise NotImplementedError

def get_hyper( args ):
    if args.model_name == 'cifar':
        if args.train_mode == 'normal':
            return cifar_normal_train_hyper
        elif args.train_mode == 'original':
            return cifar_original_train_hyper
        elif args.train_mode == 'exits':
            return cifar_exits_train_hyper
        else:
            print( f'Error: args.train_mode ({args.train_mode}) is not valid, should be normal, original or exits' )
            raise NotImplementedError
    elif args.model_name == 'vgg':
        if args.train_mode == 'normal':
            return vgg_normal_train_hyper
        elif args.train_mode == 'original':
            return vgg_original_train_hyper
        elif args.train_mode == 'exits':
            return vgg_exits_train_hyper
        else:
            print( f'Error: args.train_mode ({args.train_mode}) is not valid, should be normal, original or exits' )
            raise NotImplementedError
    elif args.model_name == 'vgg19':
        if args.train_mode == 'normal':
            return vgg19_normal_train_hyper
        elif args.train_mode == 'original':
            return vgg19_original_train_hyper
        elif args.train_mode == 'exits':
            return vgg19_exits_train_hyper
        else:
            print( f'Error: args.train_mode ({args.train_mode}) is not valid, should be normal, original or exits' )
            raise NotImplementedError
    else:
        print( f'Error: args.model_name ({args.model_name}) is not valid, should be cifar or vgg or vgg19' )
        raise NotImplementedError

def get_layer_names( args ):
    if args.model_name == 'cifar':
        return cifar_normal_layer_names
    elif args.model_name == 'vgg':
        return vgg_normal_layer_names
    elif args.model_name == 'vgg19':
        return vgg19_normal_layer_names
    else:
        print( f'Error: args.model_name ({args.model_name}) is not valid, should be cifar or vgg or vgg19' )
        raise NotImplementedError

##############################################################################
###                                  INIT                                  ###
##############################################################################

cifar_exits_train_init = utils.Namespace(
    aggregation = 'bof'
)

cifar_exits_eval_init = utils.Namespace(
    aggregation = 'bof',
    activation_threshold_list = [
        6.7, 6.0
    ],
    activation_initial_list = [
        12.0, 8.0
    ],
    beta = 2.86      # between 0 and 1, the higher the more accurate
)

vgg_exits_train_init = utils.Namespace(
    # TODO: fill in the dict. 
    # Update: currently it seems that nothing needs to be filled, so it's okay to leave it blank
)

vgg_exits_eval_init = utils.Namespace(
    # TODO: fill in
    # Update: currently it seems that nothing needs to be filled, so it's okay to leave it blank
)

vgg19_exits_train_init = utils.Namespace(
    # TODO: fill in the dict. 
    # Update: currently it seems that nothing needs to be filled, so it's okay to leave it blank
)

vgg19_exits_eval_init = utils.Namespace(
    # TODO: fill in
    # Update: currently it seems that nothing needs to be filled, so it's okay to leave it blank
)

##############################################################################
###                            HYPER PARAMETERS                            ###
##############################################################################

train_test_batch_size = 100
evaluate_test_batch_size = 1

cifar_normal_train_hyper = utils.Namespace(
    epoch_num = 100,
    learning_rate = 0.0005,
    batch_size = 128,
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

cifar_original_train_hyper = utils.Namespace(
    epoch_num = 100,
    learning_rate = 0.0005,
    batch_size = 128,
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

cifar_exits_train_hyper = utils.Namespace(
    epoch_num = 100,
    learning_rate = 0.001,
    batch_size = 128,
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

vgg_normal_train_hyper = utils.Namespace(
    epoch_num = 100,
    learning_rate = 0.0005,
    batch_size = 128,
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

vgg_original_train_hyper = utils.Namespace(
    epoch_num = 100,
    learning_rate = 0.0005,
    batch_size = 128,
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

vgg_exits_train_hyper = utils.Namespace(
    epoch_num = 100,
    learning_rate = 0.001,
    batch_size = 128,
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

vgg19_normal_train_hyper = utils.Namespace(
    epoch_num = 100,
    learning_rate = 0.0005,
    batch_size = 128,
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

vgg19_original_train_hyper = utils.Namespace(
    epoch_num = 100,
    learning_rate = 0.0005,
    batch_size = 128,
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

vgg19_exits_train_hyper = utils.Namespace(
    epoch_num = 100,
    learning_rate = 0.001,
    batch_size = 128,
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
)

cifar_train_dataset = torchvision.datasets.CIFAR10( root='./data/cifar10/', 
                                                    train=True, 
                                                    transform=cifar_normal_train_hyper.transform, 
                                                    download=True )

cifar_test_dataset = torchvision.datasets.CIFAR10(  root='./data/cifar10/', 
                                                    train=False, 
                                                    transform=cifar_normal_train_hyper.transform, 
                                                    download=False )