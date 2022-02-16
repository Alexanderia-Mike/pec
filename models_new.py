"""
DESCRIPTION:    this file contains all the models as well as the methods to call the models

AUTHOR:         Lou Chenfei

INSTITUTE:      Shanghai Jiao Tong University, UM-SJTU Joint Institute

PROJECT:        ECE4730J Advanced Embedded System Capstone Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from nikolaos.bof_utils import LogisticConvBoF
# import utils
import global_param as gp

'''
a list of models:
    cifar_exits_train:          cifar with early-exits for training
    cifar_exits_eval:           cifar with early-exits for evaluation
    cifar_normal:               cifar without early-exits
    vgg_exits_train:            vgg with early-exits for training
    vgg_exits_eval:             vgg with early-exits for evaluation
    vgg_normal:                 vgg without early-exits
    vgg19_exits_train:          vgg19 with early-exits for training
    vgg19_exits_eval:           vgg19 with early-exits for evaluation
    vgg19_normal:               vgg19 without early-exits
'''


def get_train_model( args ):
    '''
    get the model according to args.model_name and args.train_mode
    '''
    if args.model_name == 'cifar':
        if args.train_mode == 'normal':
            return cifar_normal()
        elif args.train_mode in ['original', 'exits']:
            return cifar_exits_train()
        else:
            print( f'Error: args.train_mode ({args.train_mode}) is not valid. Should be normal, original or exits' )
            raise NotImplementedError
    elif args.model_name == 'vgg':
        if args.train_mode == 'normal':
            return vgg_normal()
        elif args.train_model in ['original', 'exits']:
            return vgg_exits_train()
    elif args.model_name == 'vgg19':
        if args.train_mode == 'normal':
            return vgg19_normal()
        elif args.train_mode in ['original', 'exits']:
            return vgg19_exits_train()
    else:
        print( f'Error: args.model_name ({args.model_name}) is not valid. Should be either cifar or vgg or vgg19' )
        raise NotImplementedError


def get_eval_model( args ):
    '''
    get the model according to args.model_name and args.train_mode
    '''
    if args.model_name == 'cifar':
        return cifar_exits_eval() if args.evaluate_mode == 'exits' else cifar_normal()
    elif args.model_name == 'vgg':
        return vgg_exits_eval() if args.evaluate_mode == 'exits' else vgg_normal()
    elif args.model_name == 'vgg19':
        return vgg19_exits_eval() if args.evaluate_mode == 'exits' else vgg19_normal()
    else:
        print( f'Error: args.model_name ({args.model_name}) is not valid. Should be either cifar or vgg or vgg19' )
        raise NotImplementedError


class cifar_exits_eval( nn.Module ):
    '''
    has two early-exiting options
    '''
    def __init__(self):
        super(cifar_exits_eval, self).__init__()
        init = gp.cifar_exits_eval_init
        self.exit_num = 2
        self.aggregation = init.aggregation
        # Base network
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(5 * 5 * 128, 1024)
        self.fc2 = nn.Linear(1024, 10)
        # Exit layer 1:
        if self.aggregation == 'spatial_bof_1':
            self.exit_1 = LogisticConvBoF(32, 32, split_horizon=6)
            self.exit_1_fc = nn.Linear(4 * 32, 10)
            # Exit layer 2:
            self.exit_2 = LogisticConvBoF(128, 32, split_horizon=2)
        elif self.aggregation == 'spatial_bof_2':
            self.exit_1 = LogisticConvBoF(32, 64, split_horizon=6)
            self.exit_1_fc = nn.Linear(4 * 64, 10)
            # Exit layer 2:
            self.exit_2 = LogisticConvBoF(128, 64, split_horizon=2)
        elif self.aggregation == 'spatial_bof_3':
            self.exit_1 = LogisticConvBoF(32, 256, split_horizon=6)
            self.exit_1_fc = nn.Linear(4 * 256, 10)
            # Exit layer 2:
            self.exit_2 = LogisticConvBoF(128, 256, split_horizon=2)
        elif self.aggregation == 'bof':
            self.exit_1 = LogisticConvBoF(32, 64, split_horizon=14)
            self.exit_1_fc = nn.Linear(64, 10)
            # Exit layer 2:
            self.exit_2 = LogisticConvBoF(128, 64, split_horizon=5)
        # threshold for switching between layers
        self.activation_threshold_1 = 0
        self.activation_threshold_combined = 0
        # the number of early exits
        self.num_early_exit_1 = 0
        self.num_early_exit_3 = 0
        self.original = 0
        # the beta coefficient used for accuracy-speed trade-off, the higher the more accurate
        self.beta = 0
    
    def set_activation_thresholds( self, threshold_list:list ):
        if len( threshold_list ) != self.exit_num:
            print( f'the length of the threshold_list ({len(threshold_list)}) is invalid, should be {self.exit_num}' )
            raise NotImplementedError
        self.activation_threshold_1 = threshold_list[0]
        self.activation_threshold_combined = threshold_list[1]
    
    def set_beta( self, beta ):
        self.beta = beta

    def print_exit_percentage( self ):
        total_inference = self.num_early_exit_1 + self.num_early_exit_3 + self.original
        print( f'early exit 1: {100*self.num_early_exit_1/total_inference:.3f}% ({self.num_early_exit_1}/{total_inference})', end=' | ' )
        print( f'early exit 3: {100*self.num_early_exit_3/total_inference:.3f}% ({self.num_early_exit_3}/{total_inference})', end=' | ' )
        print( f'original: {100*self.original/total_inference:.3f}% ({self.original}/{total_inference})' )
    
    def _calculate_max_activation( self, param ):
        '''
        return the maximum activation item in [param]
        '''
        return torch.max( param )

    def forward( self, x ):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x_exit_1 = self.exit_1(x)
        exit1 = self.exit_1_fc(x_exit_1)
        if self._calculate_max_activation( exit1 ) > self.beta * self.activation_threshold_1:
            self.num_early_exit_1 += 1
            return 0, exit1
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x_exit_2 = self.exit_2(x)
        x_exit_3 = (x_exit_1 + x_exit_2) / 2
        exit3 = self.exit_1_fc(x_exit_3)
        if self._calculate_max_activation( exit3 ) > self.beta * self.activation_threshold_combined:
            self.num_early_exit_3 += 1
            return 1, exit3
        x = x.view(-1, 5 * 5 * 128)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        self.original += 1
        return 2, x


class cifar_exits_train( cifar_exits_eval ):
    '''
    adds functions to specify the exit layers
    should have:
    1. the ability to set_exit_layers
    '''
    def __init__( self ):
        super().__init__()
        self.exit_layer = 'original'
    
    def set_exit_layer(self, exit_layer):
        if exit_layer not in ['original', 'exits']:
            print( f'Error: exit_layer ({exit_layer}) is invalid. Should be original or exits' )
            raise NotImplementedError
        self.exit_layer = exit_layer
    
    # the functions starting from here should be updated by json initializations!
    def forward( self, x ):
        if self.exit_layer == 'original':
            return self.forward_original( x )
        elif self.exit_layer == 'exits':
            return self.forward_exits( x )

    def forward_original( self, x ):
        x = F.relu( self.conv1( x ) )
        x = F.relu( self.conv2( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = F.relu( self.conv3( x ) )
        x = F.relu( self.conv4( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = x.view( -1, 5 * 5 * 128 )
        x = F.relu( self.fc1( x ) )
        x = F.dropout( x, p=0.5, training=self.training )
        x = self.fc2( x )
        return x

    def forward_exits( self, x ):
        x = F.relu( self.conv1( x ) )
        x = F.relu( self.conv2( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x_exit_1 = self.exit_1(x)
        # calculate exit1
        exit1 = self.exit_1_fc(x_exit_1)
        # continue inference
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        # calculate exit3
        x_exit_2 = self.exit_2(x)
        x_exit_3 = (x_exit_1 + x_exit_2) / 2
        exit3 = self.exit_1_fc(x_exit_3)
        return ( exit1, exit3 )


class cifar_normal(nn.Module):
    def __init__(self):
        super(cifar_normal, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(5 * 5 * 128, 1024)
        self.fc2 = nn.Linear(1024, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 128)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x


class vgg_exits_eval( nn.Module ):
    '''
    has two early-exiting options
    '''
    def __init__(self):
        super(vgg_exits_eval, self).__init__()
        self.exit_num = 3
        # Base network
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)         # 32
        self.conv2 = nn.Conv2d(32, 32, 3, 1, padding=1)        # 32
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)        # 16
        self.conv4 = nn.Conv2d(64, 64, 3, 1, padding=1)        # 16
        self.conv5 = nn.Conv2d(64, 128, 3, 1, padding=1)       # 8
        self.conv6 = nn.Conv2d(64, 128, 3, 1, padding=1)       # 8
        self.fc1 = nn.Linear(4 * 4 * 128, 128)
        self.fc2 = nn.Linear(128, 10)
        # early exits
        self.exit_1 = LogisticConvBoF(32, 64, split_horizon=16)
        self.exit_2 = LogisticConvBoF(64, 64, split_horizon=8)
        self.exit_3 = LogisticConvBoF(128, 64, split_horizon=4)
        self.exit_1_fc = nn.Linear(64, 10)
        # threshold for switching between layers
        self.activation_threshold_1 = 0
        self.activation_threshold_2 = 0
        self.activation_threshold_3 = 0
        self.num_early_exit_1 = 0
        self.num_early_exit_2 = 0
        self.num_early_exit_3 = 0
        self.original = 0
        # the beta coefficient used for accuracy-speed trade-off, the higher the more accurate
        self.beta = 0
    
    def set_activation_thresholds( self, threshold_list:list ):
        if len( threshold_list ) != self.exit_num:
            print( f'the length of the threshold_list ({len(threshold_list)}) is invalid, should be {self.exit_num}' )
            raise NotImplementedError
        self.activation_threshold_1 = threshold_list[0]
        self.activation_threshold_2 = threshold_list[1]
        self.activation_threshold_3 = threshold_list[2]

    def print_exit_percentage( self ):
        total_inference = self.num_early_exit_1 + self.num_early_exit_2 + self.num_early_exit_3 + self.original
        print( f'early exit 1: {100*self.num_early_exit_1/total_inference:.3f}% ({self.num_early_exit_1}/{total_inference})', end=' | ' )
        print( f'early exit 2: {100*self.num_early_exit_2/total_inference:.3f}% ({self.num_early_exit_2}/{total_inference})', end=' | ' )
        print( f'early exit 3: {100*self.num_early_exit_3/total_inference:.3f}% ({self.num_early_exit_3}/{total_inference})', end=' | ' )
        print( f'original: {100*self.original/total_inference:.3f}% ({self.original}/{total_inference})' )

    def set_beta( self, beta ):
        self.beta = beta
    
    def _calculate_max_activation( self, param ):
        '''
        return the maximum activation item in [param]
        '''
        return torch.max( param )

    def forward( self, x ):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x_exit_1 = self.exit_1(x)
        exit1 = self.exit_1_fc(x_exit_1)
        if self._calculate_max_activation( 1 ) > self.beta * self.activation_threshold_1:
            self.num_early_exit_1 += 1
            return 0, exit1
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x_exit_2 = self.exit_2(x)
        exit2 = (x_exit_1 + x_exit_2) / 2
        exit2 = self.exit_1_fc(exit2)
        if self._calculate_average_activations( 2 ) > self.beta * self.activation_threshold_2:
            self.num_early_exit_2 += 1
            return 1, exit2
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2, 2)
        x_exit_3 = self.exit_3(x)
        exit3 = (x_exit_1 + x_exit_2 + x_exit_3) / 3
        exit3 = self.exit_1_fc(exit3)
        if self._calculate_average_activations( 3 ) > self.beta * self.activation_threshold_3:
            self.num_early_exit_3 += 1
            return 2, exit3
        x = x.view(-1, 4 * 4 * 128)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        self.original += 1
        return 3, x


class vgg_exits_train( vgg_exits_eval ):
    def __init__( self ):
        super().__init__()
        self.exit_layer = 'original'
    
    def set_exit_layer(self, exit_layer):
        if exit_layer not in ['original', 'exits']:
            print( f'Error: exit_layer ({exit_layer}) is invalid. Should be original or exits' )
            raise NotImplementedError
        self.exit_layer = exit_layer
    
    # the functions starting from here should be updated by json initializations!
    def forward( self, x ):
        if self.exit_layer == 'original':
            return self.forward_original( x )
        elif self.exit_layer == 'exits':
            return self.forward_exits( x )

    def forward_original( self, x ):
        x = F.relu( self.conv1( x ) )
        x = F.relu( self.conv2( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = F.relu( self.conv3( x ) )
        x = F.relu( self.conv4( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = F.relu( self.conv5( x ) )
        x = F.relu( self.conv6( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = x.view( -1, 4 * 4 * 128 )
        x = F.relu( self.fc1( x ) )
        x = F.dropout( x, p=0.5, training=self.training )
        x = self.fc2( x )
        return x

    def forward_exits( self, x ):
        x = F.relu( self.conv1( x ) )
        x = F.relu( self.conv2( x ) )
        x = F.max_pool2d( x, 2, 2 )
        # calculate exit1
        x_exit_1 = self.exit_1(x)
        exit1 = self.exit_1_fc(x_exit_1)
        # continue inference
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        # calculate exit2
        x_exit_2 = self.exit_2(x)
        exit2 = (x_exit_1 + x_exit_2) / 2
        exit2 = self.exit_1_fc(exit2)
        # continue inference
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2, 2)
        # calculate exit3
        x_exit_3 = self.exit_3(x)
        exit3 = (x_exit_1 + x_exit_2 + x_exit_3) / 3
        exit3 = self.exit_1_fc(exit3)
        return ( exit1, exit2, exit3 )
    pass


class vgg_normal( nn.Module ):
    def __init__(self):
        super(vgg_normal, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)         # 32
        self.conv2 = nn.Conv2d(32, 32, 3, 1, padding=1)        # 32
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)        # 16
        self.conv4 = nn.Conv2d(64, 64, 3, 1, padding=1)        # 16
        self.conv5 = nn.Conv2d(64, 128, 3, 1, padding=1)       # 8
        self.conv6 = nn.Conv2d(128, 128, 3, 1, padding=1)      # 8
        self.fc1 = nn.Linear(4 * 4 * 128, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu( self.conv1( x ) )
        x = F.relu( self.conv2( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = F.relu( self.conv3( x ) )
        x = F.relu( self.conv4( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = F.relu( self.conv5( x ) )
        x = F.relu( self.conv6( x ) )
        x = F.max_pool2d( x, 2, 2 )
        x = x.view( -1, 4 * 4 * 128 )
        # debug begin
        # print( f"x's shape is {x.shape}" )
        # debug end
        x = F.relu( self.fc1( x ) )
        x = F.dropout( x, p=0.5, training=self.training )
        x = self.fc2( x )
        return x


class vgg19_exits_eval( nn.Module ):
    def __init__(self):
        super(vgg19_exits_eval, self).__init__()
        self.exit_num = 3
        # init = gp.vgg_exits_eval_init
        # Base network
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding=1)     # maxpool2d
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)   # maxpool2d, exit, 8*8
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, 3, 1, padding=1)   # maxpool2d, exit, 4*4
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 512, 3, 1, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, padding=1)  # maxpool2d, exit, 2*2
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.conv14 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 512, 3, 1, padding=1)  # maxpool2d
        self.bn16 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, 10)
        # early exits
        self.exit_1 = LogisticConvBoF(128, 64, split_horizon=8)
        self.exit_2 = LogisticConvBoF(256, 64, split_horizon=4)
        self.exit_3 = LogisticConvBoF(512, 64, split_horizon=2)
        self.exit_1_fc = nn.Linear(64, 10)
        # threshold for switching between layers
        self.activation_threshold_1 = 0
        self.activation_threshold_2 = 0
        self.activation_threshold_3 = 0
        # the number of early exits
        self.num_early_exit_1 = 0
        self.num_early_exit_2 = 0
        self.num_early_exit_3 = 0
        self.original = 0
        # the beta coefficient used for accuracy-speed trade-off, the higher the more accurate
        self.beta = 0
    
    def set_activation_thresholds( self, threshold_list:list ):
        if len( threshold_list ) != self.exit_num:
            print( f'the length of the threshold_list ({len(threshold_list)}) is invalid, should be {self.exit_num}' )
            raise NotImplementedError
        self.activation_threshold_1 = abs( threshold_list[0] )
        self.activation_threshold_2 = abs( threshold_list[1] )
        self.activation_threshold_3 = abs( threshold_list[2] )

    def print_exit_percentage( self ):
        total_inference = self.num_early_exit_1 + self.num_early_exit_2 + self.num_early_exit_3 + self.original
        print( f'early exit 1: {100*self.num_early_exit_1/total_inference:.3f}% ({self.num_early_exit_1}/{total_inference})', end=' | ' )
        print( f'early exit 2: {100*self.num_early_exit_2/total_inference:.3f}% ({self.num_early_exit_2}/{total_inference})', end=' | ' )
        print( f'early exit 3: {100*self.num_early_exit_3/total_inference:.3f}% ({self.num_early_exit_3}/{total_inference})', end=' | ' )
        print( f'original: {100*self.original/total_inference:.3f}% ({self.original}/{total_inference})' )

    def set_beta( self, beta ):
        self.beta = beta
    
    def _calculate_max_activation( self, param ):
        '''
        return the maximum activation item in [param]
        '''
        return torch.max( torch.abs( torch.max( param ) ), torch.abs( torch.min( param ) ) )

    def forward( self, x ):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)
        x_exit_1 = self.exit_1(x)
        exit1 = self.exit_1_fc(x_exit_1)
        if self._calculate_max_activation( exit1 ) > self.beta * self.activation_threshold_1:
            self.num_early_exit_1 += 1
            return 0, exit1
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.max_pool2d(x, 2, 2)
        x_exit_2 = self.exit_2(x)
        exit2 = (x_exit_1 + x_exit_2) / 2
        exit2 = self.exit_1_fc(exit2)
        if self._calculate_max_activation( exit2 ) > self.beta * self.activation_threshold_2:
            self.num_early_exit_2 += 1
            return 1, exit2
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.max_pool2d(x, 2, 2)
        x_exit_3 = self.exit_3(x)
        exit3 = (x_exit_1 + x_exit_2 + x_exit_3) / 3
        exit3 = self.exit_1_fc(exit3)
        if self._calculate_max_activation( exit3 ) > self.beta * self.activation_threshold_3:
            self.num_early_exit_3 += 1
            return 2, exit3
        x = F.relu(self.bn13(self.conv13(x)))
        x = F.relu(self.bn14(self.conv14(x)))
        x = F.relu(self.bn15(self.conv15(x)))
        x = F.relu(self.bn16(self.conv16(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.view(-1, 512)
        x = self.fc(x)
        self.original += 1
        return 3, x


class vgg19_exits_train( vgg19_exits_eval ):
    def __init__( self ):
        super().__init__()
        self.exit_layer = 'original'
    
    def set_exit_layer(self, exit_layer):
        if exit_layer not in ['original', 'exits']:
            print( f'Error: exit_layer ({exit_layer}) is invalid. Should be original or exits' )
            raise NotImplementedError
        self.exit_layer = exit_layer
    
    # the functions starting from here should be updated by json initializations!
    def forward( self, x ):
        if self.exit_layer == 'original':
            return self.forward_original( x )
        elif self.exit_layer == 'exits':
            return self.forward_exits( x )

    def forward_original( self, x ):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn13(self.conv13(x)))
        x = F.relu(self.bn14(self.conv14(x)))
        x = F.relu(self.bn15(self.conv15(x)))
        x = F.relu(self.bn16(self.conv16(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x

    def forward_exits( self, x ):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)
        x_exit_1 = self.exit_1(x)
        exit1 = self.exit_1_fc(x_exit_1)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.max_pool2d(x, 2, 2)
        x_exit_2 = self.exit_2(x)
        exit2 = (x_exit_1 + x_exit_2) / 2
        exit2 = self.exit_1_fc(exit2)
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.max_pool2d(x, 2, 2)
        x_exit_3 = self.exit_3(x)
        exit3 = (x_exit_1 + x_exit_2 + x_exit_3) / 3
        exit3 = self.exit_1_fc(exit3)
        return (exit1, exit2, exit3)


class vgg19_normal( nn.Module ):
    def __init__(self):
        super(vgg19_normal, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding=1)     # maxpool2d
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)   # maxpool2d
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, 3, 1, padding=1)   # maxpool2d
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 512, 3, 1, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, padding=1)  # maxpool2d
        self.bn12 = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.conv14 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 512, 3, 1, padding=1)  # maxpool2d
        self.bn16 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn13(self.conv13(x)))
        x = F.relu(self.bn14(self.conv14(x)))
        x = F.relu(self.bn15(self.conv15(x)))
        x = F.relu(self.bn16(self.conv16(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 1)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    print( 'for normal:' )
    model = vgg19_normal()
    for name, _ in model.state_dict().items():
        print( name )
    print( '\nfor eval_exits:' )
    model = vgg19_exits_eval()
    for name, _ in model.state_dict().items():
        print( name )