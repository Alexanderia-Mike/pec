---
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
---

---
<h2 align="center"> ECE4730J Final Project </h2>

<h5 align="center"> UM-SJTU Joint Institute </h5>

<h5 align="center"> Electrical and Computer Engineering </h5>

---


<table>
  <thead align="center">
    <tr align="center">
      <th colspan="3" align="center"> Authors </th>
    </tr>
  </thead>
  <tbody>
		<tr>
			<td align="center"> 娄辰飞 </td>
            <td align="center"> 朱正平 </td>
            <td align="center"> 陈禹池 </td>
		</tr>
  </tbody>
</table>
---

### File Description

- `nikolaos/`
    - `bof_utils.py`  
    the data structure of BoF (Bag-of-Features)
- `global_param.py`  
network initializations, the training and evaluation hyperparameters, the hardware configurations and so on
- `inference_new.py`  
inference of trained neural networks
- `main_new.py`  
the main program
- `model.py`  
the different types of neural network models
- `train_new.py`  
training of neural networks
- `utils_new.py`  
some general functions
- `power_management_api`  
the APIs that deal with hardwares in the embedded board Jetson TX2
- `create_custom_dataloader`  
the functions to create customized dataloaders
- `generate_scripts.py`  
the python program to generate shell scripts for testings automatically. Except for only very few cases, this file is useless. Feel free to delete it

### Run the Codes

#### The overal Commands
The command looks like the followin  
`python main_new.py --model_name MODEL_NAME --pretrained_file PRETRAINED_FILE_NAME --optimizer OPTIM_NAME --train_mode TRAIN_MODE --stat_each_layer STAT --evaluate_mode EVALUATE_MODE --task TASK --device DEVICE --trained_file_suffix SUFFIX --beta BETA --save SAVE --baseline BASELINE --core_num CORE_NUM --cpu_freq_level C_FREQ_L --gpu_freq_level G_FREQ_L --scene SCENE --baseline_time B_TIME --sleep_time S_TIME`  
Their meanings are specified as below. Note that not all arguments are meaningful at the same time, for example, when the `TASK` is 'train', then `EVALUATE_MODE` becomes meaningless. Actually you don't need to include all these arguments in the command. Every option has a default argument, which is specified in the `main_new.py`, and when an option is missing in the command, the default value would be used.
- `MODEL_NAME`: the name of the model, either 'cifar', 'vgg' or 'vgg19'. 'cifar' is a smaller model with only 4 conv layers, and 'vgg' is a deeper model with 6 conv layers, and 'vgg19' is an extremely deep convolutional network with 16 layers. It achieves the best accuracy, and we hope that early-exit could show its advantage on this model.
- `PRETRAINED_FILE_NAME`: the file that stores the pretrained model. They are all stored under the directory 'models_new'
    - '[model_name]_normal_default.pt' is the model for vanilla training
    - '[model_name]_original_default.pt' is the model with early-exit layers implemented but only original layers pre-trained.
    - '[model_name]_exits_default.pt' means the model with early-exit layers implemented and well trained.
- `OPTIM_NAME`: the name of the optimizer. Either 'sgd' or 'adam'. Usually by using 'adam' the training can be much more efficient.
- `TRAIN_MODE`: the mode for training. 
    - 'normal': train the model according to the most vanilla setting ---- without any early-exit schemes
    - 'original': train the model with early-exit layers implemented but only train the original layers and ignore the early-exit layers. This mode stores a pre-trained file that is necessary for the following 'exits' mode
    - 'exits': train the early-exit layers of the model, assuming that the original layers have already been trained.
- `STAT`: whether to collect the statistics of each layer, 1 for True and 0 for False
- `EVALUATE_MODE`: the mode for evaluation.
    - 'normal': evaluate the normal mode without any implementation of early exit layers
    - 'exits': evaluate the model with early exit layers and adaptive inference methods
- `TASK`: the task type, either 'train' or 'evaluate'. 'train' means to train and store a model based on training dataset, while 'evaluate' means to inference a model based on testing dataset
- `DEVICE`: the device where the training and inference is executed. Either 'cpu' or 'cuda'
- `SUFFIX`: when `SAVE` is non-zero, the model would be saved in a .pt file whose name consists of some some setting parameters and the string `SUFFIX`. This option is set mainly used to differentiate files under the same hyperparameters, so that they do not overwrite each other.
- `BETA`: the hyperparameter to control the tradeoff between accuracy and speed-up in early-exit scheme. Refer to the paper or our report for more details
- 
- `SAVE`: if non-zero, then save the model, otherwise does not save the model
- `BASELINE`: an integer, to specify whether or not the current test is baseline (1 for true and 0 for false)
- `CORE_NUM`: the number of gpu cores that are used to do the infrence task. It can only be either 2 or 4.
- `C_FREQ_L`: the level of frequency of cpu cores used in inference time. the level of frequency of cpu cores used in inference time. The value for each level is specified in the file `global_param.py`. The value can be only among [4, 8, 12].
- `G_FREQ_L`: the level of frequency of gpu cores used in inference time. The value for each level is specified in the file `global_param.py`. The value can be only among [2, 5, 8].
- `SCENE`: a string, specifying the application scene. Can only be either "continuous" or "periodical". Refer to our report for more details of what they mean
- `B_TIME`: the time it takes for baseline to finish in periodical scene. Refer to our report for more details of what it means
- `S_TIME`: the time we want the board to sleep in baseline setting in periodical scene. Refer to our report for more details of what it means

#### Changing the Experimental Setting

we can change the setting of hyperparameters by making corresponding changes in `global_param.py`. For example, `cifar_normal_train_hyper` (line 233 in `global_param.py`) specifies the hyperparameter for normal training of cifar model. By changing the epoch number, batch size and so on, the hyperparameters for that task will be changed accordingly

#### Changing the Model Structure
Unfortunately, if you want to modify the structure for a neural network model (e.g. add a layer or enlarge the layer) or define a new structure, you'll need to do that by yourself by directly modifying the file `models_new.py`. You would probably need to modify (or create) their layer name lists, their initialization and hyperparameter dictionaries in `global_param.py` accordingly. That would be somewhat frustrating, so be prepared.

### Examples

#### Trainings
!!The training should be done on the computer / server. The embedded board does not have such computational power to do the training of large convolutional neural networks!!

The first thing you'll need to do is to train a network. Say you want to train the network model `vgg19` with early exit layers, you will first train the normal layers by executing:
```shell
python main_new.py --model_name vgg19 --optimizer adam --train_mode original --task train --device cuda --trained_file_suffix round_1 --save 1
```

After that, you're gonna fix all the normal layers and start training the early exit layers by executing the command:
```shell
python main_new.py --model_name vgg19 --optimizer adam --train_mode exits --task train --device cuda --pretrained_file models_new/vgg19_train_original_round_1.pt --trained_file_suffix round_1 --save 1
```

Now basically you've done the training, and the trained model would be stored in `models_new/vgg19_train_exits_round_1.pt`. And from the command output, you'll be able to see the accuracy after the training. If you are not satisfied with the accuracy, you could modify the model structures or hyperparameters and then do the above steps again.

#### Inferences
After training the model, you may want to test the early exit results by doing some inferences with different values of the parameter `beta`. If you want to do that on computers / servers, you will need to modify the codes in `inference_new.py` by removing all the commands regarding hardwares. Then the command you're gonna type in the shell would be like:
```shell
python3 main_new.py --model_name vgg19 --optimizer adam --train_mode exits --task evaluate --device cuda --pretrained_file models_new/vgg19_train_normal_round_1.pt --trained_file_suffix round_1 --beta 5 --save 0 --evaluate_mode normal
```
There is a shell script in `scripts/inference_with_beta.sh` that could help you do such tests with `beta` ranging from 1 to 10.5. Before running the scripts, make sure you create a directory named `experimental_results_new/` under the current directory (the directory where this README.md is located).

#### Tests on the Embedded Systems
If you finally want to test the inference tasks on embedded boards, you will need to run commands like:
```shell
sudo python3 main_new.py --pretrained_file models_new/vgg19_train_exits_round_1.pt --stat_each_layer 0 --evaluate_mode exit --beta 6 --baseline 0 --core_num 2 --cpu_freq_level 4 --gpu_freq_level 2 --scene continuous --baseline_time 30 --sleep_time 30
```
There is a shell script in `scripts/testcases.sh` that could help you do all the 74 tests that are specified in our reports. Before running the scripts, make sure you create a directory named `experimental_results_new/` under the current directory (the directory where this README.md is located).