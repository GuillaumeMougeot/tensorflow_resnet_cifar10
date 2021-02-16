# tensorflow_resnet_cifar10

This repositories contains an implementation of the [original ResNet paper](https://arxiv.org/abs/1512.03385) with tensorflow 2 and keras on the CIFAR10 dataset. 

## Requirements
Before running the training be sure the following python libraries are installed:
* tensorflow 2.3.1 or higher
* matplotlib 3.3.2 or higher
* numpy 1.18.5 or higher
* scikit-image 0.16.2 or higher

## Run
Before training, the CIFAR10 dataset needs to be converted into tfrecord files. To do so, please use 
the following command by replacing the `path/to/cifar10` with the appropriate location:
```
python prepare_data.py --data_path='/path/to/cifar10'
```
The training of all the resnets can be run with:
```
python train.py
```
If you want to train only a particular ResNet or change the training hyperparameters, please edit the global variables defined in the beginning of train.py.

## Logs
During training, this implementation will store regularly:
* the keras model
* the tensorboard logs
* images of the model predictions on a batch of test samples

## Performance
The performances below were obtained by doing only one run on all the model and taking the best test error during training. With model selection, the test errors should undoubtedly improve. 

| Name      | # layers | # params| Test err(paper) | Test err(this impl.)|
|-----------|---------:|--------:|:---------------:|:---------------------:|
|ResNet20   |    20    | 0.27M   | 8.75%           | **8.68%**|
|ResNet32   |    32    | 0.46M   | 7.51%           | **7.69%**|
|ResNet44   |    44    | 0.66M   | 7.17%           | **7.31%**|
|ResNet56   |    56    | 0.85M   | 6.97%           | **7.04%**|
|ResNet110  |   110    |  1.7M   | 6.43%           | **6.75%**|
|ResNet120  |  1202    | 19.4M   | 7.93%           | **7.33%**|

## Acknowledgement
This code is inspired by the two following repositories:
* [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10)
* [tf-resnet-cifar10](https://github.com/chao-ji/tf-resnet-cifar10)