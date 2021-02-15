#----------------------------------------------------------------------------
# Utilities for serializing data in TFRecord format.
#----------------------------------------------------------------------------

import argparse
import os
import numpy as np
import skimage as sk 
import matplotlib.pyplot as plt
import tensorflow as tf

#----------------------------------------------------------------------------
# Arguments parsing

parser = argparse.ArgumentParser(description='Dataset convertion to tfrecord format')
parser.add_argument('--data_path', type=str, required=True,
                    help='path to the cifar10 dataset')
parser.add_argument('--tfrecord_path', default='data/tfrecords/', type=str, 
                    help='path to where the tfrecord files will be stored (default: data/tfrecords/)')
parser.add_argument('--merge', type=bool, default=False, 
                    help='whether to merge or not the train set and the test set (default: False)')

#----------------------------------------------------------------------------
# TF recording
# This class can be used to serialize a dataset of images into a .tfrecords
# file.

class TFRecordWriter:
    def __init__(self, data_path):
        self.writer = tf.io.TFRecordWriter('{}.tfrecords'.format(data_path))
    
    def add_image(self, bytes_image, image_shape, label=None):
        """
        Adds a 2D image (encoded) to the .tfrecords file. It uses the writer
        defines during the initialization. 
        
        Arguments:
            bytes_image: the encoded image (in .jpeg, .png, ...)
            image_shape: the image shape
            label      : the image label

        Will store directly the compressed image instead of a numpy array.
        This function is recommanded for the storage space to be much smaller.
        When reading the data, a decoder will have to be used.

        To open an image without decoding it use: 
            bytes_image = open(path_to_image, 'rb').read()

        For now, this writer can only store a 2D image + its shape + (optional) its label.
        """
        # Converts label to int if different from None
        if label!=None and type(label)!=int: label = int(label)
        # Defines the features to be passed to tf.train.Example
        feature = {
            'shape':tf.train.Feature(int64_list=tf.train.Int64List(value=image_shape)),
            'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_image]))}
        # Adds label features
        if label!=None: feature['label']=tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        # Defines the protobuf
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Writes it
        self.writer.write(example.SerializeToString())

#----------------------------------------------------------------------------
# Prepares the Cifar10 dataset

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def prepare_cifar10(data_path, tfrecord_path=None, merge_test_set=True, data_format='NHWC'):
    """
    Prepares the Cifar10 and stores it into a .tfrecords file if
    tfrecord_path is defined.

    Arguments:
        data_path     : path to the folder of the Cifar10 dataset. The dataset
                        structure as to be the one found on the Cifar10 website.
        tfrecord_path: path to the tfrecords folder + name of the file.
                        e.g.: 'path_to_tfrecords_folder/filename'
        merge_test_set: whether to merge the train set and the test set into
                        the same file.
        data_format   : one of 'NHWC' or 'NCHW'
    
    Returns:
        data  : numpy array of shape (60000,32,32,3) storing all the images.
                The last 10000 contain the test set (idem for the labels).
        labels: corresponding labels.
    """

    data = np.empty((60000,32,32,3)) if data_format=='NHWC' else np.empty((60000,3,32,32))
    labels = np.empty((0))
    print('Loading dataset from {:s} ...'.format(data_path))

    # Adds the test set too
    filenames_list=[data_path+'/data_batch_{}'.format(i) for i in range(1,6)]+[data_path+'/test_batch']

    # Loads the data from files
    for i in range(len(filenames_list)):
        batch = unpickle(filenames_list[i])
        images = np.reshape(batch[b'data'], (10000,3,32,32))
        if data_format=='NHWC':
            images = np.transpose(images, (0,2,3,1))
        data[i*10000:(i+1)*10000] = images
        labels = np.append(labels,batch[b'labels'])
    
    if not os.path.exists(tfrecord_path):
        os.makedirs(tfrecord_path)

    # Stores the data and labels in a tfrecords file
    if tfrecord_path!=None:
        if merge_test_set:
            tfrecord_writer = TFRecordWriter(tfrecord_path+'cifar10')
            print('Serializing the merged dataset...')
            for i in range(len(data)):
                tfrecord_writer.add_image(
                    tf.io.encode_jpeg(data[i], quality=100).numpy(),
                    (32,32,3),
                    labels[i])
        else:
            tfrecord_writer_train = TFRecordWriter(tfrecord_path+'cifar10_train')
            print('Serializing the train set...')
            for i in range(50000):
                tfrecord_writer_train.add_image(
                    tf.io.encode_jpeg(data[i], quality=100).numpy(),
                    (32,32,3),
                    labels[i])
            tfrecord_writer_test = TFRecordWriter(tfrecord_path+'cifar10_test')
            print('Serializing the test set...')
            for i in range(50000,len(data)):
                tfrecord_writer_test.add_image(
                    tf.io.encode_jpeg(data[i], quality=100).numpy(),
                    (32,32,3),
                    labels[i])
    
    print('Done: Cifar10 dataset prepared.')
    return data, labels

#----------------------------------------------------------------------------
# Prepares the data

if __name__=='__main__':
    # prepare_cifar10('../data/cifar10/',merge_test_set=False,tfrecord_path='../data/tfrecords/cifar10')
    args = parser.parse_args()
    prepare_cifar10(args.data_path, merge_test_set=args.merge, tfrecord_path=args.tfrecord_path)

#----------------------------------------------------------------------------
