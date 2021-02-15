#----------------------------------------------------------------------------
# Utilities for reading serialized data in .tfrecords format and for 
# generating tf.data.Dataset.
#----------------------------------------------------------------------------

import os
import tensorflow as tf
from functools import partial

#----------------------------------------------------------------------------
# Dataset generator (from .tfrecords file).

def from_tfrecord_parse(
    record, 
    pre_process_func=None, 
    jpeg_encoded=False):
    """
    This function is made to work with the prepare_data.TFRecordWriter class.
    It parses a single tf.Example records.

    Arguments:
        record          : the tf.Example record with the features of 
                          prepare_data.TFRecordWriter
        pre_process_func: if not None, must be a pre-processing function that will be applied on the data.
        jpeg_encoded    : is the data encoded in jpeg format?

    Returns:
        image: a properly shaped and encoded 2D image.
        label: its corresponding label.
    """
    features = tf.io.parse_single_example(record, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([1], tf.int64)})
    data = tf.io.decode_jpeg(features['image']) if jpeg_encoded else tf.io.decode_raw(features['image'], tf.uint8)
    data = tf.reshape(data, features['shape'])
    labels = features['label']
    
    # data pre_processing
    if pre_process_func: 
        data, labels = pre_process_func(data, labels)
 
    return data, labels

def create_from_tfrecord(
    data_path,              
    minibatch_size, 
    pre_process_func=None,    
    shuffle=False,          
    repeat=True,            
    jpeg_encoded=False,
    use_prefetch=True,
    use_cache=True): 
    """
    Returns a tf.data.Dataset object from a .tfrecords file. This file
    should contains the images with their respective shapes and labels.
    To be sure to have a proper record file please use the
    prepare_data.TFRecordWriter class.

    Arguments:
        data_path       : .tfrecords file path containing the images and labels.
        minibatch_size  : size of the minibatch.
        pre_process_func: if not None, must be a pre-processing function that will be applied on the data.
        shuffle         : does the data needs to be shuffled? (Default: False)
        repeat          : inifity loop over the data? (Default: True)
        jpeg_encoded    : was the data encoded in jpeg_format? (Default: False)
        use_prefetch    : whether to use or not prefetch for better performance (Default: True)
        use_cache       : whether to use or not cache memory for better performance (Default: True)

    Returns:
        dataset: a tf.data.Dataset object.
    """
    assert os.path.isfile(data_path), '[Error] {} is not file.'.format(data_path)
    dataset = tf.data.TFRecordDataset(data_path, compression_type='')
    
    # map the decoding and the pre_processing functions 
    parse = partial(from_tfrecord_parse, pre_process_func=pre_process_func, jpeg_encoded=jpeg_encoded)
    num_parallel_calls = tf.data.AUTOTUNE if tf.__version__=='2.4.1' else 12
    dataset = dataset.map(parse, num_parallel_calls=num_parallel_calls) # num_parallel_calls == tf.data.AUTOTUNE for tf >= 2.4
    
    # shuffle the dataset
    if shuffle: dataset = dataset.shuffle(buffer_size=10000)
    
    # repeat the dataset
    if repeat: dataset = dataset.repeat()
    
    # create batch of data
    dataset = dataset.batch(minibatch_size)
    
    # prefetch the data and cache it for better performance 
    buffer_size = tf.data.AUTOTUNE if tf.__version__=='2.4.1' else 10000 
    if use_prefetch: dataset.prefetch(buffer_size=buffer_size) # prefetch the next batch for better performance, buffer_size = tf.data.AUTOTUNE for tf >= 2.4
    if use_cache: dataset.cache() # efficient for multiple epoch calls
    
    return dataset

#----------------------------------------------------------------------------
