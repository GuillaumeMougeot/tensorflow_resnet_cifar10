#----------------------------------------------------------------------------
# A simple classification on CIFAR10 dataset
#----------------------------------------------------------------------------

import tensorflow as tf 
from datetime import datetime
from time import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

import dataset
from augmentation import ImageAugmentation, hflip, crop
import net

#----------------------------------------------------------------------------
# GLOBAL VARIABLES

TRAIN_DATA_PATH = 'data/tfrecords/cifar10_train.tfrecords'
TEST_DATA_PATH = 'data/tfrecords/cifar10_test.tfrecords'

CIFAR10_CLASSES = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

BATCH_SIZE = 128
SHUFFLE = True # shuffle the dataset?

MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

MODEL = net.ResNet

# Uncomment/comment the following line depending on the size of the resnet you want to train
MODEL_NAME = None; NUM_BLOCKS=-1 # trains all the resnets
# MODEL_NAME = "ResNet20"; NUM_BLOCKS=[3,3,3] # trains only resnet20
# MODEL_NAME = "ResNet32"; NUM_BLOCKS=[5,5,5] # trains only resnet32
# MODEL_NAME = "ResNet44"; NUM_BLOCKS=[7,7,7] # trains only resnet44
# MODEL_NAME = "ResNet56"; NUM_BLOCKS=[9,9,9] # trains only resnet56
# MODEL_NAME = "ResNet110"; NUM_BLOCKS=[18,18,18] # trains only resnet110
# MODEL_NAME = "ResNet1202"; NUM_BLOCKS=[200,200,200] # trains only resnet1202

LOAD_MODEL = None

NBOF_STEPS = int(8e6)//BATCH_SIZE # = 160 epochs
PRINT_PERIOD = int(1e4)//BATCH_SIZE
TEST_PERIOD = int(1e5)//BATCH_SIZE
MODEL_SAVING_PERIOD = TEST_PERIOD*2
IMAGE_SAVING_PERIOD = TEST_PERIOD*2

#----------------------------------------------------------------------------
# Train loop.

def preprocess_data(image, label):
    """
    Normalizes the data.
    """
    image = tf.cast(image, tf.float32)
    image = image / 255.
    image = image - tf.reshape([0.485,0.456,0.406], [1,1,3])
    image = image / tf.reshape([0.229,0.224,0.225], [1,1,3])
    label = tf.squeeze(tf.cast(label, tf.int64))
    return image, label

def plot_images(images, preds):
    """
    Plots and saves an images of a batch of images with the corresponding predicted labels.
    """
    plt.figure()
    for i,image in enumerate(images[:32]):
        plt.subplot(4,8,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.title(CIFAR10_CLASSES[preds[i]])
        plt.imshow(image)

def create_save_dirs(dir_names):
    """
    Creates saving folders. 

    Arguments:
        dir_names: a list of name of the desired folders.
                   e.g.: ['images','cpkt','summary']
    
    Returns:
        list_dirs: a list of path of the corresponding folders.
    """
    list_dirs = []
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    for name in dir_names:
        list_dirs += ['logs/' + current_time + '/' + name]
        if not os.path.exists(list_dirs[-1]):
            os.makedirs(list_dirs[-1])
    return list_dirs

def accuracy(y_true, y_pred):
    assert len(y_true.shape)==1 and len(y_pred.shape)==1, "[Error] y_true and y_pred must have only one dimension."
    assert y_true.shape==y_pred.shape, "[Error] y_true and y_pred must have the same shapes."
    trues = tf.cast(y_true==y_pred, tf.int64) # the true positives + the true negatives
    return tf.reduce_mean(trues) 

def train(num_blocks=[3,3,3]):
    # loads the data
    train_data = dataset.create_from_tfrecord(
        TRAIN_DATA_PATH,
        pre_process_func=preprocess_data,
        minibatch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        repeat=True,
        jpeg_encoded=True)
    test_data = dataset.create_from_tfrecord(
        TEST_DATA_PATH,
        pre_process_func=None,
        minibatch_size=BATCH_SIZE,
        shuffle=False,
        repeat=False,
        jpeg_encoded=True)

    # data augmentation
    aug_gen = ImageAugmentation([hflip, crop])

    # saves setup: creates the saving directories
    model_dir, image_dir, log_dir = create_save_dirs(['model', 'images', 'summary'])
    summary_writer = tf.summary.create_file_writer(log_dir)

    # builds the model or loads it
    if LOAD_MODEL is None:
        model = MODEL(num_blocks=num_blocks)
    else:
        model = tf.keras.models.load_model(LOAD_MODEL)
        print("model loaded from "+LOAD_MODEL)

    # learning rate scheduler (including the "warm-up" from the original paper)
    init_lr = 0.1
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      [500, 32000, 48000], 
      [init_lr / 10., init_lr, init_lr / 10., init_lr / 100.])
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=MOMENTUM)

    # model metric
    acc_metric = lambda labels, logits: tf.reduce_mean(tf.cast(tf.equal(
                                        labels, tf.argmax(logits, 1)), 'float32'))

    train_step_signature = [
        tf.TensorSpec(shape=(BATCH_SIZE, 32, 32, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.int64)]

    @tf.function(input_signature=train_step_signature)
    def train_step(images, labels):
        """Performs one optimizer step on a single mini-batch."""
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)
            loss = tf.reduce_mean(loss)
            regularization_losses = model.losses
            total_loss = tf.add_n(regularization_losses + [loss])

        params = model.trainable_variables
        grads = tape.gradient(total_loss, params)
        opt.apply_gradients(zip(grads, params))
        return total_loss, logits

    @tf.function
    def test_step(images, labels):
        """Performs one evaluation step on a mini-batch"""
        logits = model(images, training=False)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)
        loss = tf.reduce_mean(loss)
        regularization_losses = model.losses
        total_loss = tf.add_n(regularization_losses + [loss])
        return total_loss, logits
    
    
    start_time = time()
    step = 0

    for images, labels in train_data.take(NBOF_STEPS):
        images = aug_gen(images) # augments image data
        loss, logits = train_step(images, labels) # train step
        train_acc = acc_metric(labels, logits) 

        # prints model summary
        if step == 0: model.summary()

        # prints model loss
        if step % PRINT_PERIOD == 0:
            # print('{:s}\r'.format(''), end='', flush=True)
            print('training: images {} time {:.2f} loss {:.4f} acc {:.4f}'.format(step*BATCH_SIZE, time()-start_time, loss.numpy(), train_acc))

        # saves model
        if step % MODEL_SAVING_PERIOD == 0:
            save_location = model_dir+'/'+MODEL_NAME
            model.save(save_location)
            print("model saved into " + save_location)
        
        # evaluates the model on the testing set
        if step % TEST_PERIOD == 0:
            count = 0
            test_loss = 0
            test_acc = 0
            for _images, _labels in test_data:
                _images, _labels = preprocess_data(_images, _labels)
                crt_loss, test_logits = test_step(_images, _labels)
                test_loss += crt_loss
                test_acc += acc_metric(_labels, test_logits)
                count += 1
            if count > 0:
                test_loss /= count
                test_acc /= count 
            print("evaluation: loss {:.4f} acc {:.4f}".format(test_loss, test_acc))

        # saves predictions on a sample of images
        if step % IMAGE_SAVING_PERIOD == 0:
            images, labels = next(iter(test_data))
            # makes a prediction
            _images, _labels = preprocess_data(images, labels)
            preds = tf.nn.softmax(model(_images, training=False))
            # displays it
            plot_images(images, tf.argmax(preds, axis=-1).numpy())
            plt.savefig(image_dir+'/'+str(step*BATCH_SIZE)+'.png')
            plt.close()

        # saves summary
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', loss, step=step*BATCH_SIZE)
            tf.summary.scalar('train_acc', train_acc, step=step*BATCH_SIZE)
            tf.summary.scalar('test_loss', test_loss, step=step*BATCH_SIZE)
            tf.summary.scalar('test_acc', test_acc, step=step*BATCH_SIZE)
        
        step += 1

    save_location = model_dir+'/'+MODEL_NAME
    model.save(save_location)
    print("\n\nfinal model saved into " + save_location)

#----------------------------------------------------------------------------

if __name__=='__main__':
    resnets = [[3,3,3],[5,5,5],[7,7,7],[9,9,9],[18,18,18],[200,200,200]]
    resnets_name = ["ResNet20", "ResNet32", "ResNet44", "ResNet56", "ResNet110", "ResNet1202"]
    
    if MODEL_NAME is None and NUM_BLOCKS == -1:
        for i in range(len(resnets)):
            MODEL_NAME = resnets_name[i]
            train(num_blocks=resnets[i])
    
    if MODEL_NAME is not None and NUM_BLOCKS in resnets:
        train(num_blocks=NUM_BLOCKS)

#----------------------------------------------------------------------------

