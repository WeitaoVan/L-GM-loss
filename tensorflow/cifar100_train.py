# https://github.com/WeitaoVan/L-GM-loss
#####################    Version: Python 3, Tensorflow 1.3   ####################

import tensorflow as tf
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py
import sys
import time
import os
from scipy.misc import imresize

import resnet_model
import sklearn.metrics as metrics
import pickle as pk
from resnet_model import sigmoid_rampup, sigmoid_rampdown, step_rampup

parser = argparse.ArgumentParser()

# Basic model parameters.
tag = 'example'
parser.add_argument('--data_dir', type=str, default='/media/wwt/860G/data/CIFAR100/',
                    help='The path to the CIFAR-10 data directory.')

parser.add_argument('--model_dir', type=str, default='/media/wwt/860G/model/tf_cifar100/'+tag,
                    help='The directory where the model will be stored.')

parser.add_argument('--log_dir', type=str, default='./log/'+tag,
                    help='Directory to put the log data.')

parser.add_argument('--resnet_size', type=int, default=32,
                    help='The size of the ResNet model to use.')

parser.add_argument('--train_epochs', type=int, default=300,
                    help='The number of epochs to train.')

parser.add_argument('--batch_size', type=int, default=200,
                    help='The number of images per batch.')


_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3

_WEIGHT_DECAY = 5e-4
_MOMENTUM = 0.9

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}

def placeholder_inputs(N=None, H=_HEIGHT, W=_WIDTH, C=_DEPTH):
    # CIFAR-100: 32x32 RGB
    images_placeholder = tf.placeholder(tf.float32, shape=(N, H, W, C), name='images')
    labels_placeholder = tf.placeholder(tf.int32, shape=(N), name='labels')
    lr = tf.placeholder(tf.float32, shape=(), name='lr')
    return images_placeholder, labels_placeholder, lr

def read_h5(file_path, key):
    if not os.path.exists(file_path):
        print('file not exist:%s' %file_path)
        exit()
    h5file = h5py.File(file_path, 'r')
    return h5file[key][...]

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('folder %s created' %path)
        
def preprocess_image(image, is_training, is_rgb=True):
    if is_rgb:
    # transform to (H, W, C)
        image = np.transpose(image, axes=[1,2,0])
    else:
        image = np.expand_dims(image, axis=2)
    if is_training:
        image = random_crop(image, _HEIGHT, _WIDTH, is_rgb=is_rgb)
        s = image.shape
        assert s[0] == _HEIGHT
        assert s[1] == _WIDTH
        if np.random.choice([True, False]):
            image = np.fliplr(image)
    #else:
        #image = center_crop(image, _HEIGHT, _WIDTH)    
    image = image/127.5 - 1.0
    return image

def random_crop(x, crop_h, crop_w, is_rgb=True):
    h, w = x.shape[:2]
    h_ini = int(np.random.rand() * (h - crop_h + 1))
    w_ini = int(np.random.rand() * (w - crop_w + 1))
    if is_rgb:
        return x[h_ini : h_ini + crop_h, w_ini : w_ini + crop_w, :]
    else:
        return x[h_ini : h_ini + crop_h, w_ini : w_ini + crop_w]

def center_crop(x, crop_h, crop_w):
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return x[j:j+crop_h, i:i+crop_w, :]

def fill_feed_dict(imgs, labels, step, batch_size, is_training,
               images_placeholder, labels_placeholder, MAX=50000, preprocess_fn=preprocess_image):
    idx_range = range(step*batch_size, min(MAX, (step+1)*batch_size))
    img_batch = [preprocess_fn(imgs[idx,...], is_training) for idx in idx_range]
    img_batch = np.array(img_batch)
    label_batch = np.squeeze(labels[idx_range])
    feed_dict = {images_placeholder: img_batch, labels_placeholder:label_batch}
    return feed_dict

def fill_image(imgs, step, batch_size, images_placeholder, is_train=False, MAX=50000, preprocess_fn=preprocess_image):
    # default is_training=False
    idx_range = range(step*batch_size, min(MAX, (step+1)*batch_size))
    img_batch = [preprocess_fn(imgs[idx,...], is_train) for idx in idx_range]
    img_batch = np.array(img_batch)
    feed_dict = {images_placeholder: img_batch}
    return feed_dict

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_path,
            preprocess_fn=preprocess_image):

    # And run one epoch of eval.
    eval_batch = images_placeholder.get_shape().as_list()[0]
    img_h5 = read_h5(data_path, 'data')
    label_h5 = read_h5(data_path, 'label')
    total_num = img_h5.shape[0]
    imgs = img_h5[...]
    labels = label_h5[...]
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = total_num // eval_batch
    effective_num = steps_per_epoch * eval_batch
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(imgs, labels, step, eval_batch, False,
                               images_placeholder,labels_placeholder, 
                               MAX=total_num, preprocess_fn=preprocess_fn)
        
        true_count += sess.run(eval_correct, feed_dict=feed_dict)

    precision = float(true_count) / effective_num
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (effective_num, true_count, precision))
    return precision


def run_training(mode, num_classes, train_file, test_file):
    '''mode: 1-LGM loss, 2-softmax loss, 3-center loss
    '''
    print('mode=%d'%mode)
    print('data_dir=%s'%FLAGS.data_dir)
    print('train file=%s' %train_file)
    with tf.Graph().as_default() as g:
        # Generate placeholders for the images and labels.
        images_placeholder, labels_placeholder, lr = placeholder_inputs(FLAGS.batch_size, _HEIGHT, _WIDTH, 3)
        
        # data
        img_train_h5 = read_h5(FLAGS.data_dir + train_file, 'data')
        label_train_h5 = read_h5(FLAGS.data_dir + train_file, 'label')
        ## Build a Graph that computes predictions from the inference model.
        # normal
        is_training = True
        if mode == 1:
            logits, likelihood_reg, means = resnet_model.inference_lgm(images_placeholder, FLAGS.resnet_size, is_training, 
                                                                labels=labels_placeholder, num_classes=num_classes) # lgm loss
        elif mode == 2:
            logits = resnet_model.inference(images_placeholder, FLAGS.resnet_size, is_training, num_classes=num_classes) # softmax
        elif mode == 3: 
            logits, likelihood_reg, centers, centers_op = resnet_model.inference_center(images_placeholder, FLAGS.resnet_size, is_training,
                                                                                     labels=labels_placeholder, loss_weight=0.0005, 
                                                                                     num_classes=num_classes) # center loss
        ## Add to the Graph the Ops for loss calculation.
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.to_int64(labels_placeholder), logits=logits, name='xentropy'))
        loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        tf.summary.scalar('cross-entropy', cross_entropy)
        if mode != 2:
            tf.summary.scalar('likelihood_reg', likelihood_reg)
            loss += likelihood_reg
        optimizer = tf.train.MomentumOptimizer(lr, _MOMENTUM, use_nesterov=True)
        global_step = tf.Variable(0, name='global_step', trainable=False)  
        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        depend_ops = update_ops
        if mode == 3:
            depend_ops += [centers_op]
        with tf.control_dependencies(depend_ops):
            train_op = optimizer.minimize(loss, global_step)     

        # evaluation
        is_training = True # The tf.layers.batch_normalization works properly only when training=True. Reasons unknown.
        if mode==1:
            logits_eval, _, _ = resnet_model.inference_lgm(images_placeholder, FLAGS.resnet_size, is_training, reuse=True, num_classes=num_classes) # lgm
        elif mode==2:
            logits_eval = resnet_model.inference(images_placeholder, FLAGS.resnet_size, is_training, reuse=True, num_classes=num_classes) # softmax
        elif mode==3:
            logits_eval, _, _ = resnet_model.inference_center(images_placeholder, FLAGS.resnet_size, is_training, reuse=True, num_classes=num_classes) # center loss
        correct = tf.nn.in_top_k(logits_eval, labels_placeholder, 1)
        eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))        

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=None)

        # Create a session for running Ops on the Graph.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        sum_op = tf.summary.merge_all()
        sess.run(init)

        # load pre-trained
        if False:
            save_path = 'model.ckpt-10000'
            saver.restore(sess, save_path)
            print('\n[*]model loaded from %s\n' %save_path)    

        # Start the training loop.
        print('use %d images to train' %_NUM_IMAGES['train'])
        print('trian epochs: %d' %FLAGS.train_epochs)
        steps_per_epoch = _NUM_IMAGES['train'] // FLAGS.batch_size
        lr_value = 0.1
        g.finalize()
        for epc in range(FLAGS.train_epochs):
            idxArr = np.random.permutation(_NUM_IMAGES['train']) # shuffle
            img_train = img_train_h5[idxArr]
            label_train = label_train_h5[idxArr]
            if epc in [150, 225]:
                lr_value *= 0.1
                print('lr changed to %f'%lr_value)
            for step in range(steps_per_epoch):
                start_time = time.time()
    
                # Fill a feed dictionary with the actual set of images and labels
                # for this particular training step.
                feed_dict = fill_feed_dict(img_train, label_train, step, FLAGS.batch_size, True,
                                     images_placeholder, labels_placeholder, MAX=_NUM_IMAGES['train'])
                feed_dict[lr] = lr_value
                _, crosse_entropy_, sum_str, gs = sess.run([train_op, cross_entropy, sum_op, global_step], feed_dict=feed_dict)
    
                summary_writer.add_summary(sum_str, gs)
                summary_writer.flush()   
                duration = time.time() - start_time
                # Write the summaries and print an overview fairly often.
                if gs % 100 == 0:
                    print('(Epoch %d) GlobalStep %d: loss = %.3f (%.3f sec/step)' % (epc+1, gs, crosse_entropy_, duration))
                # Save a checkpoint and evaluate the model periodically.
                if gs % 1000 == 0 or gs == 1:
                    checkpoint_file = os.path.join(FLAGS.model_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=gs)
                    print('model saved to %s' %checkpoint_file)
                    # Evaluate against the validation set.
                    print('Validation Data Eval:')
                    do_eval(sess, eval_correct, images_placeholder, labels_placeholder,
                    FLAGS.data_dir + test_file)


                 
def main(_):
    make_if_not_exist(FLAGS.log_dir)
    make_if_not_exist(FLAGS.model_dir)
    num_classes = 100 # CIFAR100 demo
    run_training(1, num_classes, 'train_aug.h5', 'test.h5')
    

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)