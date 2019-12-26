"""
Train a neural network on the given dataset with given configuration

"""
import argparse
import math
import re
import sys
import traceback

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
from data_utils import *
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.python import debug as tf_debug
from train_utils import *

parser = argparse.ArgumentParser(description='run ml regressors on dataset',argument_default=argparse.SUPPRESS)
parser.add_argument('--train_data_path', help='path to the training dataset',default=None, type=str, required=False)
parser.add_argument('--test_data_path', help='path to the test dataset', default=None, type=str,required=False)
parser.add_argument('--label', help='output variable', default=None, type=str,required=False)
parser.add_argument('--input', help='input attributes set', default=None, type=str, required=False)
parser.add_argument('--config_file', help='configuration file path', default=None, type=str, required=False)
parser.add_argument('--test_metric', help='test_metric to use', default=None, type=str, required=False)

parser.add_argument('--priority', help='priority of this job', default=0, type=int, required=False)

args,_ = parser.parse_known_args()

hyper_params = {'batch_size':32, 'num_epochs':4000, 'EVAL_FREQUENCY':1000, 'learning_rate':1e-4, 'momentum':0.9, 'lr_drop_rate':0.5, 'epoch_step':500, 'nesterov':True, 'reg_W':0., 'optimizer':'Adam', 'reg_type':'L2', 'activation':'relu', 'patience':100}

# NN architecture

SEED = 66478

def run_regressors(train_X, train_y, valid_X, valid_y, test_X, test_y, logger=None, config=None):
    assert config is not None
    hyper_params.update(config['paramsGrid'])
    assert  logger is not None
    rr = logger

    def model_slim(data, architecture, train=True, num_labels=1, activation='relu', dropouts=[]):
        if train:
            reuse = None
        else:
            reuse = True

        if activation == 'relu':
            activation = tf.nn.relu
        assert '-' in architecture
        archs = architecture.strip().split('-')
        net = data
        pen_layer = net
        prev_layer = net
        prev_num_outputs = None
        prev_block_num_outputs = None
        prev_stub_output = net
        for i in range(len(archs)):
            arch = archs[i]
            if 'x' in arch:
                arch = arch.split('x')
                num_outputs = int(re.findall(r'\d+',arch[0])[0])
                layers = int(re.findall(r'\d+',arch[1])[0])
                j = 0
                aux_layers = re.findall(r'[A-Z]',arch[0])
                for l in range(layers):
                    if aux_layers and aux_layers[0] == 'B':
                        if len(aux_layers)>1 and aux_layers[1]=='A':
                            rr.fprint('adding fully connected layers with %d outputs followed by batch_norm and act' % num_outputs)

                            net = slim.layers.fully_connected(net, num_outputs=num_outputs,
                                                              scope='fc' + str(i) + '_' + str(j),
                                                              activation_fn=None, reuse=reuse)
                            net = slim.layers.batch_norm(net, center=True, scale=True, reuse=reuse, scope='fc_bn'+str(i)+'_'+str(j))
                            net = activation(net)
                        else:
                            rr.fprint('adding fully connected layers with %d outputs followed by batch_norm' % num_outputs)
                            net = slim.layers.fully_connected(net, num_outputs=num_outputs,
                                                              scope='fc' + str(i) + '_' + str(j),
                                                              activation_fn=activation, reuse=reuse)
                            net = slim.layers.batch_norm(net, center=True, scale=True, reuse=reuse,
                                             scope='fc_bn' + str(i) + '_' + str(j))

                    else:
                        rr.fprint('adding fully connected layers with %d outputs' % num_outputs)

                        net = slim.layers.fully_connected(net, num_outputs=num_outputs,
                                                          scope='fc' + str(i) + '_' + str(j), activation_fn=activation,
                                                              reuse=reuse)
                    if 'R' in aux_layers:
                        if prev_num_outputs and prev_num_outputs==num_outputs:
                            rr.fprint('adding residual, both sizes are same')

                            net = net+prev_layer
                        else:
                            rr.fprint('adding residual with fc as the size are different')
                            net = net + slim.layers.fully_connected(prev_layer, num_outputs=num_outputs,
                                                                  scope='fc' + str(i) + '_' +'dim_'+ str(j),
                                                          activation_fn=None, reuse=reuse)
                    prev_num_outputs = num_outputs
                    j += 1
                    prev_layer = net
                aux_layers_sub = re.findall(r'[A-Z]', arch[1])
                if 'R' in aux_layers_sub:
                    if prev_block_num_outputs and prev_block_num_outputs == num_outputs:
                        rr.fprint('adding residual to stub, both sizes are same')
                        net = net + prev_stub_output
                    else:
                        rr.fprint('adding residual to stub with fc as the size are different')
                        net = net + slim.layers.fully_connected(prev_stub_output, num_outputs=num_outputs,
                                                            scope='fc' + str(i) + '_' + 'stub_dim_' + str(j),
                                                            activation_fn=None, reuse=reuse)

                if 'D' in aux_layers_sub and (train or num_labels == 1) and len(dropouts) > i:
                    rr.fprint('adding dropout', dropouts[i])
                    net = tf.nn.dropout(net, dropouts[i], seed=SEED)
                prev_stub_output = net
                prev_block_num_outputs = num_outputs
                prev_layer = net

            else:
                if 'R' in arch:
                    act_fun = tf.nn.relu
                    rr.fprint('using ReLU at last layer')
                else:
                    act_fun = None
                pen_layer = net
                rr.fprint('adding final layer with ' + str(num_labels) + ' output')
                net = slim.layers.fully_connected(net, num_outputs=num_labels, scope='fc' + str(i),
                                                  activation_fn=act_fun, reuse=reuse)

        net = tf.squeeze(net)
        return net, pen_layer

        net = tf.squeeze(net)
        return net, pen_layer

    def error_rate(predictions, labels, step=0, dataset_partition=''):

        return np.mean(np.absolute(predictions - labels))

    def error_rate_classification(predictions, labels, step=0, dataset_partition=''):
        return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])

    tf.reset_default_graph()
    train_X = train_X.reshape(train_X.shape[0], -1).astype("float32")
    valid_X = valid_X.reshape(valid_X.shape[0], -1).astype("float32")
    test_X = test_X.reshape(test_X.shape[0], -1).astype("float32")

    num_input = train_X.shape[1]
    batch_size = hyper_params['batch_size']
    learning_rate = hyper_params['learning_rate']
    optimizer = hyper_params['optimizer']
    architecture = config['architecture']
    num_epochs = hyper_params['num_epochs']
    model_path = config['model_path']
    patience = hyper_params['patience']
    save_path = config['save_path']
    loss_type = config['loss_type']
    if 'dropouts' in hyper_params:
        dropouts = hyper_params['dropouts']
    else:
        dropouts = []
    test_metric = mean_squared_error
    if config['test_metric']=='mae':
        test_metric = mean_absolute_error
    use_valid = config['use_valid']
    EVAL_FREQUENCY = hyper_params['EVAL_FREQUENCY']


    train_y = train_y.reshape(train_y.shape[0]).astype("float32")
    valid_y = valid_y.reshape(valid_y.shape[0]).astype("float32")
    test_y = test_y.reshape(test_y.shape[0]).astype("float32")

    train_data = train_X
    train_labels = train_y
    test_data = test_X
    test_labels = test_y
    validation_data = valid_X
    validation_labels = valid_y


    rr.fprint("train matrix shape of train_X: ",train_X.shape, ' train_y: ', train_y.shape)
    rr.fprint("valid matrix shape of train_X: ",valid_X.shape, ' valid_y: ', valid_y.shape)
    rr.fprint("test matrix shape of valid_X:  ",test_X.shape, ' test_y: ', test_y.shape)
    rr.fprint('architecture is: ',architecture)
    rr.fprint('learning rate is ',learning_rate)

    train_data_node = tf.placeholder(tf.float32, shape=(batch_size, num_input))
    eval_data = tf.placeholder(tf.float32, shape=(batch_size, num_input))

    logits,_ = model_slim(train_data_node, architecture, dropouts=dropouts)
    train_labels_node = tf.placeholder(tf.float32, shape=(batch_size))
    assert  loss_type == 'mae'
    if loss_type == 'mae':
        loss = tf.reduce_mean(tf.abs(train_labels_node - logits))  # * (180 / math.pi)

    batch = tf.Variable(0)

    assert optimizer=='Adam'
    if optimizer=='Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)

    eval_prediction,_ = model_slim(eval_data, architecture,train=False, dropouts=dropouts)

    def eval_in_batches(data, sess):
        size = data.shape[0]
        if size < batch_size:
            raise ValueError('batch size for evals larger than dataset: %d' % size)
        predictions = np.ndarray(shape=(size), dtype=np.float32)
        for begin in range(0, size, batch_size):
            end = begin + batch_size
            if end <= size:
                # predictions[:,begin:end] \
                output = sess.run(eval_prediction, feed_dict={eval_data: data[begin:end, ...]})
                predictions[begin:end] = output
            else:
                batch_predictions = sess.run(eval_prediction, feed_dict={eval_data: data[-batch_size:, ...]})
                predictions[-batch_size:] = batch_predictions
        return predictions

    start_time = time.time()
    print ('num_epochs is ', num_epochs)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    rr.fprint('Initialized')
    train_writer = tf.summary.FileWriter('summary', graph_def=sess.graph_def)

    train_size = train_X.shape[0]

    best_val_error = 100

    patience_steps = int(patience * train_size/batch_size)
    best_step = 0

    saver = tf.train.Saver()

    rr.fprint('model path is ', model_path)
    if model_path and os.path.exists(model_path+'.meta'):
        rr.fprint('Restoring model from %s' % model_path)
        saver.restore(sess, model_path)
    if save_path and not model_path and os.path.exists(save_path+'.meta'):
        rr.fprint('Restoring model from %s' % save_path)
        saver.restore(sess, save_path)

    rr.fprint('start training')

    #with dsess as sess:
    step=0
    #for step in xrange(int(num_epochs*train_size) // batch_size +1):
    while True:
        offset = (step * batch_size) % (train_size - batch_size)
        batch_data = train_data[offset:(offset + batch_size),...]
        batch_labels = train_labels[offset:(offset + batch_size)]
        feed_dict = {train_data_node: batch_data,
                     train_labels_node: batch_labels}
        _, logits_, l_ = sess.run([optimizer, logits, loss], feed_dict=feed_dict)
        if step % EVAL_FREQUENCY == 0:
            elapsed_time = time.time() - start_time

            if use_valid:
                val_predictions = eval_in_batches(validation_data, sess)
                val_error = test_metric(val_predictions, validation_labels)
            test_predictions = eval_in_batches(test_data, sess)
            test_error = test_metric(test_predictions, test_labels)
            if not use_valid:
                val_error = test_error
            if best_val_error > val_error:
                best_val_error = val_error
                best_step = step
                if save_path:
                    save_path_ = saver.save(sess, save_path)
                    rr.fprint('Model saved at: %s' % save_path_)

            rr.fprint(
                'Step %d (epoch %.2d), %.1f s minibatch loss: %.5f, validation error: %.5f, test error: %.5f, best validation error: %.5f' % (
                step, int(step * batch_size) / train_size,
                elapsed_time, l_, val_error, test_error, best_val_error))

            if best_step + patience_steps <= step:
                rr.fprint('No improvement observed in last %d steps, best error in validation set is %f'%(patience_steps, best_val_error))
                return best_val_error
            sys.stdout.flush()
            start_time = time.time()
        step += 1
    train_writer.close()
    return best_val_error


if __name__=='__main__':
    args = parser.parse_args()
    config = {}
    config['train_data_path'] = args.train_data_path
    config['test_data_path'] = args.test_data_path
    config['label'] = args.label
    config['input_type'] = args.input
    config['log_folder'] = 'logs_dl'
    config['log_file'] = 'dl_log_' + get_date_str() + '.log'
    config['test_metric'] = args.test_metric
    config['architecture'] = 'infile'
    if args.config_file:
        config.update(load_config(args.config_file))
    if not os.path.exists(config['log_folder']):
        createDir(config['log_folder'])
    logger = Record_Results(os.path.join(config['log_folder'], config['log_file']))
    logger.fprint('job config: ' + str(config))
    train_X, train_y, valid_X, valid_y, test_X, test_y = load_csv(config['train_data_path'],
                                                                  test_data_path=config['test_data_path'],
                                                                  input_types=config['input_types'],
                                                                  label=config['label'], logger=logger)
    run_regressors(train_X, train_y, valid_X, valid_y, test_X, test_y, logger=logger, config=config)
    logger.fprint('done')


