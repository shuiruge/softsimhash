import os
import argparse
import numpy as np
import tensorflow as tf
from softsimhash.softsimhash import SoftSimhash
from softsimhash.loss import Loss
from softsimhash.utils import l2_log_regularize


SCRIPT_PATH = os.path.abspath(__file__)
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_PATH), '../dat')


parser = argparse.ArgumentParser()
parser.add_argument('--n_features', type=int,
                    help='number of features')
parser.add_argument('--data_dir', default=DATA_DIR, type=str,
                    help='directory to the *.tfrecord files.')
parser.add_argument('--model_dir', default=None, type=str)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--train_steps', default=20000, type=int)
parser.add_argument('--eval_steps', default=100, type=int)
parser.add_argument('--n_evals', default=10000, type=int)
parser.add_argument('--lambda_', default=1.0, type=float)
parser.add_argument('--threshold', default=6, type=int,
                    help='threshold in hardenizing')


def get_input_fn(tfrecord_filenames, batch_size, mode, n_evals):

    assert mode in ('train', 'evaluate')

    def parse(example):
        # TODO: Your code below
        input_features = ...
        label = ...
        return (input_features, label)

    def input_fn():
        raw_dataset = tf.data.TFRecordDataset(tfrecord_filenames)
        dataset = raw_dataset.map(parse)
        if mode == 'evaluate':
            dataset = dataset.take(n_evals)
        else:
            dataset = dataset.skip(n_evals)
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat().batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    return input_fn


def get_weight_variable(n_features, name='weight'):
    weight_initializer = tf.initializers.random_uniform(
        maxval=(1 / tf.sqrt(float(n_features))))
    weight = tf.get_variable(name, shape=[n_features],
                             initializer=weight_initializer)
    return weight


def model_fn(features, labels, mode, params):
    weight = get_weight_variable(params['n_features'])
    # Keep weight always positive
    weight = tf.nn.softplus(weight)

    hash0 = features['hash0']
    hash1 = features['hash1']
    ssh0 = SoftSimhash.generate(weight, hash0)
    ssh1 = SoftSimhash.generate(weight, hash1)

    # Loss
    loss = Loss()(ssh0, ssh1, labels)
    reg_loss = loss + params['lambda_'] * l2_log_regularize(weight)

    # Prediction
    hd = ssh0.hamming_distance(ssh1)
    predictions = tf.where(
        hd < tf.ones_like(hd) * params['threshold'],
        tf.ones_like(hd),  # the same device.
        tf.zeros_like(hd))  # not the same device.

    # Metrics
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predictions,
                                   name='accuracy_op')
    precision = tf.metrics.precision(labels=labels,
                                     predictions=predictions,
                                     name='precision_op')
    recall = tf.metrics.recall(labels=labels,
                               predictions=predictions,
                               name='recall_op')
    metrics = {'accuracy': accuracy,
               'precision': precision,
               'recall': recall}

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer().minimize(
            reg_loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode, loss=reg_loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode, predictions={'is_save_device': predictions})

    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=reg_loss, eval_metric_ops=metrics)

    else:
        raise Exception('Bad `mode` value: {}'.format(str(mode)))


def main(argv):
    args = parser.parse_args(argv[1:])

    all_filenames = os.listdir(args.data_dir)
    tfrecord_filenames = [
        os.path.join(args.data_dir, _) for _ in all_filenames
        if _.split('.')[-1] == 'tfrecord']
    tf.logging.info('Found data-files: {}'
                    .format(', '.join(tfrecord_filenames)))

    # Model
    feature_columns = []
    for key in ['hash0', 'hash1']:
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_dir,
        params={'feature_columns': feature_columns,
                'lambda_': args.lambda_,
                'threshold': args.threshold})

    # Train the Model.
    estimator.train(
        input_fn=get_input_fn(tfrecord_filenames,
                              args.batch_size,
                              mode='train',
                              n_evals=args.n_evals),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = estimator.evaluate(
        input_fn=get_input_fn(tfrecord_filenames,
                              args.batch_size,
                              mode='evaluate',
                              n_evals=args.n_evals),
        steps=args.eval_steps)
    tf.logging.info('Test set result : {accuracy:0.3f}, {precision:0.3f}, '
                    '{recall:0.3f}'.format(**eval_result))

    # Display the trained weight value
    weight_val = estimator.get_variable_value('weight')
    weight_val = np_softplus(weight_val)
    tf.logging.info('Trained weight:\n{}'.format(weight_val))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
