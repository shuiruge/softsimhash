import os
import argparse
import numpy as np
import tensorflow as tf
from softsimhash import SoftSimhash, Loss


SCRIPT_PATH = os.path.abspath(__file__)
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_PATH), '../dat')


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=DATA_DIR, type=str,
                    help='directory to the tfrecord files.')
parser.add_argument('--batch_size', default=512, type=int, help='batch size')
parser.add_argument('--train_steps', default=20000, type=int,
                    help='number of training steps')
parser.add_argument('--eval_steps', default=100, type=int,
                    help='number of evaluation steps')
parser.add_argument('--threshold', default=6, type=int,
                    help='threshold in hardenizing')
parser.add_argument('--allow_neg_weight', default=False, type=bool)


def softplus(x, limit=30):
  x = np.array(x)
  return np.where(x > np.ones_like(x) * limit,
                  x, np.log(1.0 + np.exp(x)))


def get_input_fn(tfrecord_filenames, batch_size):

    def parse(example):
        return  # TODO: your code.

    def input_fn():
        raw_dataset = tf.data.TFRecordDataset(tfrecord_filenames)
        dataset = raw_dataset.map(parse)
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat().batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    return input_fn


def model_fn(features, labels, mode, params):
    weight = tf.Variable(params['init_weight'], name='weight')
    if not params['allow_neg_weight']:
        # Keep weight always positive
        weight = tf.nn.softplus(weight)

    hash0 = features['hash0']
    hash1 = features['hash1']
    ssh0 = SoftSimhash.generate(weight, hash0)
    ssh1 = SoftSimhash.generate(weight, hash1)

    # Loss
    loss = Loss()(ssh0, ssh1, labels)

    # Prediction
    hd = ssh0.hamming_distance(ssh1)
    predictions = tf.where(
        hd < tf.ones_like(hd) * params['threshold'],
        tf.ones_like(hd),  # the same device.
        tf.zeros_like(hd))  # not the same device.

    # Accuracy
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predictions,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer().minimize(
            loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode, predictions={'is_save_device': predictions})

    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    else:
        raise Exception('Bad `mode` value: {}'.format(str(mode)))


def main(argv):
    args = parser.parse_args(argv[1:])

    all_filenames = os.listdir(args.data_dir)
    tfrecord_filenames = [
        _ for _ in all_filenames if _.split('.')[-1] == 'tfrecord']
    # Use absolute path
    tfrecord_filenames = [
        os.path.join(args.data_dir, _) for _ in tfrecord_filenames]
    tf.logging.info('Found data-files: {}'
                    .format(', '.join(tfrecord_filenames)))

    # Model
    n_features = len(FEATURES)
    init_weight = np.random.normal(size=[n_features])

    feature_columns = []
    for key in ['hash0', 'hash1']:
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params={
            'feature_columns': feature_columns,
            'init_weight': init_weight,
            'threshold': args.threshold,
            'allow_neg_weight': args.allow_neg_weight,
        })

    # Train the Model.
    estimator.train(
        input_fn=get_input_fn(tfrecord_filenames,
                              args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = estimator.evaluate(
        input_fn=get_input_fn(tfrecord_filenames,
                              args.batch_size),
        steps=args.eval_steps)
    tf.logging.info('Test set accuracy: {accuracy:0.3f}'
                    .format(**eval_result))

    # Display the trained weight
    trained_weight = estimator.get_variable_value('weight')
    if not args.allow_neg_weight:
        trained_weight = softplus(trained_weight)
    tf.logging.info('Trained weight:\n{}'.format(trained_weight))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
