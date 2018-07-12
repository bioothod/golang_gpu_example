import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, required=True, help='Output path for frozen protobuf file')

def main(args=None):
    with tf.Graph().as_default() as g:
        ph0 = tf.placeholder(tf.int32, [None, 3], name='input/ph0')
        ph1 = tf.placeholder(tf.int32, [None, 3], name='input/ph1')

        #op = tf.reduce_sum(ph0 * ph1, axis=1, name='output/op')
        op = tf.matmul(ph0, ph1, transpose_a=True, name='output/op')

        with tf.Session(graph=g) as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, g.as_graph_def(), ['output/op'])

            try:
                with tf.gfile.GFile(FLAGS.output, "wb+") as f:
                    f.write(output_graph_def.SerializeToString())
            except Exception as e:
                print('Could not save graph to {}: {}'.format(FLAGS.output, e))
                return

if __name__ == '__main__':
    FLAGS = parser.parse_args()

    tf.app.run()
