# Author: Eric Cao
# Created: 2020/07/10

from src.vggish import vggish_input, vggish_slim, vggish_params, vggish_postprocess
import tensorflow.compat.v1 as tf
import pandas
import os
import numpy

flags = tf.app.flags

flags.DEFINE_string(
    'wav_file', '../res/output.wav',
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'vggish/vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish/vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

FLAGS = flags.FLAGS

def extract_features(path):
    log_mel_example = vggish_input.wavfile_to_examples(path)

    # Prepare a postprocessor to munge the model embeddings.
    pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: log_mel_example})

    # print(embedding_batch)
    postprocessed_batch = pproc.postprocess(embedding_batch)
    # print(postprocessed_batch)
    return postprocessed_batch


def average_and_integrate(path_src, path_tgt_avr, path_tgt_itg):
    data1 = []
    data2 = []
    files = os.listdir(path_src)
    for file in files:
        file_path_src = path_src + '/' + file
        postprocessed_batch = extract_features(file_path_src)
        tmp1 = postprocessed_batch.mean(axis=0)
        data1.append(tmp1)
        tmp2 = numpy.array(postprocessed_batch)
        data2.append(tmp2.ravel())
    matrix1 = pandas.DataFrame(data1)
    matrix1.to_csv(path_tgt_avr, sep=',', index=False, header=None)
    matrix2 = pandas.DataFrame(data2)
    matrix2.to_csv(path_tgt_itg, sep=',', index=False, header=None)


if __name__ == "__main__":
    var = "/日用百货/坏的"
    path_src = 'D:/抖音数据/二代数据/音频' + var
    path_tgt_avr = 'D:/抖音数据/二代数据/特征向量' + var + '/平均特征向量.csv'
    path_tgt_itg = 'D:/抖音数据/二代数据/特征向量' + var + '/整合特征向量.csv'
    average_and_integrate(path_src, path_tgt_avr, path_tgt_itg)