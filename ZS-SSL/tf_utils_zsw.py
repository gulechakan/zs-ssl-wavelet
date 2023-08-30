import tensorflow as tf
import math
import os
import parser_ops
import UnrollNetWavelet

parser = parser_ops.get_parser()
args = parser.parse_args()

def test_graph(directory):
    """
    This function creates a test graph for testing
    """

    tf.reset_default_graph()
    # %% placeholders for the unrolled network
    sens_mapsP = tf.placeholder(tf.complex64, shape=(None, args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB), name='sens_maps')
    trn_maskP = tf.placeholder(tf.complex64, shape=(None, args.nrow_GLOB, args.ncol_GLOB), name='trn_mask')
    loss_maskP = tf.placeholder(tf.complex64, shape=(None, args.nrow_GLOB, args.ncol_GLOB), name='loss_mask')
    nw_inputP = tf.placeholder(tf.float32, shape=(None, args.nrow_GLOB, args.ncol_GLOB, 2), name='nw_input')

    #!<< Here, I am not sure about the inputs and outputs, the important part is the inside of UnrolledNet
    nw_output, nw_kspace_output, lam, x0, all_intermediate_outputs, mu = \
               UnrollNetWavelet.UnrolledNet(nw_inputP, sens_mapsP, trn_maskP, loss_maskP).model

    # %% unrolled network outputs
    nw_output = tf.identity(nw_output, name='nw_output')
    nw_kspace_output = tf.identity(nw_kspace_output, name='nw_kspace_output')
    all_intermediate_outputs = tf.identity(all_intermediate_outputs, name='all_intermediate_outputs')
    x0 = tf.identity(x0, name='x0')
    mu = tf.identity(mu, name='mu')
    lam = tf.identity(lam, name='lam')

    # %% saves computational graph for test
    saver = tf.train.Saver()
    sess_test_filename = os.path.join(directory, 'model_test')
    with tf.Session(config=tf.ConfigProto()) as sess:
        sess.run(tf.global_variables_initializer())
        saved_test_model = saver.save(sess, sess_test_filename, latest_filename='checkpoint_test')

    print('\n Test graph is generated and saved at: ' + saved_test_model)

    return True



def tf_real2complex(input_data):
    """
    Parameters
    ----------
    input_data : nrow x ncol x 2

    Returns
    -------
    merges concatenated channels and outputs complex image of size nrow x ncol.

    """

    return tf.complex(input_data[..., 0], input_data[..., 1])



def tf_complex2real(input_data):
    """
    Parameters
    ----------
    input_data : nrow x ncol.

    Returns
    -------
    outputs concatenated real and imaginary parts as nrow x ncol x 2

    """

    return tf.stack([tf.real(input_data), tf.imag(input_data)], axis=-1)