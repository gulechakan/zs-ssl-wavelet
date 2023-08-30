import tensorflow as tf
import data_consistency as ssdu_dc
import tf_utils_zsw
import models.networks as networks
import parser_ops
import dwtcoeffs
import nodes

parser = parser_ops.get_parser()
args = parser.parse_args()

wavenum = 4
subbnad_num = 14
level = 4
wavelet_type1 = dwtcoeffs.db4
wavelet_type2 = dwtcoeffs.db3
wavelet_type3 = dwtcoeffs.db2
wavelet_type4 = dwtcoeffs.haar

class UnrolledNet():
    """

    Parameters
    ----------
    input_x: batch_size x nrow x ncol x 2
    sens_maps: batch_size x ncoil x nrow x ncol

    trn_mask: batch_size x nrow x ncol, used in data consistency units
    loss_mask: batch_size x nrow x ncol, used to define loss in k-space

    args.nb_unroll_blocks: number of unrolled blocks
    #!<< Hakan: args.nb_res_blocks: number of residual blocks in ResNet  //// probably no need for this

    Returns
    ----------

    x: nw output image
    nw_kspace_output: k-space corresponding nw output at loss mask locations

    x0 : dc output without any regularization.
    all_intermediate_results: all intermediate outputs of regularizer and dc units
    mu: learned penalty parameter


    """

    def __init__(self, input_x, sens_maps, trn_mask, loss_mask):
        self.input_x = input_x
        self.sens_maps = sens_maps
        self.trn_mask = trn_mask
        self.loss_mask = loss_mask
        self.model = self.Unrolled_SSDU()

    def Unrolled_SSDU(self):
        
        #!<< Define wavelet coefficients
        d4_coeffs_decomp_hp = tf.constant([-0.010597401784997278,
                                           -0.032883011666982945,
                                           0.030841381835986965,
                                           0.18703481171888114,
                                           -0.02798376941698385,
                                           -0.6308807679295904,
                                           0.7148465705525415,
                                           -0.23037781330885523])

        d4_coeffs_decomp_lp = tf.constant([0.23037781330885523,
                                           0.7148465705525415,
                                           0.6308807679295904,
                                           -0.02798376941698385,
                                           -0.18703481171888114,
                                           0.030841381835986965,
                                           0.032883011666982945,
                                           -0.010597401784997278])

        d4_coeffs_recon_hp = tf.constant([-0.23037781330885523,
                                          0.7148465705525415,
                                          -0.6308807679295904,
                                          -0.02798376941698385,
                                          0.18703481171888114,
                                          0.030841381835986965,
                                          -0.032883011666982945,
                                          -0.010597401784997278])

        d4_coeffs_recon_lp = tf.constant([-0.010597401784997278,
                                          0.032883011666982945,
                                          0.030841381835986965,
                                          -0.18703481171888114,
                                          -0.02798376941698385,
                                          0.6308807679295904,
                                          0.7148465705525415,
                                          0.23037781330885523])

        d3_coeffs_decomp_hp = tf.constant([0.035226291882100656,
                                           0.08544127388224149,
                                           -0.13501102001039084,
                                           -0.4598775021193313,
                                           0.8068915093133388,
                                           -0.3326705529509569])

        d3_coeffs_decomp_lp = tf.constant([0.3326705529509569,
                                           0.8068915093133388,
                                           0.4598775021193313,
                                           -0.13501102001039084,
                                           -0.08544127388224149,
                                           0.035226291882100656])

        d3_coeffs_recon_hp = tf.constant([-0.3326705529509569,
                                          0.8068915093133388,
                                          -0.4598775021193313,
                                          -0.13501102001039084,
                                          0.08544127388224149,
                                          0.035226291882100656])

        d3_coeffs_recon_lp = tf.constant([0.035226291882100656,
                                          -0.08544127388224149,
                                          -0.13501102001039084,
                                          0.4598775021193313,
                                          0.8068915093133388,
                                          0.3326705529509569])

        d2_coeffs_decomp_hp = tf.constant([-0.12940952255092145,
                                           -0.22414386804185735,
                                           0.836516303737469,
                                           -0.48296291314469025])

        d2_coeffs_decomp_lp = tf.constant([0.48296291314469025,
                                           0.836516303737469,
                                           0.22414386804185735,
                                           -0.12940952255092145])

        d2_coeffs_recon_hp = tf.constant([-0.48296291314469025,
                                          0.836516303737469,
                                          -0.22414386804185735,
                                          -0.12940952255092145])

        d2_coeffs_recon_lp = tf.constant([-0.12940952255092145,
                                          0.22414386804185735,
                                          0.836516303737469,
                                          0.48296291314469025])

        haar_coeffs_decomp_hp = tf.constant([0.70710677, -0.70710677])

        haar_coeffs_decomp_lp = tf.constant([0.70710677, 0.70710677])

        haar_coeffs_recon_hp = tf.constant([-0.70710677, 0.70710677])

        haar_coeffs_recon_lp = tf.constant([0.70710677, 0.70710677])
        
        d4_coeffs_decomp_hp = tf.expand_dims(tf.expand_dims(d4_coeffs_decomp_hp, -1), -1)
        d4_coeffs_decomp_lp = tf.expand_dims(tf.expand_dims(d4_coeffs_decomp_lp, -1), -1)
        d4_coeffs_recon_hp = tf.expand_dims(tf.expand_dims(d4_coeffs_recon_hp, -1), -1)
        d4_coeffs_recon_lp = tf.expand_dims(tf.expand_dims(d4_coeffs_recon_lp, -1), -1)

        d3_coeffs_decomp_hp = tf.expand_dims(tf.expand_dims(d3_coeffs_decomp_hp, -1), -1)
        d3_coeffs_decomp_lp = tf.expand_dims(tf.expand_dims(d3_coeffs_decomp_lp, -1), -1)
        d3_coeffs_recon_hp = tf.expand_dims(tf.expand_dims(d3_coeffs_recon_hp, -1), -1)
        d3_coeffs_recon_lp = tf.expand_dims(tf.expand_dims(d3_coeffs_recon_lp, -1), -1)

        d2_coeffs_decomp_hp = tf.expand_dims(tf.expand_dims(d2_coeffs_decomp_hp, -1), -1)
        d2_coeffs_decomp_lp = tf.expand_dims(tf.expand_dims(d2_coeffs_decomp_lp, -1), -1)
        d2_coeffs_recon_hp = tf.expand_dims(tf.expand_dims(d2_coeffs_recon_hp, -1), -1)
        d2_coeffs_recon_lp = tf.expand_dims(tf.expand_dims(d2_coeffs_recon_lp, -1), -1)

        haar_coeffs_decomp_hp = tf.expand_dims(tf.expand_dims(haar_coeffs_decomp_hp, -1), -1)
        haar_coeffs_decomp_lp = tf.expand_dims(tf.expand_dims(haar_coeffs_decomp_lp, -1), -1)
        haar_coeffs_recon_hp = tf.expand_dims(tf.expand_dims(haar_coeffs_recon_hp, -1), -1)
        haar_coeffs_recon_lp = tf.expand_dims(tf.expand_dims(haar_coeffs_recon_lp, -1), -1)
        
        x, denoiser_output, dc_output = self.input_x, self.input_x, self.input_x
        print("fener1")
        all_intermediate_results = [[0 for _ in range(2)] for _ in range(args.nb_unroll_blocks)]
        print("fener2")
        mu_init = tf.constant(0., dtype=tf.float32)
        print("fener3")
        x0 = ssdu_dc.dc_block(self.input_x, self.sens_maps, self.trn_mask, mu_init)
        print("fener10")
        beta = [[tf.zeros((args.nrow_GLOB, args.ncol_GLOB), dtype=tf.complex64) for _ in range(args.batchSize)] for _ in range(wavenum)]
        v = [[tf.zeros((args.nrow_GLOB, args.ncol_GLOB), dtype=tf.complex64) for _ in range(args.batchSize)] for _ in range(wavenum)]

        umaxd4 = calc_threshold(x0, d4_coeffs_decomp_hp, d4_coeffs_decomp_lp, wavelet_type1)
        umaxd3 = calc_threshold(x0, d3_coeffs_decomp_hp, d3_coeffs_decomp_lp, wavelet_type2)
        umaxd2 = calc_threshold(x0, d2_coeffs_decomp_hp, d2_coeffs_decomp_lp, wavelet_type3)
        umaxd1 = calc_threshold(x0, haar_coeffs_decomp_hp, haar_coeffs_decomp_lp, wavelet_type4)

        with tf.name_scope('SSDUModel'):
            with tf.variable_scope('Weights', reuse=tf.AUTO_REUSE):
                for i in range(args.nb_unroll_blocks):
                    # regularization and dual update
                    if i == 0:
                        etta = tf.constant(0., shape=(wavenum, 1), dtype=tf.float32)
                    else:
                        etta = getEtta()

                    # z1
                    [z1, beta, v] = Z_M_steps(umaxd4, x, d4_coeffs_decomp_lp,
                                                         d4_coeffs_decomp_hp, d4_coeffs_recon_lp, d4_coeffs_recon_hp,
                                                         wavelet_type1, beta, etta, v, i, 0)

                    # z2
                    [z2, beta, v] = Z_M_steps(umaxd3, x, d3_coeffs_decomp_lp,
                                                         d3_coeffs_decomp_hp,
                                                         d3_coeffs_recon_lp, d3_coeffs_recon_hp,
                                                         wavelet_type2, beta, etta, v, i, 1)

                    # z3
                    [z3, beta, v] = Z_M_steps(umaxd2, x, d2_coeffs_decomp_lp,
                                                         d2_coeffs_decomp_hp,
                                                         d2_coeffs_recon_lp, d2_coeffs_recon_hp,
                                                         wavelet_type3, beta, etta, v, i, 2)

                    # z4
                    [z4, beta, v] = Z_M_steps(umaxd1, x, haar_coeffs_decomp_lp,
                                                         haar_coeffs_decomp_hp,
                                                         haar_coeffs_recon_lp, haar_coeffs_recon_hp,
                                                         wavelet_type4, beta, etta, v, i, 3)
                    
                    lam = getLambda()

                    denoiser_output = lam[0] * z1 + lam[1] * z2 + lam[2] * z3 + lam[3] * z4

                    # rhs = self.input_x + denoiser_output
                    
                    # x = networks.ResNet(x, args.nb_res_blocks)
                    # denoiser_output = x
                    mu = networks.mu_param()
                    rhs = self.input_x + mu * denoiser_output

                    x = ssdu_dc.dc_block(rhs, self.sens_maps, self.trn_mask, mu)
                    dc_output = x

                    # ...................................................................................................
                    all_intermediate_results[i][0] = tf_utils_zsw.tf_real2complex(tf.squeeze(denoiser_output))
                    all_intermediate_results[i][1] = tf_utils_zsw.tf_real2complex(tf.squeeze(dc_output))

            nw_kspace_output = ssdu_dc.SSDU_kspace_transform(x, self.sens_maps, self.loss_mask)

        return x, nw_kspace_output, lam, x0, all_intermediate_results, mu
    
    
    
    
def calc_threshold(x, hp, lp, wavelet_type):
    """
        This function gets inf norm of each subband of W_l*x
    """
    umax_list = [[0 for _ in range(subbnad_num)] for _ in range(args.batchSize)]
    for j in range(args.batchSize):
        u = calc_transform(x, hp, lp, wavelet_type, j)
        u_list = get_u_subbands(u)

        for k in range(subbnad_num):
            umax_list[j][k] = tf.math.reduce_max(tf.math.abs(u_list[k]))

    return umax_list



def calc_transform(x, hp, lp, wavelet_type, j):
    input_signal1 = tf.expand_dims(x[j, :, :, 0], -1)
    input_signal2 = tf.expand_dims(x[j, :, :, 1], -1)

    wave1 = nodes.dwt2d(input_signal1, lp, hp, wavelet=wavelet_type,
                        levels=level)
    wave2 = nodes.dwt2d(input_signal2, lp, hp, wavelet=wavelet_type,
                        levels=level)
    u = tf.squeeze(tf_utils_zsw.tf_real2complex(tf.stack([wave1, wave2], axis=-1)))

    return u


def get_u_subbands(u):
    u_list = [0 for _ in range(subbnad_num)]
    temp_shape0 = u.shape[0]
    temp_shape1 = u.shape[1]

    for k in range(level):
        [u_list[3 * k + 0], u_list[3 * k + 1], u_list[3 * k + 2], temp_shape0, temp_shape1] = subband_procesing(u, temp_shape0, temp_shape1)
    u_list[12] = u[0: temp_shape0 // 2, 0: temp_shape1]
    u_list[13] = u[temp_shape0 // 2: temp_shape0, 0: temp_shape1]

    return u_list



def Z_M_steps(umax, x, coeffs_decomp_lp, coeffs_decomp_hp, coeffs_recon_lp, coeffs_recon_hp, wavelet_type, beta, etta, v, i, l):
    """
       This function performs thresholding in wavelet domain (regularization) and dual updates
    """
    zl_returned = [0 for _ in range(args.batchSize)]
    for j in range(args.batchSize):
        beta_temp = beta[l][j]
        v_temp = v[l][j]

        u = calc_transform(x, coeffs_decomp_hp, coeffs_decomp_lp, wavelet_type, j)

        if i != 0:
            temp = tf_utils_zsw.tf_complex2real(u - v_temp)
            beta_temp = beta_temp + tf.squeeze(tf_utils_zsw.tf_real2complex(tf.stack([tf.math.multiply(etta[l], temp[...,0]), tf.math.multiply(etta[l], temp[...,1])], axis=-1)))
        u = u + beta_temp

        u_list = get_u_subbands(u)

        scaling_factor = getScaling_factor()

        for k in range(subbnad_num):
            u_list[k] = subband_threshold(u_list[k], umax[j][k], scaling_factor[k, l])

        v_temp = tf.concat([u_list[12], u_list[13]], axis=0)

        for k in range(level):
            v_temp = subband_reformation(v_temp, u_list[11-3*k], u_list[10-3*k], u_list[9-3*k])

        input_to_X = tf_utils_zsw.tf_complex2real(v_temp - beta_temp)

        input_signal3 = tf.expand_dims(input_to_X[...,0], -1)
        input_signal4 = tf.expand_dims(input_to_X[...,1], -1)

        img3 = nodes.idwt2d(input_signal3, coeffs_recon_lp, coeffs_recon_hp, wavelet=wavelet_type,
                                 levels=level)
        img4 = nodes.idwt2d(input_signal4, coeffs_recon_lp, coeffs_recon_hp, wavelet=wavelet_type,
                                 levels=level)

        x_new = tf.concat([img3, img4], axis=-1)

        beta[l][j] = beta_temp
        v[l][j] = v_temp

        if j == 0:
            zl_returned = tf.expand_dims(x_new, 0)
        else:
            zl_returned = tf.concat([zl_returned, tf.expand_dims(x_new, 0)], axis=0)

    return zl_returned, beta, v



def subband_procesing(u, temp_shape0, temp_shape1):
    prev_temp_shape0 = temp_shape0
    prev_temp_shape1 = temp_shape1
    temp_shape0 = temp_shape0 // 2
    temp_shape1 = temp_shape1 // 2

    u1 = u[temp_shape0: prev_temp_shape0, temp_shape1: prev_temp_shape1]
    u2 = u[0: temp_shape0, temp_shape1: prev_temp_shape1]
    u3 = u[temp_shape0: prev_temp_shape0, 0: temp_shape1]

    return u1, u2, u3, temp_shape0, temp_shape1



def getEtta():
    """
    create a shared variable (across unrolled iterations) called etta.
    """
    with tf.variable_scope('Etta'):
        etta = tf.get_variable(name='etta', shape=(wavenum), dtype=tf.float32, initializer=tf.random_normal_initializer(-5, 0.2))
    return tf.nn.sigmoid(etta)



def getScaling_factor():
    """
    create a shared variable (across unrolled iterations) called scaling_factor.
    """
    with tf.variable_scope('Scaling_factor'):
        scaling_factor = tf.get_variable(name='Scaling_factor', shape=(subbnad_num, wavenum), dtype=tf.float32, initializer=tf.random_normal_initializer(-5, 0.2))
    return tf.nn.sigmoid(scaling_factor)



def subband_threshold(ui, umax, scaling_factor):
    T = tf.math.multiply(scaling_factor, umax)
    ui = soft_threshold(ui, T)

    return ui



def soft_threshold(z, T):
    """
        This function performs soft thresholding with parameter T for complex values
    """
    size_z = z.shape

    z = tf.reshape(z, [-1])

    # intermediate values
    a = tf.math.maximum(tf.math.abs(z) - T, 0)
    b = a + T
    c = tf.math.divide(a, b)
    d = tf.math.multiply(tf.cast(c, tf.complex64), z)

    sz = tf.reshape(d, size_z)

    return sz



def subband_reformation(temp, u3, u2, u1):
    temp1 = tf.concat([temp, u3], axis=0)
    temp2 = tf.concat([u2, u1], axis=0)

    return tf.concat([temp1, temp2], axis=1)



def getLambda():
    """
    create a shared variable (across unrolled iterations) called lambda.
    """
    with tf.variable_scope('Lambda'):
        lam = tf.get_variable(name='lam', shape=(wavenum), dtype=tf.float32, initializer=tf.random_normal_initializer(-5, 0.2))
    return tf.nn.sigmoid(lam)