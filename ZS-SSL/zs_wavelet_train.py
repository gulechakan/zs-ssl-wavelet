import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
from datetime import datetime
import os
import utils
import tf_utils_zsw
import parser_ops
import UnrollNetWavelet

parser = parser_ops.get_parser()
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #set it to available GPU

save_dir ='saved_models'
directory = os.path.join(save_dir, 'ZS_SSL_' + args.data_opt + '_Rate'+ str(args.acc_rate)+'_'+ str(args.num_reps)+'reps')
if not os.path.exists(directory):
    os.makedirs(directory)
    
#!<< For here, organize the inside of test graph - UnrolledNet part
print('create a test model for the testing')
test_graph_generator = tf_utils_zsw.test_graph(directory)

#................................................................................
start_time = time.time()
print('.................ZS-SSL Training.....................')
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

# .......................Load the Data..........................................
print('Loading data  for training............... ')
data = sio.loadmat(args.data_dir) 
kspace_train,sens_maps, original_mask= data['kspace'], data['sens_maps'], data['mask']
args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB  = kspace_train.shape

print('Normalize the kspace to 0-1 region')
kspace_train= kspace_train / np.max(np.abs(kspace_train[:]))

#..................Generate validation mask.....................................
cv_trn_mask, cv_val_mask = utils.uniform_selection(kspace_train,original_mask, rho=args.rho_val)
remainder_mask, cv_val_mask=np.copy(cv_trn_mask),np.copy(np.complex64(cv_val_mask))

#!<< kspace and sensitivity maps have size of nSlices x nrow x ncol x ncoil 
#!<< mask has size of nrow x ncol
#!<< original mask is the \Omega
#!<< remainder_mask is theta and lambda
#!<< cv_val_mask is T

print('size of kspace: ', kspace_train[np.newaxis,...].shape, ', maps: ', sens_maps.shape, ', mask: ', original_mask.shape)

trn_mask, loss_mask = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64), \
                                np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

#!<< trn_mask is theta
#!<< loss_mask is lambda

# train data
nw_input = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
ref_kspace = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)
#...............................................................................
# validation data
ref_kspace_val = np.empty((args.num_reps,args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)
nw_input_val = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

#!<< sub_kspace is the theta kspace which is splitted from the acquired undersampled kspace data (training k-space)
#!<< ref_kspace is the loss metric (in my terms) which is the greek letter resembling A
#!<< nw_input is the image of the sub-kspace theta (SENSE image) 

#!<< ref_kspace_val is the validation loss metric
#!<< nw_input_val is the sense image of kspace_train * cv_trn_mask
#!<< cv_trn_mask is omega\T

print('create training&loss masks and generate network inputs... ')
#train data
for jj in range(args.num_reps):
    trn_mask[jj, ...], loss_mask[jj, ...] = utils.uniform_selection(kspace_train,remainder_mask, rho=args.rho_train)

    sub_kspace = kspace_train * np.tile(trn_mask[jj][..., np.newaxis], (1, 1, args.ncoil_GLOB))
    ref_kspace[jj, ...] = kspace_train * np.tile(loss_mask[jj][..., np.newaxis], (1, 1, args.ncoil_GLOB))
    nw_input[jj, ...] = utils.sense1(sub_kspace,sens_maps)

#..............................validation data.....................................
nw_input_val = utils.sense1(kspace_train * np.tile(cv_trn_mask[:, :, np.newaxis], (1, 1, args.ncoil_GLOB)),sens_maps)[np.newaxis]
ref_kspace_val=kspace_train*np.tile(cv_val_mask[:, :, np.newaxis], (1, 1, args.ncoil_GLOB))[np.newaxis]

# %% Prepare the data for the training
sens_maps = np.tile(sens_maps[np.newaxis],(args.num_reps,1,1,1))
sens_maps = np.transpose(sens_maps, (0, 3, 1, 2))
ref_kspace = utils.complex2real(np.transpose(ref_kspace, (0, 3, 1, 2)))
nw_input = utils.complex2real(nw_input)

# %% validation data 
ref_kspace_val = utils.complex2real(np.transpose(ref_kspace_val, (0, 3, 1, 2)))
nw_input_val = utils.complex2real(nw_input_val)

print('size of ref kspace: ', ref_kspace.shape, ', nw_input: ', nw_input.shape, ', maps: ', sens_maps.shape, ', mask: ', trn_mask.shape)

# %% set the batch size
total_batch = int(np.floor(np.float32(nw_input.shape[0]) / (args.batchSize)))
print("hakan1")
kspaceP = tf.placeholder(tf.float32, shape=(None, None, None, None, 2), name='refkspace')
print("hakan2")
sens_mapsP = tf.placeholder(tf.complex64, shape=(None, None, None, None), name='sens_maps')
print("hakan3")
trn_maskP = tf.placeholder(tf.complex64, shape=(None, None, None), name='trn_mask')
print("hakan4")
loss_maskP = tf.placeholder(tf.complex64, shape=(None, None, None), name='loss_mask')
print("hakan5")
nw_inputP = tf.placeholder(tf.float32, shape=(None, args.nrow_GLOB, args.ncol_GLOB, 2), name='nw_input')
print("hakan6")
# %% creating the dataset
train_dataset = tf.data.Dataset.from_tensor_slices((kspaceP,nw_inputP,sens_mapsP,trn_maskP,loss_maskP)).shuffle(buffer_size= 10*args.batchSize).batch(args.batchSize)
print("hakan7")
cv_dataset = tf.data.Dataset.from_tensor_slices((kspaceP,nw_inputP,sens_mapsP,trn_maskP,loss_maskP)).shuffle(buffer_size=10*args.batchSize).batch(args.batchSize)
print("hakan8")
iterator=tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
print("hakan9")
train_iterator=iterator.make_initializer(train_dataset)
print("hakan10")
cv_iterator = iterator.make_initializer(cv_dataset)
print("hakan11")
ref_kspace_tensor,nw_input_tensor,sens_maps_tensor,trn_mask_tensor,loss_mask_tensor = iterator.get_next('getNext')
print("hakan12")

#%% make training model
nw_output_img, nw_output_kspace, *_ = UnrollNetWavelet.UnrolledNet(nw_input_tensor, sens_maps_tensor, trn_mask_tensor, loss_mask_tensor).model
print("hakan13")
scalar = tf.constant(0.5, dtype=tf.float32)
loss = tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace) / tf.norm(ref_kspace_tensor)) + \
       tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace, ord=1) / tf.norm(ref_kspace_tensor, ord=1))

all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

saver = tf.train.Saver(max_to_keep=1) #only keep the model corresponding to lowest validation error
sess_trn_filename = os.path.join(directory, 'model')
totalLoss,totalTime=[],[]
total_val_loss = []
avg_cost = 0
print('training......................................................')
lowest_val_loss = np.inf
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print('Number of trainable parameters: ', sess.run(all_trainable_vars))
    feedDict = {kspaceP: ref_kspace, nw_inputP: nw_input, trn_maskP: trn_mask, loss_maskP: loss_mask, sens_mapsP: sens_maps}

    print('Training...')
    # if for args.stop_training consecutive epochs validation loss doesnt go below the lowest val loss,\
    #  stop the training
    
    ep, val_loss_tracker = 0, 0 
    while ep<args.epochs and val_loss_tracker<args.stop_training:
        sess.run(train_iterator, feed_dict=feedDict)
        avg_cost = 0
        tic = time.time()
        try:
            for jj in range(total_batch):
                tmp, _, _ = sess.run([loss, update_ops, optimizer])
                avg_cost += tmp / total_batch    
            toc = time.time() - tic
            totalLoss.append(avg_cost)
        except tf.errors.OutOfRangeError:
            pass
        #%%..................................................................
        # perform validation
        sess.run(cv_iterator, feed_dict={kspaceP: ref_kspace_val, nw_inputP: nw_input_val, trn_maskP: cv_trn_mask[np.newaxis], loss_maskP: cv_val_mask[np.newaxis], sens_mapsP: sens_maps[0][np.newaxis]})
        val_loss = sess.run([loss])[0]
        total_val_loss.append(val_loss)
        # ..........................................................................................................
        print("Epoch:", ep, "elapsed_time =""{:f}".format(toc), "trn loss =", "{:.5f}".format(avg_cost)," val loss =", "{:.5f}".format(val_loss))        
        if val_loss<=lowest_val_loss:
            lowest_val_loss = val_loss    
            saver.save(sess, sess_trn_filename, global_step=ep)
            val_loss_tracker = 0 #reset the val loss tracker each time a new lowest val error is achieved
        else:
            val_loss_tracker += 1
        sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'trn_loss': totalLoss, 'val_loss': total_val_loss})
        ep += 1
    
end_time = time.time()
print('Training completed in  ', str(ep), ' epochs, ',((end_time - start_time) / 60), ' minutes')






