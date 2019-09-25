import sys
import tensorflow as tf
import numpy as np
import random
import datetime
import os
import argparse
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Import Network and preprocessing modules from slim
slim_path = 'slim/'
sys.path.insert(0,slim_path)
from nets import nets_factory
from preprocessing import preprocessing_factory

# Fix all the random seeds
np.random.seed(1)
tf.set_random_seed(1)
random.seed(1)


def validate_arguments(args):
    nets = ['vgg_16', 'vgg_19', 'inception_v3',  'resnet_v1_152', \
            'inception_v1', 'resnet_v1_50']

    if not(args.model_name in nets):
        print ('invalid network')
        exit(-1)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='vgg_16', help='Model Name')
args = parser.parse_args()
validate_arguments(args)
    
model_name = args.model_name
batch_size = 1
expt_name = 'uap_no_data'


# Create directories for saving perturbations
path2results = 'results'
if not os.path.exists(path2results):
    os.mkdir(path2results)
if not os.path.exists(path2results + '/' + model_name):
    os.mkdir(path2results + '/' + model_name)
if not os.path.exists(path2results + '/' + model_name + '/' + expt_name):
    os.mkdir(path2results + '/' + model_name + '/' + expt_name)


# Path to pretrained checkpoint of the model
pretrained_checkpoint = 'model_wts/'+model_name+'.ckpt'

# Number of neurons in the final layer of the model
no_classes = {'inception_v3' : 1001, 'vgg_16':1000, 'vgg_19':1000, \
            'resnet_v1_152':1000, 'resnet_v1_50':1000, 'inception_v1' : 1001}
# Learning rate for crafting perturbation
learning_rate = {'inception_v3' : 1e-3, 'vgg_16':1e-1, 'vgg_19':1e-1, \
            'resnet_v1_152':1e-1, 'resnet_v1_50':1e-1, 'inception_v1' : 1e-3}
# Flag to specify if input images need to be normalized 
normalize_ip = {'inception_v3' : True, 'vgg_16':False, 'vgg_19':False, \
            'resnet_v1_152':False, 'resnet_v1_50':False, 'inception_v1' : True}
# Perceptibility constraint
threshold = {'inception_v3' : 20.0/255, 'vgg_16':10.0, 'vgg_19':10.0, \
            'resnet_v1_152':10.0, 'resnet_v1_50':10.0, 'inception_v1' : 20.0/255}

num_classes = no_classes[model_name]

# setup session
config = tf.ConfigProto(intra_op_parallelism_threads=1, \
            inter_op_parallelism_threads=1, \
            gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)

# load networks function. 
network_fn = nets_factory.get_network_fn(model_name,\
                num_classes=(num_classes),is_training=False)

# set up inputs
eval_image_size = network_fn.default_image_size
# initializing adversarial image
noise_image = tf.Variable(tf.random_uniform(
                    [1, eval_image_size, eval_image_size, 3],
                    -threshold[model_name],threshold[model_name] \
                    , dtype='float32'), dtype='float32', name='noise_image')
# clipping for imperceptibility constraint
noise_image_clipped = tf.clip_by_value(noise_image, \
                    -threshold[model_name], threshold[model_name])

net_input = noise_image_clipped
# setup model
logits, normal_network = network_fn(net_input)

sess.run(tf.global_variables_initializer())
# Load pretrained weights
net_varlist = [v for v in tf.get_collection(tf.GraphKeys.VARIABLES)][1:]
saver = tf.train.Saver(var_list=net_varlist)
saver.restore(sess, pretrained_checkpoint)


# select pre ReLU tensors for optimization
tf_ops = tf.get_default_graph().get_operations()

if(model_name in ['vgg_16', 'vgg_19']):
    opt_layers = [i.inputs[0] for i in tf_ops if i.type=="Relu"]
elif(model_name=='resnet_v1_152'):
    y = [x for x in tf_ops if x.type=='Relu']
    opt_layers = [i.inputs[0] for i in y \
                if i.name.split('/')[-2]=='bottleneck_v1' \
                or i.name == 'resnet_v1_152/conv1/Relu']
elif(model_name=='resnet_v1_50'):
    y = [x for x in tf_ops if x.type=='Relu']
    opt_layers = [i.inputs[0] for i in y \
                if i.name.split('/')[-2]=='bottleneck_v1' \
                or i.name == 'resnet_v1_50/conv1/Relu']
elif(model_name=='inception_v3'):
    y = [x for x in tf_ops if x.type in ['Relu', 'MaxPool']]
    opt_ops = [['Conv2d_2b_3x3']
       ,['Conv2d_4a_3x3']
       ,['Mixed_5b/Branch_0/Conv2d_0a_1x1','Mixed_5b/Branch_1/Conv2d_0b_5x5'\
       ,'Mixed_5b/Branch_2/Conv2d_0c_3x3','Mixed_5b/Branch_3/Conv2d_0b_1x1']
       ,['Mixed_5c/Branch_0/Conv2d_0a_1x1','Mixed_5c/Branch_1/Conv_1_0c_5x5'\
       ,'Mixed_5c/Branch_2/Conv2d_0c_3x3','Mixed_5c/Branch_3/Conv2d_0b_1x1']
       ,['Mixed_5d/Branch_0/Conv2d_0a_1x1','Mixed_5d/Branch_1/Conv2d_0b_5x5'\
       ,'Mixed_5d/Branch_2/Conv2d_0c_3x3','Mixed_5d/Branch_3/Conv2d_0b_1x1']
       ,['Mixed_6a/Branch_0/Conv2d_1a_1x1','Mixed_6a/Branch_1/Conv2d_1a_1x1'\
       ,'Mixed_6a/Branch_2/MaxPool_1a_3x3']
       ,['Mixed_6b/Branch_0/Conv2d_0a_1x1','Mixed_6b/Branch_1/Conv2d_0c_7x1'\
       ,'Mixed_6b/Branch_2/Conv2d_0e_1x7','Mixed_6b/Branch_3/Conv2d_0b_1x1']
       ,['Mixed_6c/Branch_0/Conv2d_0a_1x1','Mixed_6c/Branch_1/Conv2d_0c_7x1'\
       ,'Mixed_6c/Branch_2/Conv2d_0e_1x7','Mixed_6c/Branch_3/Conv2d_0b_1x1']
       ,['Mixed_6d/Branch_0/Conv2d_0a_1x1','Mixed_6d/Branch_1/Conv2d_0c_7x1'\
       ,'Mixed_6d/Branch_2/Conv2d_0e_1x7','Mixed_6d/Branch_3/Conv2d_0b_1x1']
       ,['Mixed_6e/Branch_0/Conv2d_0a_1x1','Mixed_6e/Branch_1/Conv2d_0c_7x1'\
       ,'Mixed_6e/Branch_2/Conv2d_0e_1x7','Mixed_6e/Branch_3/Conv2d_0b_1x1']
       ,['Mixed_7a/Branch_0/Conv2d_1a_3x3','Mixed_7a/Branch_1/Conv2d_1a_3x3'\
       ,'Mixed_7a/Branch_2/MaxPool_1a_3x3']
       ,['Mixed_7b/Branch_0/Conv2d_0a_1x1','Mixed_7b/Branch_1/Conv2d_0b_1x3'\
       ,'Mixed_7b/Branch_1/Conv2d_0b_3x1','Mixed_7b/Branch_2/Conv2d_0c_1x3'\
       ,'Mixed_7b/Branch_2/Conv2d_0d_3x1','Mixed_7b/Branch_3/Conv2d_0b_1x1']
       ,['Mixed_7c/Branch_0/Conv2d_0a_1x1','Mixed_7c/Branch_1/Conv2d_0b_1x3'\
       ,'Mixed_7c/Branch_1/Conv2d_0c_3x1','Mixed_7c/Branch_2/Conv2d_0c_1x3'\
       ,'Mixed_7c/Branch_2/Conv2d_0d_3x1','Mixed_7c/Branch_3/Conv2d_0b_1x1']
      ]
    opt_layers = []
    for opts in opt_ops:
        opt_tensors = []
        for op_name in opts:
            xx = [x for x in y if op_name in x.name][0]
            if(xx.type=='Relu'):
                opt_tensors.extend([xx.inputs[0]])
            elif(xx.type=='MaxPool'):
                opt_tensors.extend([xx.outputs[0]])
        opt_tensor = tf.concat(axis=3, values=opt_tensors)
        opt_layers.append(opt_tensor)
elif(model_name=='inception_v1'):
    y = [x for x in tf_ops if x.type in ['Relu']]
    opt_ops = [
        ['Conv2d_1a_7x7'],
        ['Conv2d_2b_1x1'],
        ['Conv2d_2c_3x3'],
        ['Mixed_3b/Branch_0/Conv2d_0a_1x1','Mixed_3b/Branch_1/Conv2d_0b_3x3'\
        ,'Mixed_3b/Branch_2/Conv2d_0b_3x3','Mixed_3b/Branch_3/Conv2d_0b_1x1'],
        ['Mixed_3c/Branch_0/Conv2d_0a_1x1','Mixed_3c/Branch_1/Conv2d_0b_3x3'\
        ,'Mixed_3c/Branch_2/Conv2d_0b_3x3','Mixed_3c/Branch_3/Conv2d_0b_1x1'],
        ['Mixed_4b/Branch_0/Conv2d_0a_1x1','Mixed_4b/Branch_1/Conv2d_0b_3x3'\
        ,'Mixed_4b/Branch_2/Conv2d_0b_3x3','Mixed_4b/Branch_3/Conv2d_0b_1x1'],
        ['Mixed_4c/Branch_0/Conv2d_0a_1x1','Mixed_4c/Branch_1/Conv2d_0b_3x3'\
        ,'Mixed_4c/Branch_2/Conv2d_0b_3x3','Mixed_4c/Branch_3/Conv2d_0b_1x1'],
        ['Mixed_4d/Branch_0/Conv2d_0a_1x1','Mixed_4d/Branch_1/Conv2d_0b_3x3'\
        ,'Mixed_4d/Branch_2/Conv2d_0b_3x3','Mixed_4d/Branch_3/Conv2d_0b_1x1'],
        ['Mixed_4e/Branch_0/Conv2d_0a_1x1','Mixed_4e/Branch_1/Conv2d_0b_3x3'\
        ,'Mixed_4e/Branch_2/Conv2d_0b_3x3','Mixed_4e/Branch_3/Conv2d_0b_1x1'],
        ['Mixed_4f/Branch_0/Conv2d_0a_1x1','Mixed_4f/Branch_1/Conv2d_0b_3x3'\
        ,'Mixed_4f/Branch_2/Conv2d_0b_3x3','Mixed_4f/Branch_3/Conv2d_0b_1x1'],
        ['Mixed_5b/Branch_0/Conv2d_0a_1x1','Mixed_5b/Branch_1/Conv2d_0b_3x3'\
        ,'Mixed_5b/Branch_2/Conv2d_0a_3x3','Mixed_5b/Branch_3/Conv2d_0b_1x1'],
        ['Mixed_5c/Branch_0/Conv2d_0a_1x1','Mixed_5c/Branch_1/Conv2d_0b_3x3'\
        ,'Mixed_5c/Branch_2/Conv2d_0b_3x3','Mixed_5c/Branch_3/Conv2d_0b_1x1'],
    ]
    opt_layers = []
    for opts in opt_ops:
        opt_tensors = []
        for op_name in opts:
            xx = [x for x in y if op_name in x.name][0]
            opt_tensors.extend([xx.inputs[0]])
        opt_tensor = tf.concat(axis=3, values=opt_tensors)
        opt_layers.append(opt_tensor)

print(opt_layers)

# setup optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate[model_name])

losses = []
updates = []
loss = 0
for tensor in opt_layers:
    # dilate loss at each layer
    loss_l = tf.log(tf.nn.l2_loss(tensor))
    loss += loss_l
    ct_loss = -loss
    losses.append(ct_loss)
    grads = optimizer.compute_gradients(ct_loss, [noise_image])
    update = optimizer.apply_gradients(grads)
    updates.append(update)
    
psm_loss = tf.log(tf.nn.l2_loss(logits))
lall = (loss + psm_loss)
loss_logits = -lall
grads = optimizer.compute_gradients(loss_logits, [noise_image])
update = optimizer.apply_gradients(grads)
losses.append(loss_logits)
updates.append(update)

# initialize optimizer variables
opt_vars = [optimizer.get_slot(var, name) \
            for name in optimizer.get_slot_names() for var in [noise_image]]
opt_vars.extend(list(optimizer._get_beta_accumulators()))
opt_init = tf.variables_initializer(opt_vars)
sess.run(opt_init)
# tensor operation for recaling by 2
rescale_op = noise_image.assign(tf.divide(noise_image, 2.0))

# Saturation Measure
saturation = tf.div(tf.reduce_sum(tf.to_float(
        tf.equal(tf.abs(noise_image_clipped), threshold[model_name]))), \
        tf.to_float(tf.size(noise_image_clipped)))


max_iter = 2000

#################################################################################
#   SEQUENTIAL DILATION ALGORITHM                                               #
#################################################################################
# optimize for dilate loss at all layers
for opt_layer in range(len(losses)):
    print('*** ' +str(datetime.datetime.now())+ ": " + str(opt_layer) + ' ***')
    loss_hist = []

    # Rescale noise image after every optimization
    if(opt_layer!=0):
        sess.run(rescale_op)
        print(sat, " Rescaled \n")
    # optimize for dilate loss at current layer
    for i in range(max_iter):
        _, loss_np = sess.run([updates[opt_layer], losses[opt_layer]])
        loss_hist.append(loss_np)
#################################################################################

        sat = sess.run(saturation)
        if(i%100 == 0):
            print("Iter " + str(i) + ": Loss " + str(loss_np) + \
                    ", sat : "+str(sat))

    plt.plot(loss_hist)
    plt.savefig(path2results + '/' + model_name + '/' + expt_name + '/loss_'+str(opt_layer)+'.png')
    plt.clf()
    eta = sess.run(noise_image)
    plt.imshow(eta[0])
    plt.savefig(path2results + '/' + model_name + '/' + expt_name + '/'+str(opt_layer)+'.png')
    plt.clf()
    np.save(path2results + '/' + model_name + '/' + expt_name + '/'+str(opt_layer)+'.npy', eta)

eta = sess.run(noise_image)
plt.imshow(eta[0])
plt.savefig(path2results + '/' + model_name + '/' + expt_name + '/final.png')
plt.clf()
np.save(path2results + '/' + model_name + '/' + expt_name + '/final.npy', eta)
print('####### '+str(datetime.datetime.now()) + ' End #######')