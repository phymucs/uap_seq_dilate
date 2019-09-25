import sys
import tensorflow as tf
import numpy as np
import argparse

# Import Network and preprocessing modules from slim
slim_path = 'slim/'
sys.path.insert(0,slim_path)
from nets import nets_factory
from preprocessing import preprocessing_factory

# Mean from ILSVRC dataset
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
means = np.array([_R_MEAN, _G_MEAN, _B_MEAN])
# min and max means post mean subtraction
min_pix = 0.0-means
max_pix = 255.0-means

parser = argparse.ArgumentParser()
parser.add_argument(
    '--expt', default='uap_no_data',  help='Experiment Name')
parser.add_argument(
    '--model', default='vgg_16',  help='Model Name')
parser.add_argument('--batch_size', default=100,  
                    help='Batch Size')
parser.add_argument('--test_file', default='data/ilsvrc_test.txt',
                    help='Path to the file containing test images')
args = parser.parse_args()

expt_name = args.expt
model_name = args.model
batch_size = int(args.batch_size)
test_file = args.test_file

# path to the file containing ground truth labels
gt_file = "data/ilsvrc_test_gt.txt"

# Path to pretrained checkpoint of the model
pretrained_checkpoint = 'model_wts/'+model_name+'.ckpt'

# Number of neurons in the final layer of the model
no_classes = {'inception_v3' : 1001, 'vgg_16':1000, 'vgg_19':1000, \
        'resnet_v1_152':1000, 'resnet_v1_50':1000, 'inception_v1' : 1001}
# Flag to specify if input images need to be normalized 
normalize_ip = {'inception_v3' : True, 'vgg_16':False, 'vgg_19':False, \
        'resnet_v1_152':False, 'resnet_v1_50':False, 'inception_v1' : True}
# Perceptibility constraint
threshold = {'inception_v3' : 20.0/255, 'vgg_16':10.0, 'vgg_19':10.0, \
        'resnet_v1_152':10.0, 'resnet_v1_50':10.0, 'inception_v1' : 20.0/255}

num_classes = no_classes[model_name]

class DataLoader():

    def __init__(self, image_preprocessing_fn, image_size, batch_size, \
                    normalize = True):
        # Placeholder to pass list of image input paths
        self.image_path = tf.placeholder(tf.string, [None])
        img_list = []
        for i in range(batch_size):
            # read image
            file_data = tf.read_file(self.image_path[i])
            # Decode the image data
            img = tf.image.decode_jpeg(file_data, channels=3)
            img = tf.to_float(img)
            # Inception models take input in [0,1] range
            if(normalize == True):
                img = img/255.0
            # Do model specific input preprocessing
            image = image_preprocessing_fn(img, image_size, image_size)
            img_list.append(image)
        
        self.input_images = tf.stack(img_list, 0)

# setup session
config = tf.ConfigProto(intra_op_parallelism_threads=1, \
            inter_op_parallelism_threads=1, \
            gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)

# load perturbation
pert_name = 'results/'+model_name+'/'+expt_name+'/final.npy'
eta = np.load(pert_name)
# clipping for imperceptibility constraint
eta = np.clip(eta, -threshold[model_name],threshold[model_name])
noise_tensor = tf.constant(eta)


# load networks function. 
network_fn = nets_factory.get_network_fn(model_name,\
                num_classes=(num_classes),is_training=False)

# set up inputs
eval_image_size = network_fn.default_image_size
image_preprocessing_fn = preprocessing_factory.get_preprocessing(\
                            model_name,is_training=False)
data_loader = DataLoader(image_preprocessing_fn, eval_image_size, \
                            batch_size, normalize=normalize_ip[model_name])
net_inp = tf.concat([data_loader.input_images, \
                    data_loader.input_images+noise_tensor], 0)

# clip inputs with perturbation to normal image range
if normalize_ip[model_name]==True:
    net_inp = tf.clip_by_value(net_inp, -1.0, 1.0)
else:
    num_channels = 3
    channels = tf.split(net_inp, axis=3, num_or_size_splits=num_channels)
    for i in range(num_channels):
        channels[i] = tf.clip_by_value(channels[i], min_pix[i], max_pix[i])
    net_inp = tf.concat(axis=3, values=channels)

# setup model
logits, normal_network = network_fn(net_inp)

# Load pretrained weights
net_varlist = [v for v in tf.get_collection(tf.GraphKeys.VARIABLES)]
saver = tf.train.Saver(var_list=net_varlist)
saver.restore(sess, pretrained_checkpoint)

# load test images list and their ground truths
image_list = image_list = open(test_file).readlines()
image_list = [x.strip() for x in image_list]
gt_list = open(gt_file).readlines()
gt_list = np.array([int(x.strip()) for x in gt_list], dtype='int32')

no_correct = 0.0 # number of correctly predicted normal images
acc_post = 0.0 # number of correctly predicted adversarial images
no_fooled = 0.0 # number of fooled images
no_images = len(image_list) # total number of images in test set
no_batch = int(np.floor(no_images/batch_size)) #no of batches
print("No of Batches : ", no_batch)
normal_predictions = []
pert_predictions = []

for i in range(no_batch):
    im_batch = image_list[i*batch_size: (i+1)*batch_size]
    gt_batch = gt_list[i*batch_size: (i+1)*batch_size]
    if(no_classes[model_name]==1001):
        gt_batch = [x+1 for x in gt_batch]
    feed_dict = {data_loader.image_path: im_batch}

    network_out = sess.run(logits,feed_dict=feed_dict)
    network_out = np.argmax(network_out, 1)
    normal_out = network_out[:batch_size]
    pert_out = network_out[batch_size:]
    
    no_correct = no_correct + np.sum(gt_batch==normal_out)
    acc_post = acc_post + np.sum(gt_batch==pert_out)
    normal_predictions.extend(normal_out)
    pert_predictions.extend(pert_out)
    no_fooled = no_fooled + np.sum(pert_out!=normal_out)

    if(i%50==0):
        print("Iter : %d \nAccuacy: %f" % ((i+1)*batch_size, \
                        no_correct/((i+1)*batch_size)))
        print("Post Attack Acc: %f" % (acc_post/((i+1)*batch_size)))
        print("FR: %f" % (no_fooled/((i+1)*batch_size)))


print("\n############ \nAccuacy: %f" % (no_correct/no_images))
print("\nPost Attack Acc: %f" % (acc_post/no_images))
print("\nFR: %f" % (no_fooled/no_images))

file_ptr = open('results/'+model_name+'_eval_'+expt_name+'.txt', "a+")
file_ptr.write("\n***********************************************")
file_ptr.write(pert_name)
file_ptr.write("\n############ \nAccuacy: %f" % (no_correct/no_images))
file_ptr.write("\nPost Attack Acc: %f" % (acc_post/no_images))
file_ptr.write("\nFR: %f" % (no_fooled/no_images))
file_ptr.close()
