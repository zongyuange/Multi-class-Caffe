#!/usr/bin/env python

# Purpose: Multi-label Caffe
# Author:  ZongYuan Ge (PhD candidate of ACRV)
# Date:    19/05/16

############################ Library ########################################
import sys 
import os

import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from copy import copy

############################### Path ########################################
data_root    = '/home/n8744149/ge9/Multi-label/autocar/'
setup_path   = '/home/n8744149/ge9/Software/caffe-2016-03-13/examples/pycaffe/';
caffe_root   = '/home/n8744149/ge9/Software/caffe-2016-03-13/'
sys.path.insert(0,setup_path) 
sys.path.append(caffe_root + 'python')
sys.path.append(setup_path + 'layers')

import caffe
from caffe import layers as L, params as P # Shortcuts to define the net prototxt.
import caffenet
import tools

############################ Setting ########################################
# matplotlib inline
plt.rcParams['figure.figsize'] = (6, 6)
# classes = np.asarray(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
#                      'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
#                      'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])
# initialize caffe for gpu mode
caffe.set_mode_gpu()

############################ Make Solver ########################################
workdir = '/home/n8744149/ge9/Multi-label/'
if not os.path.isdir(workdir):
    os.makedirs(workdir)

solverprototxt = tools.CaffeSolver(trainnet_prototxt_path = osp.join(workdir, "trainnet.prototxt"), testnet_prototxt_path = osp.join(workdir, "valnet.prototxt"))
solverprototxt.sp['display'] = "1"
solverprototxt.sp['base_lr'] = "0.0001"
solverprototxt.write(osp.join(workdir, 'solver.prototxt'))


# write train net.
with open(osp.join(workdir, 'trainnet.prototxt'), 'w') as f:
    # provide parameters to the data layer as a python dictionary. Easy as pie!
    data_layer_params = dict(num_class = 2, batch_size = 128, im_shape = [227, 227], split = 'train', data_root = data_root)
    f.write(caffenet.caffenet_multilabel(data_layer_params, 'SelfdriveMultilabelDataLayerSync'))


# write validation net.
with open(osp.join(workdir, 'valnet.prototxt'), 'w') as f:
    data_layer_params = dict(num_class = 2, batch_size = 128, im_shape = [227, 227], split = 'val', data_root = data_root)
    f.write(caffenet.caffenet_multilabel(data_layer_params, 'SelfdriveMultilabelDataLayerSync'))


############################ Run Solver ########################################
solver = caffe.SGDSolver(osp.join(workdir, 'solver.prototxt'))
solver.net.copy_from(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
solver.test_nets[0].share_with(solver.net)
solver.step(1)

for itt in range(6):
    solver.step(1)
    print 'itt:{:3d}'.format((itt + 1) * 100), 'accuracy:{0:.4f}'.format(tools.check_accuracy(solver.test_nets[0], 1))
print 'Baseline accuracy:{0:.4f}'.format(tools.check_baseline_accuracy(solver.test_nets[0], 5823/128))


# Check the data 
transformer = tools.SimpleTransformer() # This is simply to add back the bias, re-shuffle the color channels to RGB, and so on...
image_index = 0 # First image in the batch.
plt.figure()
plt.imshow(transformer.deprocess(copy(solver.net.blobs['data'].data[image_index, ...])))
gtlist = solver.net.blobs['label'].data[image_index, ...].astype(np.int)
plt.title('GT: {}'.format(classes[np.where(gtlist)]))
plt.axis('off');

# Look at some predictions
test_net = solver.test_nets[0]
for image_index in range(5):
    plt.figure()
    plt.imshow(transformer.deprocess(copy(test_net.blobs['data'].data[image_index, ...])))
    gtlist = test_net.blobs['label'].data[image_index, ...].astype(np.int)
    estlist = test_net.blobs['score'].data[image_index, ...] > 0
    plt.title('GT: {} \n EST: {}'.format(classes[np.where(gtlist)], classes[np.where(estlist)]))
    plt.axis('off')
