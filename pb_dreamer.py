# Adapted by github.com/jnordberg from https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/deepdream
# Adapted by github.com/ProGamerGov from https://github.com/jnordberg/dreamcanvas
# wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
# unzip -d model inception5h.zip

import argparse
import scipy.ndimage as spi
from skimage.io import imread,imsave
import numpy as np
import os
import sys
import tensorflow as tf
import time

from io import BytesIO
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--input_image', type=str, help="The input image for DeepDream. Ex: input.png", required=True)
parser.add_argument('--output_image', default='output.png', help="The name of your output image. Ex: output.png", type=str)
parser.add_argument('--channel', default='139', help="The target channel of your chosen layer.", type=int)
parser.add_argument('--layer', default='mixed4d_3x3_bottleneck_pre_relu', help="The name of the target layer.", type=str)
parser.add_argument('--iter', default='10', help="The number of iterations", type=int)
parser.add_argument('--octaves', default='4', help="The number of octaves.", type=int)
parser.add_argument('--octave_scale', default='1.4', help="The step size.", type=float)
parser.add_argument('--step_size', default='1.5', help="The step size.", type=float)
parser.add_argument('--tile_size', default='512', help="The size of your tiles.", type=int)
parser.add_argument('--model', default='/home/ubuntu/Protobuf-Dreamer/model/tensorflow_inception_graph.pb', help="Path to your .pb model file.", type=str)
parser.add_argument('--print_model', help="Print the layers and inputs from the model.", action='store_false')
parser.parse_args()
args = parser.parse_args()
input_img = args.input_image
output_name = args.output_image
channel_value = args.channel
layer_name = args.layer
iter_value = args.iter
octave_value = args.octaves
octave_scale_value = args.octave_scale
step_size = args.step_size
tile_size = args.tile_size
model_path = args.model
print_model = args.print_model
input_img = spi.imread(input_img, mode="RGB")

model_fn = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_path)
# creating TensorFlow session and loading the model
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})

# Optionally print the inputs and layers of the specified graph.
if not print_model:
  print(graph.get_operations())

def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)

def calc_grad_tiled(img, t_grad, tile_size=512):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

def render_deepdream(t_grad, img0, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    # split the image into a number of octaves
    img = img0
    octaves = []
    for i in range(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        hi = img-resize(lo, hw)
        img = lo
        octaves.append(hi)

    # generate details octave by octave
    for octave in range(octave_n):
        if octave>0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
        for i in range(iter_n):
            #g = calc_grad_tiled(img, t_grad)
	    g = calc_grad_tiled(img, t_grad, tile_size)
            img += g*(step / (np.abs(g).mean()+1e-7))

    return Image.fromarray(np.uint8(np.clip(img/255.0, 0, 1)*255))

last_layer = None
last_grad = None
last_channel = None
def render(img, layer='mixed4d_3x3_bottleneck_pre_relu', channel=139, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    global last_layer, last_grad, last_channel
    if last_layer == layer and last_channel == channel:
        t_grad = last_grad
    else:
        if channel == 4242:
            t_obj = tf.square(T(layer))
        else:
            t_obj = T(layer)[:,:,:,channel]
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
        last_layer = layer
        last_grad = t_grad
        last_channel = channel
    img0 = np.float32(img)
    return render_deepdream(t_grad, img0, iter_n, step, octave_n, octave_scale)
	
	
output_img = render(input_img, layer=layer_name, channel=channel_value, iter_n=iter_value, step=step_size, octave_n=octave_value, octave_scale=octave_scale_value)
imsave(output_name, output_img)
