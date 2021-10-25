import sys
# sys.path.append('<path_to_python3.6_installation>/python3.6/site-packages/aimet_common/x86_64-linux-gnu')
sys.path.append('/home/rafal/.local/lib/python3.6/site-packages/aimet_common/x86_64-linux-gnu')

import tensorflow as tf
from aimet_common.utils import AimetLogger
from aimet_tensorflow.cross_layer_equalization import equalize_model

PATH = 'net'
INPUT_OP_NAME = 'input_tensor'
OUTPUT_OP_NAME = 'Sigmoid'

def cross_layer_equalization_auto():
    sess = tf.Session()
    saver = tf.train.import_meta_graph(f'{PATH}.meta')
    saver.restore(sess, PATH)
    graph = tf.get_default_graph()
    with graph.as_default():
        with sess.as_default():
            cle_session = equalize_model(sess, INPUT_OP_NAME, OUTPUT_OP_NAME)

cross_layer_equalization_auto()