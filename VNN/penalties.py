from math import *
import tensorflow as tf


def proximal_l0(alpha, c):
    alpha_abs =  abs(alpha)
    theta = sqrt(2*c)
    
    # H is the hard threshold. It is 0 if alpha_abs[i][j] < theta.
    # It is 1 otherwise.
    h = (alpha_abs >= theta)
    h = tf.cast(h, dtype = tf.float32) 
    
    # Alpha[i][j] retains its in itial value or becomes 0.
    alpha_new = h * alpha   

    return alpha_new
