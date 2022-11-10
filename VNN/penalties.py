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



def proximal_glasso_nonoverlap(alpha, c):
    
    # Euclidean is equivalent to frobenius for matrices, 
    # and 2-norm for vectors.
    alpha_norm = tf.norm(alpha, ord="euclidean")
    
    if alpha_norm > c:
        alpha_new = (alpha/alpha_norm)*(alpha_norm - c)
    
        
        
    else:
        # Set the weights of the group to 0.
        alpha_new = tf.zeros(
                alpha.shape,
                dtype=tf.dtypes.float32)

    return alpha_new
