from math import *
import tensorflow as tf
import numpy as np


def proximal_l0(alpha, c, debug=False):
    alpha_abs =  abs(alpha)
    theta = sqrt(2*c)
        

    # H is the hard threshold. It is 0 if alpha_abs[i][j] < theta.
    # It is 1 otherwise.
    h = (alpha_abs >= theta)
    h = tf.cast(h, dtype = tf.float32) 
    
    # Alpha[i][j] retains its in itial value or becomes 0.
    alpha_new = h * alpha   
   
    # Print calculations if debug mode is turned on.
    if debug:
        alpha_array = np.array(alpha.numpy())
        print("Abs(Weights)\t\tTheta\t\t\t=\tNew weights")
        for b in range(len(alpha_array)):
            print(f"{alpha_abs[b]}\t>\t{theta}\t=\t{alpha_new[b]}\n")
        print("\n")
    
    raise
    return alpha_new





def proximal_glasso_nonoverlap(alpha, c, debug=False):
    
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

    

    if debug:
        print(f"Normalization of weights matix: {alpha_norm}\n")
        print(f"C: {c}\n")
        print(f"New weight matrix:\n{alpha_new}\n")
    
    return alpha_new
