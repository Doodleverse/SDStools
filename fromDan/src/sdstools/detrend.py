
import numpy as np 



def detrend_shoreline_rel_mean(input_matrix):
    "doc string here"
    shore_change = (input_matrix - input_matrix.mean(axis=0)).T
    return shore_change


def detrend_shoreline_rel_start(input_matrix, N=10):
    "doc string here"
    shore_change = (input_matrix - input_matrix[:N,:].mean(axis=0)).T
    return shore_change