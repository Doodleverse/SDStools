
from skimage.restoration import inpaint
import numpy as np

def inpaint_spacetime_matrix(input_matrix):
    mask = np.isnan(input_matrix)
    return inpaint.inpaint_biharmonic(input_matrix, mask)

