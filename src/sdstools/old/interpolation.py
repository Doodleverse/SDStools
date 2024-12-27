
from skimage.restoration import inpaint
import numpy as np
from skimage.morphology import remove_small_objects, remove_small_holes

def inpaint_spacetime_matrix(input_matrix):
    mask = np.isnan(input_matrix)
    return inpaint.inpaint_biharmonic(input_matrix, mask)


def inpaint_spacetime_matrix_masklarge(input_matrix, area_threshold =50 ):
    mask = np.isnan(input_matrix)
    mask = remove_small_objects(mask, area_threshold)
    mask = remove_small_holes(mask, area_threshold)

    return inpaint.inpaint_biharmonic(input_matrix, mask)