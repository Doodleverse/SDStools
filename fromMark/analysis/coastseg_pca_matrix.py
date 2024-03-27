"""
Mark Lundine
Principal Component Analysis on satellite shoreline matrices
(time in y-dimension, space (transect) in x-direction).
"""

import numpy
import matplotlib.pyplot as plt
import os
import geopandas
import pandas
from sklearn.decomposition import PCA


def pca_coast_seg(coastseg_matrix_resampled_path,
                  number_of_components):
