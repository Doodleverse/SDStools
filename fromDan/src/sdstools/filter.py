

import numpy as np
from functools import partial
from skimage.restoration import calibrate_denoiser, denoise_wavelet
# rescale_sigma=True required to silence deprecation warnings
_denoise_wavelet = partial(denoise_wavelet, rescale_sigma=True)


def filter_wavelet_auto(cs_matrix_inpaint):
    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges = {'sigma': np.arange(0.02, 0.2, 0.02),
                        'wavelet': ['db1', 'db2'],
                        'convert2ycbcr': [False, False]}

    # Denoised image using default parameters of `denoise_wavelet`
    default_output = denoise_wavelet(cs_matrix_inpaint, rescale_sigma=True)

    # Calibrate denoiser
    calibrated_denoiser = calibrate_denoiser(cs_matrix_inpaint,
                                            _denoise_wavelet,
                                            denoise_parameters=parameter_ranges)

    # Denoised image using calibrated denoiser
    cs_inpaint_denoised = calibrated_denoiser(cs_matrix_inpaint)
    return cs_inpaint_denoised



