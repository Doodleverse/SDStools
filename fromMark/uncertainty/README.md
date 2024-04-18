#SDSTools/fromMark/uncertainty

Uncertainty in our cross-shore positions derived from satellite shorelines comes from at least these following factors:

* Which satellite (L5, L7, L8, S2, Planet, etc.)

* Waterline delineation and intersection method (segmentation + contour following + smoothing + intersection with transects)

* Physics (Tides and waves)

We should come up with a way to propogate uncertainty so that we can get uncertainties for each observation. This will help a lot.

It would be nice to have a way of estimating this without doing a comparison of satellite-derived shorelines to other datasets. This method requires validation data everywhere we run the models.

$$$\sigma$^2 = \sigma^{2}_{satellite} + $\sigma$^{2}_{method} + $\sigma$^{2}_{physics}$$

Maybe we could formulate this as a regression problem??
