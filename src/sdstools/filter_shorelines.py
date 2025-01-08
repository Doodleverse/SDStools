"""
Applying Filters to shorelines and plot
"""
import numpy as np
    
def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val)

def apply_filters(shorelines,
                  which_filter
                  thresh=0):

    if which_filter == 'overall'
        weighted_overall_scores = (0.2*shorelines['kde_value']+
                              0.5*shorelines['model_scores']+
                              0.3*shorelines['model_scores_seg'])/3
        weighted_overall_scores = min_max_normalize(weighted_overall_scores)
        shorelines['weighted_overall_score'] = weighted_overall_scores
        shorelines = shorelines[shorelines['weighted_overall_score']>=thresh].reset_index(drop=True)
    elif which_filter == 'image':
        shorelines = shorelines[shorelines['model_scores']>=thresh]
    elif which_filter == 'seg':
        shorelines = shorelines[shorelines['model_scores_seg']>=thresh].reset_index(drop=True)
    elif which_filter == 'kde':
        shorelines = shorelines[shorelines['kde_value']>=thresh].reset_index(drop=True)

    return shorelines


                  


                           
