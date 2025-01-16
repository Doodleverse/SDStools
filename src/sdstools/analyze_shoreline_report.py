"""
Sharon Fitzpatrick Batiste
1/16/2024

This script reads the extract_shorelines_report.txt file and extracts the following information:
- Count of each method used (MNDWI thresholding and classification thresholding)
- Count of each method used by satellite (L7, L8, L9, S2)
- Method used by date

Note find_wl_contours1 and find_wl_contours2 are the methods used in the CoastSat and CoastSeg code to extract contours from images.
The script is designed to be used in the context of the CoastSeg project.
"""

import re


def parse_shoreline_report(file_path):
    # Initialize counters and storage structures
    method_count = {'MNDWI_thresholding': 0, 'classification_thresholding': 0}
    satellite_count = {'L7': {'MNDWI_thresholding': 0, 'classification_thresholding': 0},
                       'L8': {'MNDWI_thresholding': 0, 'classification_thresholding': 0},
                       'L9': {'MNDWI_thresholding': 0, 'classification_thresholding': 0},
                       'S2': {'MNDWI_thresholding': 0, 'classification_thresholding': 0}}
    method_by_date = {}

    with open(file_path, 'r') as file:
        data = file.readlines()
    
    current_satellite = ""
    for line in data:
        # Detect satellite and image processing start
        if "Processing image" in line:
            satellite = re.search(r'L\d|S2', line).group(0)
            current_satellite = satellite
            date = re.search(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', line).group(0)
        
        # Detect which method is used
        if 'Using find_wl_contours1' in line:
            # find_wl_contours1 replace with  MNDWI threshold
            method_count['MNDWI_thresholding'] += 1
            satellite_count[current_satellite]['MNDWI_thresholding'] += 1
            method_by_date[date] = 'MNDWI_thresholding'
        elif 'Using find_wl_contours2' in line:
            # find_wl_contours2 replace with  classification based threshold
            method_count['classification_thresholding'] += 1
            satellite_count[current_satellite]['classification_thresholding'] += 1
            method_by_date[date] = 'classification_thresholding'

    return method_count, satellite_count, method_by_date

# Example usage
file_path = r""
results = parse_shoreline_report(file_path)
print("Method Count:", results[0])
print("Satellite Count by Method:", results[1])
print("Method by Date:", results[2])
