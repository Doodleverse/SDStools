"""
Sharon Fitzpatrick Batiste
1/16/2024
This script plots the geo-reference accuracy for each image from any of the landsat satellites that have this metric available.
It reads each satellite directory's metadatafolder and extracts the georeference accuracy for each image.
Then plots each of these accuracies grouped by what month the image was captured and colored by the satellite it was captured by. 
Currently works for Landsat 7,8,9.

Expects the provided directory to be in format and each filename containing the metadata files to contain the date the image was captured 
in the format YYYY-MM-DD-hh-mm-ss_satellite_sessionname.txt . This is the default format from coastsat and coastseg style downloads.


sample_session_name
|_L7
  |_meta
    |_2023-12-06-13-25-32_L7_sample_session_name.txt
    |_2023-12-07-13-25-32_L7_sample_session_name.txt
|_L8
  |_meta
    |_2023-12-06-13-25-32_L8_sample_session_name.txt
    |_2023-12-07-13-25-32_L8_sample_session_name.txt
|_L9
  |_meta
    |_2023-12-06-13-25-32_L9_sample_session_name.txt
    |_2023-12-07-13-25-32_L9_sample_session_name.txt


Expects the metadata text files to be in a format like the following example below:
filename	2023-12-11-13-28-02_L7_ID_wra5_datetime01-10-25__03_58_44_ms.tif
epsg	32617
acc_georef	11.537
image_quality	9
im_width	335
im_height	334

"""
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

def read_data_from_file(file_path):
    """ Extract filename and acc_georef from a single file. """
    data = {}
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("filename"):
                parts = line.split()
                if len(parts) >= 2:
                    data['filename'] = parts[1]
            if line.startswith("acc_georef"):
                parts = line.split()
                if len(parts) >= 2:
                    data['acc_georef'] = float(parts[1])
    return data

def extract_date_from_filename(filename):
    """ Extract the date from the filename using regex. """
    date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})')
    match = date_pattern.search(filename)
    if match:
        return pd.to_datetime(match.group(1), format="%Y-%m-%d-%H-%M-%S")
    return None

def process_directory(directory, satellite_name):
    """ Process each text file in the directory. """
    records = []
    for txt_file in glob.glob(os.path.join(directory, "*.txt")):
        file_data = read_data_from_file(txt_file)
        if 'filename' in file_data and 'acc_georef' in file_data:
            date = extract_date_from_filename(file_data['filename'])
            if date:
                records.append({
                    "satellite": satellite_name,
                    "filename": file_data['filename'],
                    "acc_georef": file_data['acc_georef'],
                    "date": date
                })
    return records

def save_data_to_csv(records, output_file):
    """ Save the data to a CSV file. """
    df = pd.DataFrame(records)
    df.sort_values(by=["satellite", "date"], inplace=True)
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

def plot_data(records):
    """ Plot the date against acc_georef from the records, each satellite with different color. """
    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(10, 6))
    import matplotlib.colors as mcolors  # Correct import statement
    colors = list(mcolors.TABLEAU_COLORS)  # Get a list of color names
    for (label, group_df), color in zip(df.groupby('satellite'), colors):
        group_df['month'] = group_df['date'].dt.month  # Extract month from date
        ax.scatter(group_df["month"], group_df["acc_georef"], label=label, color=color, alpha=0.6)
    
    ax.set_xlabel("Month")
    ax.set_ylabel("acc_georef")
    ax.set_title("acc_georef Over Months by Satellite")
    ax.set_xticks(range(1, 13))  # Set x-ticks to be the months
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend()
    plt.tight_layout()
    plt.show()


#example : base_directory = r"C:\development\doodleverse\coastseg\CoastSeg\data\ID_1_datetime06-04-24__12_09_54"
base_directory = ""
satellites = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
available_satellites = ['L5', 'L7', 'L8', 'L9']
satellites = [satellite_dir for satellite_dir in satellites if os.path.basename(satellite_dir) in available_satellites]
all_records = []
for satellite_dir in satellites:
    meta_dir = os.path.join(satellite_dir, "meta")
    print(f"Processing {meta_dir}")
    if os.path.exists(meta_dir):
        records = process_directory(meta_dir,satellite_name=os.path.basename(satellite_dir))
        all_records.extend(records)

if all_records:
    output_csv = "acc_georef_results.csv"
    save_data_to_csv(all_records, output_csv)
    plot_data(all_records)
else:
    print("No valid data found in the files.")
