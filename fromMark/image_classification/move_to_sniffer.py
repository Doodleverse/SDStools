import os
import glob
import shutil
import random

def move_to_sniffer(num_images,
                    sniffer_image_dir,
                    data_dir,
                    model_session_dir):
    """
    Looks at all downloaded sites and randomly selects jpgs to put in Sniffer directory
    inputs:
    num_images (int): number of images to put into sniffer directory
    sniffer_image_dir (str): path to Sniffer/images
    data_dir (str): path to CoastSeg/data
    model_session_dir (str): path to CoastSeg/sessions
    """
    model_session_dirs = [f.path for f in os.scandir(model_session_dir) if f.is_dir()]
    num_per_site = int(num_images/len(data_dirs))
    image_paths = [None]*len(num_images)
    
    k=0
    for site in model_session_dirs:
        sample_directory = os.path.join(site, 'jpg_files', 'detection')
        imgs = glob.glob(sample_directory+'\*.jpg')
        imgs_shuffled = random.shuffle(imgs)
        for i in range(num_per_site):
            image_paths[k] = imgs_shuffled[i]
            k=k+1

    ##Move images to sniffer directory
    image_paths = [ ele for ele in image_paths if ele is not None ]
    for image in image_paths:
        name = os.path.basename(image)
        input_path = image
        output_path = os.path.join(sniffer_image_dir, name)
        shutil.copyfile(input_path, output_path)







        
            
    
