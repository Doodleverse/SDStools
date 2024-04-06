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

    ##Move images to Sniffer directory
    image_paths = [ ele for ele in image_paths if ele is not None ]
    sniffer_paths = [None]*len(image_paths)
    seg_jpg_paths = [None]*len(image_paths)
    seg_npz_paths = [None]*len(image_paths)
    i=0
    for image in image_paths:
        name = os.path.basename(image)
        input_path = image
        output_path = os.path.join(sniffer_image_dir, name)
        seg_jpg_path = os.path.dirname(os.path.dirname(input_path))+name+'pred_seg.jpg'
        seg_jpg_paths[i] = seg_jpg_path
        seg_npz_path = os.path.dirname(os.path.dirname(input_path))+name+'_res.npz'
        seg_npz_paths[i] = seg_npz_path
        sniffer_paths[i] = output_path
        shutil.copyfile(input_path, output_path)

    ##Make a csv with all of the paths
    output_df = pd.DataFrame({'detection_paths':image_paths,
                              'sniffer_paths':sniffer_paths,
                              'seg_jpg_paths':seg_jpg_paths,
                              'seg_npz_paths':seg_npz_paths,
                              }
                             )
    output_csv = os.path.join(sniffer_image_dir, 'image_list.csv')
    output_df.to_csv(output_csv)







        
            
    
