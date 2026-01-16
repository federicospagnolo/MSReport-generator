# %%
'''
For lesion information
Usage: python lesion_information.py <name for the output excel> <Subject1's ID> <Path to subject1's image> <Path to subject1's label mask> <Subject2's ID> <Path to subject2's image> <Path to subject2's label mask>

'''
import os
import sys
import nibabel as nib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from scipy import ndimage
from pathlib import Path
import scipy.ndimage as ndimage
from lesion_extraction import get_lesion_types_masks

project_directory = os.getcwd()
df_name = '{0}.xlsx'.format(sys.argv[1])  # * Output file name

save_each_subject = True  # * For each subject, save a table
param_list = sys.argv[2:4]  # * ID, image and label
path_samseg = sys.argv[4]

save_labelmap = False


def dp(path1, path2):
    return os.path.join(path1, path2)


def form_cluster(label_data, struct=np.ones([3, 3, 3]), only_labelmap=False):
    if len(label_data.shape)==2:
        # print("A 2D image is given. Structural elements changes")
        struct = np.ones([3,3])
    label_map, _ = ndimage.label(label_data, structure=struct)
    if only_labelmap:
        return label_map
    unique_label, count_label = np.unique(label_map, return_counts=True)
    bg_ind = np.argwhere(unique_label == 0)
    unique_label = np.delete(unique_label, bg_ind)
    count_label = np.delete(count_label, bg_ind)
    return label_map, unique_label, count_label


def check_image_existence(file_path):
    if not os.path.isfile(file_path):
        print(file_path)
        sys.exit("The {} does not exist".format(os.path.basename(file_path)))


image_name_list = param_list[0::3]
mask_name_list = param_list[1::3]

# %%
for image_name, mask_name in zip(image_name_list, mask_name_list):
    check_image_existence(image_name)
    check_image_existence(mask_name)
    
    print(image_name)
    ID = (image_name.split("data/", 1)[1]).split("/flair", 1)[0].replace('/', '_')
    print("Generating report for subject " + ID + "...")
    
    img_proxy = nib.load(image_name)
    img_affine = img_proxy.affine
    img_data = img_proxy.get_fdata()
    mask_proxy = nib.load(mask_name)
    mask_data = mask_proxy.get_fdata()
    image_path = Path(image_name)
    visit_path = image_path.parent
    lesion_uncs_path = dp(visit_path, f"lesion_uncs_{ID}_pred.nii.gz")
    lesion_uncs_image = nib.load(lesion_uncs_path).get_fdata()
    patient_uncs_path = dp(visit_path, "patient_uncs_SMSC.csv")
    patient_uncs_file = pd.read_csv(patient_uncs_path)
    row = patient_uncs_file[patient_uncs_file['filename'] == f'{ID}_pred.npz']
    PSU = row['PSU'].values[0]

    unit_volume = np.asarray(mask_proxy.header['pixdim'][1:4]).prod()

    df = pd.DataFrame(columns=['ID', 'Lesion Count', 'Lesion Type', 'Lesion Index', 'Lesion Center',
                               'Lesion Voxels', 'Lesion Volume', 'LLU', 'PSU', 'Note'])

    #label_map, unique_label, count_label = form_cluster(mask_data)
    label_map = get_lesion_types_masks(mask_data, mask_data, 'non_zero', n_jobs = 1)['TPL']
    #print("unpruned are: ", np.max(label_map))
    label = 1
    while label <= np.max(label_map):
          patch_vector = np.where(label_map==label)
          if len(patch_vector[0])<4:
               label_map[label_map==label] = 0
               label_map[label_map>label] = label_map[label_map>label] - 1
          else:
               label = label + 1     

    n_labels = np.max(label_map)
    #print("pruned are: ", n_labels)
    
    unique_label = [element for element in range(1, n_labels+1)]
    lesion_map = nib.Nifti1Image(label_map, img_affine)
    nib.save(lesion_map, image_path.parent / f"lesion_map.nii.gz")

    seg_cortex_undil = nib.load(path_samseg + '/Cortex.nii.gz').get_fdata()
    seg_infratentorial_undil = nib.load(path_samseg + '/Infratentorial.nii.gz').get_fdata()
    seg_ventricles_undil = nib.load(path_samseg + '/Ventricles.nii.gz').get_fdata()
    seg_wm_undil = nib.load(path_samseg + '/WM_Mask.nii.gz').get_fdata()
    
    struct1 = ndimage.generate_binary_structure(3, 1) # define shape of dilation
    
    seg_cortex = ndimage.binary_dilation(seg_cortex_undil, structure=struct1, iterations=1)
    seg_infratentorial = ndimage.binary_dilation(seg_infratentorial_undil, structure=struct1, iterations=1).astype(int)
    seg_ventricles = ndimage.binary_dilation(seg_ventricles_undil, structure=struct1, iterations=1).astype(int)
    seg_wm = ndimage.binary_dilation(seg_wm_undil, structure=struct1, iterations=2).astype(int)
    
    #if save_labelmap:
    #    mask_file_name = os.path.basename(mask_name)
    #    nib.save(nib.Nifti1Image(label_map, affine=img_proxy.affine, header=img_proxy.header), image_path.parent / f"lesion_map_{mask_file_name}")
    for n, label_idx_in_label_map in enumerate(unique_label):

        the_cluster = label_map == label_idx_in_label_map
        masked_cluster = img_data[the_cluster]
        LLU_cluster = lesion_uncs_image[the_cluster]
        if np.all(LLU_cluster == LLU_cluster[0]):
             LLU = LLU_cluster[0]  # all equal → pick first
        else:
             LLU = LLU_cluster.mean()  # not all equal → pick mean
             print("LLU was not the same for all lesion voxels (!)")
        
        lesion_seg = the_cluster.astype(int)
        com = ndimage.center_of_mass(lesion_seg)
        com = (com[0].astype(int), com[1].astype(int), com[2].astype(int))
        
        cortex = bool(np.sum(lesion_seg & seg_cortex))
        infratentorial = bool(np.sum(lesion_seg & seg_infratentorial))
        periventricular = bool(np.sum(lesion_seg & seg_ventricles))
        wm = bool(np.sum(lesion_seg & seg_wm))      
        if infratentorial:
            lesion_type = 'infratentorial'
        elif periventricular:
            lesion_type = 'periventricular'
        elif cortex:
            lesion_type = 'juxtacortical'    
        else:
            lesion_type = 'WM'               
        
        num_voxel = len(masked_cluster)
        cluster_in_mask_data = np.unique(mask_data[the_cluster])
        note = ''
        if len(cluster_in_mask_data) > 1:
            note = ''.join(str(element)
                           for element in cluster_in_mask_data[1:])
            cluster_in_mask_data = cluster_in_mask_data[0]
        if n==0:
            lesion_number = np.max(unique_label)
        else:
            lesion_number = None
            PSU = None   
        #mean_value = masked_cluster.mean()
        #std_value = masked_cluster.std()

        df.loc[n] = [ID, lesion_number, lesion_type, label_idx_in_label_map, com, num_voxel,
                     num_voxel*unit_volume, LLU, PSU, note]
    df = df.sort_values(by=['Lesion Index'])
    if save_each_subject:
        df.to_excel(image_path.parent/ '{}_{}.xlsx'.format(
            df_name.replace('.xlsx', ''), ID), index=False)
    if os.path.isfile(dp(project_directory, df_name)):
        df_old = pd.read_excel(dp(project_directory, df_name))
        df = pd.concat([df_old, df], sort=False, ignore_index=True)
    df.to_excel(dp(project_directory, df_name), index=False)

# %%
