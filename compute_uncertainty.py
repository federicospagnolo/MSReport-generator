import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import sys
import warnings
warnings.filterwarnings("ignore")
from scipy import special, ndimage
from joblib import Parallel, delayed
from functools import partial

sys.path.insert(1, os.getcwd())
sys.path.insert(1, Path(os.getcwd()).parent)
from transforms import get_cc_mask, process_probs
from logger import save_options
from data_handling import NpzDataset
from uncertainty_measures import *

parser = argparse.ArgumentParser(description='Get all command line arguments.')
# number of samples
parser.add_argument('--n_samples', type=int, required=True, help="number of ensemble samples to use for uncertainty computation")
# data
parser.add_argument('--path_pred', type=str, required=True,
                    help='Specify the path to the directory with *_pred.npz files (predictions from precomputing/save_ens_pred_npz.py)')
parser.add_argument('--class_num', type=int, default=1,
                    help='Number of the class for which the predictions are made | do not change for the binary segmentation tasks')
parser.add_argument('--probs', action='store_true', default=True, 
                    help="whether the npz files contain predicted probabilities or logits | depends on the save_ens_pred_npz.py settings")
# parallel computation
parser.add_argument('--n_jobs', type=int, default=1,
                    help='Number of parallel workers for F1 score computation')
parser.add_argument('--det_threshold', type=float, default=0.1,
                    help='threshold for the intersection over union. see lesion_extraction.py')
parser.add_argument('--det_method', type=str, default='iou_adj',
                    help='method to classify lesions as tp, fp, us. see lesion_extraction.py')
# save dir
parser.add_argument('--set_name', required=True, type=str,
                    help='the name of the test set on which the evaluation is done for filename formation')
parser.add_argument('--path_save', type=str, required=True,
                    help='Specify the path to the directory where uncertainties will be saved')
# tuned hyperparameters
parser.add_argument('--l_min', type=int, default=2, help='minimum lesion size -1')
parser.add_argument('--proba_threshold', type=float, required=True)
parser.add_argument('--temperature', type=float, default=1)
# patient uncertainty distribution
parser.add_argument('--psu_data_filepath', type=str, required=False, default=None,
                    help='Specify the path to the CSV file with patient uncertainties | used to build plots with the distribution of uncertainties')


def main(args):
    np.random.seed(0)

    # create folder to save uncertainty maps and plots
    os.makedirs(args.path_save, exist_ok=True)
    save_options(args, os.path.join(args.path_save, "les_uncs_options.txt"))

    # load a dataset of npz predictions and extract the filenames
    npz_dataset = NpzDataset(pred_path=args.path_pred)
    filenames = list(map(os.path.basename, npz_dataset.pred_filepaths))
    
    # dataframe to save patient uncertainties
    pat_uncs_df = []
    
    # for each predicted npz file compute & save uncertainties
    for i_f, fn in enumerate(filenames):
        # load data for a given patient
        data = npz_dataset[i_f]
        if args.probs:
            mems_prob = data['pred_probs'][:args.n_samples] # each should be [2, H, W, D]
        else:
            mems_prob = data['pred_logits']
            # softmax probabilities for all models in ensemble
            mems_prob = np.stack([
                special.softmax(s / args.temperature, axis=0)[args.class_num]
                for s in mems_prob[:args.n_samples]
            ], axis=0)
        assert len(mems_prob.shape) == 4

        # get the affine
        affine = data['affine']
        
        # labeled ground truth mask
        gt_bin = data['targets']
        gt_lab = get_cc_mask(gt_bin)

        # binary prediction mask from an ensemble model
        ens_seg_bin = process_probs(
            prob_map=np.mean(mems_prob, axis=0),
            threshold=args.proba_threshold,
            l_min=args.l_min
        )
        
        ens_pred_file = os.path.join(args.path_save, f"pred.nii.gz")
        nib.Nifti1Image(ens_seg_bin, affine=affine).to_filename(ens_pred_file)

        # multi-labeled prediction mask from an ensemble model
        ens_seg_lab = get_cc_mask(ens_seg_bin)
        
        # labeled masks from each ensemble member
        mems_seg_bin = [
            process_probs(
                prob_map=ep,
                threshold=args.proba_threshold,
                l_min=args.l_min
            )
            for ep in mems_prob
        ]
        
        mems_seg_lab = np.stack(
            [get_cc_mask(ep) for ep in mems_seg_bin],
            axis=0
        )
        
        # TODO: SAVE NIFTY prediction if needed for further visualization
        # nib.Nifti1Image(ens_seg_bin.astype('float32'), affine=affine).to_filename(
        #     os.path.join(args.path_save, f"ens_pred_{fn.replace('.npz', '.nii.gz')}")
        # )

        # compute VOXEL uncertainty maps
        vox_uncs_mask: np.ndarray = entropy_of_expected(
            np.concatenate(
                (np.expand_dims(mems_prob, axis=-1),
                 np.expand_dims(1. - mems_prob, axis=-1)),
                axis=-1
            )
        )
        
        # save voxel uncertainty maps in nifty format
        # TODO: redefine the saving strategy if needed
        vox_uncs_filepath = os.path.join(args.path_save, f"voxel_uncs_{fn.replace('.npz', '.nii.gz')}")
        nib.Nifti1Image(vox_uncs_mask, affine=affine).to_filename(vox_uncs_filepath)
        
        # compute LESION & PATIENT uncertainty maps
        # if the prediction is empty, lesion and patient uncertainty are 0
        if ens_seg_lab.sum() == 0:
            les_uncs_mask = np.zeros_like(ens_seg_lab, dtype='float32')
            pat_uncs_value = 0.0
            
            # Save empty lesion uncertainty map
            les_uncs_filepath = os.path.join(args.path_save, f"lesion_uncs_{fn.replace('.npz', '.nii.gz')}")
            nib.Nifti1Image(les_uncs_mask, affine=affine).to_filename(les_uncs_filepath)
        else:
            cc_labels = np.unique(ens_seg_lab)
            cc_labels = cc_labels[cc_labels != 0.0]
            
            if len(cc_labels) > 0:
                with Parallel(n_jobs=args.n_jobs) as parallel_backend:
                    process = partial(lesion_structural_uncertainty, ens_pred_multi=mems_seg_lab)
                    
                    # returns lists of float values, each corresponding to a connected component
                    les_uncs_list = parallel_backend(delayed(process)(
                        cc_mask=(ens_seg_lab == cc_label).astype("float")
                    ) for cc_label in cc_labels)

                # create a mask in case uncertainty was computed
                les_uncs_mask = np.zeros_like(ens_seg_lab, dtype='float32')
                for cc_label, les_uncs_value in zip(cc_labels, les_uncs_list):
                    les_uncs_mask += les_uncs_value * (ens_seg_lab == cc_label).astype('float')
            else:
                les_uncs_mask = np.zeros_like(ens_seg_lab, dtype='float32')
        
            # save lesion uncertainty maps in nifty format
            les_uncs_filepath = os.path.join(args.path_save, f"lesion_uncs_{fn.replace('.npz', '.nii.gz')}")
            nib.Nifti1Image(les_uncs_mask, affine=affine).to_filename(les_uncs_filepath)
            
            # compute PATIENT uncertainty as average lesion uncertainty
            pat_uncs_value = 1 - np.mean([intersection_over_union(ens_seg_bin, mems_seg_bin[i]) for i in range(args.n_samples)])
            
        # append patient uncertainty to the dataframe
        pat_uncs_df.append({
            "filename": fn,
            "PSU": pat_uncs_value
        })
        
        # create a patient uncertainty plot if possible
        if args.psu_data_filepath is not None and pat_uncs_value > 0:
            try:
                # read the data with patient uncertainties
                # TODO: update the uncertainty data loading according to the internal project structure
                psu_data = pd.read_csv(args.psu_data_filepath, index_col=0)[["DSC-based mem_th", "set name", "hash"]]
                psu_data = psu_data[psu_data["set name"] != "ood"]["DSC-based mem_th"].values
                
                # TODO: define the save path according to the internal project structure
                plot_psu_distribution(
                    psu_data=psu_data,
                    new_psu_value=pat_uncs_value,
                    save_path=os.path.join(args.path_save, f"patient_uncertainty_distribution_{fn.replace('.npz', '.png')}")
                )
            except Exception as e:
                print(f"Warning: Could not create PSU distribution plot for {fn}: {e}")
            
        # save patient uncertainties dataframe after each patient is processed
        # TODO: redefine the saving strategy if needed
        pd.DataFrame(pat_uncs_df).to_csv(
            os.path.join(args.path_save, f"patient_uncs_{args.set_name}.csv")
        )


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
