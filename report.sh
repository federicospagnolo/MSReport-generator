#!/bin/bash
# Script to obtain automatic report using SYNTHSEG
# Usage: bash report.sh ./data

# SETUP CONDA
CONDA_PATH=/home/think/miniconda3
. $CONDA_PATH/etc/profile.d/conda.sh
conda activate SMSC_report

# SETUP FSL
FSLDIR=/home/think/fsl
PATH=${FSLDIR}/bin:${PATH}
. ${FSLDIR}/etc/fslconf/fsl.sh

shopt -s nullglob

FILES=( "$1"/*/*/t1n_brain.nii.gz )
NUM_FILES=${#FILES[@]}
echo "Found $NUM_FILES visit files"

for f_t1 in "${FILES[@]}"; do
    f=$(dirname "$f_t1")
    f=$(realpath "$f")
    echo "Processing $f"

    [ -d "$f" ] || mkdir -p "$f"
    mkdir -p "$f/SYNTHSEG"

    # Lesion segmentation
    if [ ! -f "$f/pred.nii.gz" ]; then
        echo "Running lesion segmentation inference..."
        "$CONDA_PATH/envs/SMSC_report/bin/python" inference_SMSC.py \
            --flair "$f/flair_brain.nii.gz" \
            --mprage "$f/t1n_brain.nii.gz" \
            --target "$f/lesion_mask_final.nii.gz" \
            --output "$f" \
            --models ./models
        echo "Lesion segmentation done."
        
        echo "Computing uncertainty..."
        "$CONDA_PATH/envs/SMSC_report/bin/python" compute_uncertainty.py \
            --n_samples 5 \
            --path_pred "$f" \
            --set_name SMSC \
            --path_save "$f" \
            --proba_threshold 0.5 \
            --n_jobs 6
        echo "Done."
    fi

    # SYNTHSEG Docker, input must be the original (no BET) T1 image to avoid parcellation errors!
    if [ ! -f "$f/SYNTHSEG/seg.nii.gz" ]; then
        echo "Running SYNTHSEG..."
        docker run --rm \
        -v /home/think/freesurfer/license.txt:/opt/freesurfer/license.txt:ro \
        -v "$f":/data \
        -v /home/think/freesurfer_models:/usr/local/freesurfer/8.0.0-1/models \
        -e FS_LICENSE=/opt/freesurfer/license.txt \
        freesurfer/freesurfer:8.0.0 \
        mri_WMHsynthseg \
            --i /data/t1n_3d_sb.nii.gz \
            --o /data/SYNTHSEG/seg.nii.gz \
            --csv_vols /data/SYNTHSEG/vols.csv \
            --threads 4   
    fi
    
	#obtain mask of each structure
    fslmaths $f/SYNTHSEG/seg.nii.gz -thr 1.5 -uthr 2.5 $f/SYNTHSEG/LeftWM.nii.gz
    fslmaths $f/SYNTHSEG/seg.nii.gz -thr 2.5 -uthr 3.5 $f/SYNTHSEG/LeftCerebralCortex.nii.gz
    fslmaths $f/SYNTHSEG/seg.nii.gz -thr 3.5 -uthr 4.5 $f/SYNTHSEG/LeftLateralVentricle.nii.gz
    fslmaths $f/SYNTHSEG/seg.nii.gz -thr 6.5 -uthr 7.5 $f/SYNTHSEG/LeftCerebellumWM.nii.gz
    fslmaths $f/SYNTHSEG/seg.nii.gz -thr 7.5 -uthr 8.5 $f/SYNTHSEG/LeftCerebellumCortex.nii.gz
    fslmaths $f/SYNTHSEG/seg.nii.gz -thr 15.5 -uthr 16.5 $f/SYNTHSEG/Brainstem.nii.gz
    fslmaths $f/SYNTHSEG/seg.nii.gz -thr 40.5 -uthr 41.5 $f/SYNTHSEG/RightWM.nii.gz
    fslmaths $f/SYNTHSEG/seg.nii.gz -thr 41.5 -uthr 42.5 $f/SYNTHSEG/RightCerebralCortex.nii.gz
    fslmaths $f/SYNTHSEG/seg.nii.gz -thr 42.5 -uthr 43.5 $f/SYNTHSEG/RightLateralVentricle.nii.gz
    fslmaths $f/SYNTHSEG/seg.nii.gz -thr 45.5 -uthr 46.5 $f/SYNTHSEG/RightCerebellumWM.nii.gz
    fslmaths $f/SYNTHSEG/seg.nii.gz -thr 46.5 -uthr 47.5 $f/SYNTHSEG/RightCerebellumCortex.nii.gz
	
    #obtain WM mask
    fslmaths $f/SYNTHSEG/LeftWM.nii.gz -add $f/SYNTHSEG/RightWM.nii.gz $f/SYNTHSEG/WM_Mask.nii.gz
    fslmaths $f/SYNTHSEG/WM_Mask.nii.gz -bin $f/SYNTHSEG/WM_Mask.nii.gz 	
	
    #obtain cortex
    fslmaths $f/SYNTHSEG/LeftCerebralCortex.nii.gz -add $f/SYNTHSEG/RightCerebralCortex.nii.gz $f/SYNTHSEG/CerebralCortex.nii.gz
    fslmaths $f/SYNTHSEG/CerebralCortex.nii.gz -bin $f/SYNTHSEG/CerebralCortex.nii.gz
    fslmaths $f/SYNTHSEG/CerebralCortex.nii.gz -mul $f/pred.nii.gz $f/SYNTHSEG/common.nii.gz 
    fslmaths $f/SYNTHSEG/CerebralCortex.nii.gz -sub $f/SYNTHSEG/common.nii.gz $f/SYNTHSEG/Cortex.nii.gz
	
    #obtain ventricles
    fslmaths $f/SYNTHSEG/LeftLateralVentricle.nii.gz -add $f/SYNTHSEG/RightLateralVentricle.nii.gz $f/SYNTHSEG/LateralVentricles.nii.gz
    fslmaths $f/SYNTHSEG/LateralVentricles.nii.gz -bin $f/SYNTHSEG/LateralVentricles.nii.gz
    fslmaths $f/SYNTHSEG/LateralVentricles.nii.gz -mul $f/pred.nii.gz $f/SYNTHSEG/common2.nii.gz
    fslmaths $f/SYNTHSEG/LateralVentricles.nii.gz -sub $f/SYNTHSEG/common2.nii.gz $f/SYNTHSEG/Ventricles.nii.gz
	
    #obtain infratentorial
    fslmaths $f/SYNTHSEG/Brainstem.nii.gz -add $f/SYNTHSEG/LeftCerebellumWM.nii.gz -add $f/SYNTHSEG/RightCerebellumWM.nii.gz $f/SYNTHSEG/Infratentorial.nii.gz
    fslmaths $f/SYNTHSEG/Infratentorial.nii.gz -bin $f/SYNTHSEG/Infratentorial.nii.gz
	
    rm $f/SYNTHSEG/common.nii.gz
    rm $f/SYNTHSEG/common2.nii.gz
	
    $CONDA_PATH/envs/SMSC_report/bin/python lesion_information.py report $f/flair_brain.nii.gz $f/pred.nii.gz $f/SYNTHSEG
done
