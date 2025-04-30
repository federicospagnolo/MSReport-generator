#!/bin/bash
# Script to obtain automatic report using SAMSEG
# Usage: sudo ./report.sh /home/federicospagnolo/storage/groups/think/Federico/Report_generation/data
FSLDIR=/home/federicospagnolo/usr
PATH=${FSLDIR}/bin:${PATH}
. ${FSLDIR}/etc/fslconf/fsl.sh
FREESURFER_HOME=/usr/local/freesurfer/8.0.0
. $FREESURFER_HOME/SetUpFreeSurfer.sh
CONDA_PATH=/home/federicospagnolo/miniconda3
. $CONDA_PATH/etc/profile.d/conda.sh
conda activate clwmlseg
python /home/federicospagnolo/storage/groups/think/Federico/Report_generation/predict.py --model_checkpoint /home/msxplain/Report_generation/model_epoch_61.pth --input_val_paths $1 $1 --input_prefixes flair_3d_sbr.nii.gz t1n_3d_sb.nii.gz --num_workers 0 --cache_rate 0.01 --threshold 0.3
echo "Prediction file saved"

FILES="$1/*/*/t1n_3d_s.nii.gz"

for f_t1 in $FILES; do
       	f=$(dirname $f_t1)
       	echo "Processing directory: $f"
        #relative_path=$(echo "$f" | awk -F '/home/msxplain/' '{print $2}')
	echo $f_t1
	#run_samseg --input $f/t1n_3d_s.nii.gz --output $f/SAMSEG --threads 2 > /dev/null
	mri_WMHsynthseg --i $f/t1n_3d_s.nii.gz --o $f/SYNTHSEG/seg.nii.gz --csv_vols $f/SYNTHSEG/vols.csv --threads 2
        echo "echo1"
        #mri_convert $f/SAMSEG/seg.mgz $f/SAMSEG/seg.nii.gz
	echo "echo2"
	#docker run -v /home/msxplain:/root a63c687a06d9 run_samseg --input $relative_path/t1n_3d_s.nii.gz --output $relative_path/SAMSEG --threads 2
	#run_samseg --input $f/t1n_3d_s.nii.gz --output $f/SAMSEG --threads 2
	#run_samseg --input $f/t1n_3d_s.nii.gz --output $f/SAMSEG --threads 2 > /dev/null
	#docker run -v /home/msxplain:/root a63c687a06d9 mri_convert $relative_path/SAMSEG/seg.mgz $relative_path/SAMSEG/seg.nii.gz
	#mri_convert $f/SAMSEG/seg.mgz $f/SAMSEG/seg.nii.gz
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
	fslmaths $f/SYNTHSEG/CerebralCortex.nii.gz -mul $f/SYNTHSEG/pred.nii.gz $f/SYNTHSEG/common.nii.gz 
	fslmaths $f/SYNTHSEG/CerebralCortex.nii.gz -sub $f/SYNTHSEG/common.nii.gz $f/SYNTHSEG/Cortex.nii.gz
	
	#obtain ventricles
	fslmaths $f/SYNTHSEG/LeftLateralVentricle.nii.gz -add $f/SYNTHSEG/RightLateralVentricle.nii.gz $f/SYNTHSEG/LateralVentricles.nii.gz
	fslmaths $f/SYNTHSEG/LateralVentricles.nii.gz -bin $f/SYNTHSEG/LateralVentricles.nii.gz
	fslmaths $f/SYNTHSEG/LateralVentricles.nii.gz -mul $f/SYNTHSEG/pred.nii.gz $f/SYNTHSEG/common2.nii.gz
	fslmaths $f/SYNTHSEG/LateralVentricles.nii.gz -sub $f/SYNTHSEG/common2.nii.gz $f/SYNTHSEG/Ventricles.nii.gz
	
	#obtain infratentorial
	#fslmaths $f/SYNTHSEG/Brainstem.nii.gz -add $f/SYNTHSEG/LeftCerebellumWM.nii.gz -add $f/SYNTHSEG/RightCerebellumWM.nii.gz -add $f/SYNTHSEG/RightCerebellumCortex.nii.gz -add $f/SYNTHSEG/LeftCerebellumCortex.nii.gz $f/SYNTHSEG/Infratentorial.nii.gz
	fslmaths $f/SYNTHSEG/Brainstem.nii.gz -add $f/SYNTHSEG/LeftCerebellumWM.nii.gz -add $f/SYNTHSEG/RightCerebellumWM.nii.gz $f/SYNTHSEG/Infratentorial.nii.gz
	fslmaths $f/SYNTHSEG/Infratentorial.nii.gz -bin $f/SYNTHSEG/Infratentorial.nii.gz
	
	#remove masks of each structure
	#rm $f/SAMSEG/LeftWM.nii.gz
	#rm $f/SAMSEG/LeftCerebralCortex.nii.gz
	#rm $f/SAMSEG/LeftLateralVentricle.nii.gz
	#rm $f/SAMSEG/LeftInfLateralVentricle.nii.gz
	#rm $f/SAMSEG/LeftCerebellumWM.nii.gz
	#rm $f/SAMSEG/LeftCerebellumCortex.nii.gz
	#rm $f/SAMSEG/Brainstem.nii.gz
	#rm $f/SAMSEG/RightWM.nii.gz
	#rm $f/SAMSEG/RightCerebralCortex.nii.gz
	#rm $f/SAMSEG/RightLateralVentricle.nii.gz
	#rm $f/SAMSEG/RightInfLateralVentricle.nii.gz
	#rm $f/SAMSEG/RightCerebellumWM.nii.gz
	#rm $f/SAMSEG/RightCerebellumCortex.nii.gz
	rm $f/SYNTHSEG/common.nii.gz
	rm $f/SYNTHSEG/common2.nii.gz
	
	python /home/msxplain/Report_generation/lesion_information.py report $f/flair_3d_sbr.nii.gz $f/SYNTHSEG/pred.nii.gz $f/SYNTHSEG
done
