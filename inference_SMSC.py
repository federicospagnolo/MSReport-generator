#%%
__author__ = "Po-Jui Lu"
__email__ = "p.lu@unibas.ch"
__version__ = "1.0"

"""
The script for automatic lesion segmentation on SMSC registered FLAIR and MPRAGE brain images.
"""
# %% Import modules
import argparse
import os
import sys
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import scipy.ndimage as ndimage
import torch
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
from model import SwinUNETR
from monai.data import (
    CacheDataset,
    DataLoader,
)
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    ConcatItemsd,
    CropForeground,
    CropForegroundd,
    LoadImaged,
    MapTransform,
    SelectItemsd,
    NormalizeIntensityd,
    Spacingd,
)
from monai.transforms.utils import allow_missing_keys_mode


class EnsureNDd(MapTransform):
    """
    Ensure input images are N-dimensional tensors with channel-first.
    Missing dimensions will be added as singleton axes.
    
    Example target:
        ndim=5  -> output shape (C, D, H, W, T)
        ndim=4  -> output shape (C, H, W, D)
        ndim=3  -> output shape (C, H, W)
    """

    def __init__(self, keys, ndim=5, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.ndim = ndim

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = torch.as_tensor(d[key])

            # Ensure channel dimension exists
            if img.ndim == self.ndim:
                d[key] = img
                continue

            if img.ndim == self.ndim - 1:
                # Assume missing channel → add channel at front
                img = img.unsqueeze(0)

            elif img.ndim < self.ndim:
                # If fewer dimensions, add channel if missing
                if img.ndim == 3:  # (H, W, D) or (H, W, T)
                    img = img.unsqueeze(0)  # (C=1, H, W, D/T)
                elif img.ndim == 2:  # (H, W)
                    img = img.unsqueeze(0)  # (C=1, H, W)

                # Pad until reaching ndim
                while img.ndim < self.ndim:
                    img = img.unsqueeze(0)

            elif img.ndim > self.ndim:
                raise ValueError(
                    f"Input {key} has {img.ndim} dims, which is more than target {self.ndim}."
                )

            d[key] = img
        return d
    

class CropForegroundChanged(CropForegroundd):
    """
    Dictionary-based version :py:class:`monai.transforms.CropForeground`.
    Crop only the foreground object of the expected images.
    The typical usage is to help training and evaluation if the valid part is small in the whole medical image.
    The valid part can be determined by any field in the data with `source_key`, for example:
    - Select values > 0 in image field as the foreground and crop on all fields specified by `keys`.
    - Select label = 3 in label field as the foreground to crop on all fields specified by `keys`.
    - Select label > 0 in the third channel of a One-Hot label field as the foreground to crop all `keys` fields.
    Users can define arbitrary function to select expected foreground from the whole source image or specified
    channels. And it can also add margin to every dim of the bounding box of foreground object.

    This transform is capable of lazy execution. See the :ref:`Lazy Resampling topic<lazy_resampling>`
    for more information.
    """

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        self.cropper: CropForeground
        box_start, box_end = self.cropper.compute_bounding_box(img=d[self.source_key])
        box_start -= 8
        box_end += 8
        if self.start_coord_key is not None:
            d[self.start_coord_key] = box_start  # type: ignore
        if self.end_coord_key is not None:
            d[self.end_coord_key] = box_end  # type: ignore

        lazy_ = self.lazy if lazy is None else lazy
        for key, m in self.key_iterator(d, self.mode):
            d[key] = self.cropper.crop_pad(img=d[key], box_start=box_start, box_end=box_end, mode=m, lazy=lazy_)
        return d


def form_cluster(data_array, struct=np.ones([3, 3, 3]), only_labelmap=False):
    """Get individual clusters

    Args:
        data_array (numpy array): The image, where to find clusters
        struct (numpy array or scipy struct array, optional): The connectivity. Defaults to np.ones([3, 3, 3]) for all-direction connectivity.
        only_labelmap (bool):

    Returns:
        label_map [numpy array]: The image having labeled clusters.
        unique_label [numpy array]: The array containing unique cluster indices.
        label_counts [numpy array]: The correpsonding voxel numbers.
    """
    if len(data_array.shape) == 2:
        # print("A 2D image is given. Structural elements changes")
        struct = np.ones([3, 3])
    label_map, _ = ndimage.label(data_array, structure=struct)
    if only_labelmap:
        return label_map
    unique_label, count_label = np.unique(label_map, return_counts=True)
    bg_ind = np.argwhere(unique_label == 0)
    unique_label = np.delete(unique_label, bg_ind)
    count_label = np.delete(count_label, bg_ind)
    return label_map, unique_label, count_label

def dp(path1: str, path2: str) -> str:
    return os.path.join(path1, path2)

# %%

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
accelerator = "gpu"
device = 1
torch.backends.cudnn.benchmark = True
# print_config()
# %% Gather image paths.
script_folder = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(
        description='''WML segmention using SwinUNETR.
                        If no arguments are given, all the default values will be used.''', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--models', metavar='-m', dest='models_path', default=dp(
        script_folder, "best_model.pth"), help="Path to the trained models")
parser.add_argument('--flair', metavar="FLAIR", dest="flair_path",default=dp(
        script_folder,"FLAIR_brain.nii.gz"), help="Absolute path for the FLAIR brain image")
parser.add_argument('--target', metavar="TARGET", dest="target_path",default=dp(
        script_folder,"TARGET_brain.nii.gz"), help="Absolute path for the taget brain image")        
parser.add_argument(
        "--mprage",
        metavar="MPRAGE",
        dest="mprage_path",
        default=dp(
        script_folder,"T1_brain.nii.gz"),
        help="Absolute path for the registered MPRAGE image brain image",
    )
# parser.add_argument('--lesion_mask', metavar='Lesion mask name', dest='name_lesion_mask',
#                         default="WML", help="Prefix for the lesion mask name")
# parser.add_argument(
#         "--verbose",
#         dest="verbose",
#         action="store_true",
#         help="Enable verbose output",
#     )
parser.add_argument(
        "--output",
        metavar="Output",
        dest="output_path",
        default="subject_0",
        help="Path for the output",
    )

args = parser.parse_args()
if not len(sys.argv) > 1:
        print("No argument is given. All the default values will be used.")
    # print(args)
flair_path = Path(args.flair_path)
mp2rage_path =  Path(args.mprage_path)
target_path = Path(args.target_path)
output_pred_dir = Path(args.output_path)
models_path = Path(args.models_path)
output_suffix = ""
post_thr = 4 #More than 3 voxels

flair_name = flair_path.name
mp2rage_name = mp2rage_path.name
# %%
random_seed = 42

ckpt_files = list(models_path.glob("*.ckpt"))
print(f"Found {len(ckpt_files)} models")
val_files = [
    {
        "flair": flair_path.as_posix(),
        "mprage": mp2rage_path.as_posix(),
    }
]


# %%
config_dict = {}
config_dict["roi_size"] = [64]*3
config_dict["num_class"] = 2

val_transforms =  Compose(
    [
        LoadImaged(keys=["flair","mprage",  ], dtype=np.float32),
        EnsureNDd(keys=["flair", "mprage", ],ndim=4),
        Spacingd(keys=["flair", "mprage", ],pixdim=(1.0, 1.0, 1.0),
               mode=("bilinear", "bilinear"),
               align_corners=True,
               dtype=np.float32),
        CropForegroundChanged(keys=[ "flair","mprage"], source_key="flair", allow_smaller=False),
        NormalizeIntensityd(keys=["flair", "mprage"], nonzero=True),

        ConcatItemsd(keys=["flair", "mprage"], name="image", dim=0),
        SelectItemsd(
                keys=["image", ], 
            ),

    ]
)

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, num_workers=None, cache_num=0
)

val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=False)

# %%
class Net(pl.LightningModule):
    def __init__(self, config_dict: Dict):
        super().__init__()
        self.roi_size = config_dict["roi_size"]
        self.num_class = config_dict["num_class"]

        self._model =SwinUNETR(
             img_size=self.roi_size,
             in_channels=2,
             out_channels=1,
             num_heads=(3, 6, 12, 24),
             feature_size=12,)
        
        self.post_pred = torch.sigmoid
        self.post_label = AsDiscrete(to_onehot=None, threshold=0.5 )

    def forward(self, x):
        return self._model(x)

    def predict_step(self, batch, batch_idx):
        roi_size = self.roi_size
        sw_batch_size = 10
        outputs = sliding_window_inference(
            batch["image"], roi_size, sw_batch_size, self.forward, overlap=0.5
        )
        outputs = self.post_pred(outputs)

        return outputs

ensemble_list = []

for f in ckpt_files:
    net = Net(config_dict)
    ckpt = torch.load(f, weights_only=True)
    net.load_state_dict(ckpt["state_dict"])

    trainer = pl.Trainer(
    devices=device,
    accelerator=accelerator,
    fast_dev_run=False,
    enable_model_summary=False,
    logger=False,enable_progress_bar=False
    )

    prediction_outputs = trainer.predict(model=net, dataloaders=val_loader) 

# %%
    post_val_transforms = Compose(
    [
        LoadImaged(keys=["flair",], dtype=np.float32),
        EnsureNDd(keys=["flair",], ndim=4),
        Spacingd(keys=["flair", ],pixdim=(1.0, 1.0, 1.0),
               mode=("bilinear"),
               align_corners=True,
               dtype=np.float32),
        CropForegroundChanged(keys=["flair",], source_key="flair", allow_smaller=False),

    ]
    )

    binarize_label = AsDiscrete(argmax=False, threshold=0.5, to_onehot=None)

    subj_file_dict = val_files[0]
    transformed_data = post_val_transforms(subj_file_dict)
    label_path = Path(subj_file_dict["flair"])
    label_proxy = nib.load(label_path)

    prediction_prob = prediction_outputs[0].clone().detach().cpu().squeeze(0)
    if prediction_prob.affine.ndim == 3:
        prediction_prob.affine = prediction_prob.affine[0,...]

    prediction_prob.applied_operations = transformed_data["flair"].applied_operations
    predict_dict = {"flair": prediction_prob}
    with allow_missing_keys_mode(post_val_transforms):
        inverted_pred_dict = post_val_transforms.inverse(predict_dict)
    prediction_prob = inverted_pred_dict["flair"].squeeze(0).numpy()
# print("Prediction pro",prediction_prob.shape)
# print("="*10)
    new_img_proxy = nib.Nifti1Image(
    prediction_prob, affine=label_proxy.affine, header=label_proxy.header
    )
    new_img_proxy.set_data_dtype(
    new_img_proxy.get_data_dtype()
    )  # uint32 instead of uint16
    #nib.save(
    #new_img_proxy,
    #output_pred_dir /  f"probmap{output_suffix}.nii.gz",
    #)
    prediction_label = (prediction_prob >= 0.5).astype(np.uint8)
    label_map, unique_label, count_label = form_cluster(prediction_label)
    for the_small_label in unique_label[count_label < post_thr]:
        prediction_label[label_map == the_small_label] = 0
    
    output_name =  f"lesion_mask{output_suffix}.nii.gz"
    if output_name == label_path.name:
        output_name += "_check"

    header = label_proxy.header
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(slope=np.nan, inter=np.nan)

    new_img_proxy = nib.Nifti1Image(
    prediction_label, affine=label_proxy.affine, header=header
    )
    
    ensemble_list.append(prediction_prob)
    
ens_prediction_prob = np.stack(ensemble_list, axis=0)    
orig_flair_header = nib.load(flair_path)
orig_flair_affine = orig_flair_header.affine
orig_flair = orig_flair_header.get_fdata()
brain_mask = (orig_flair != 0).astype(np.uint8)
target = nib.load(target_path).get_fdata()

to_save = {'shape': prediction_label.shape, # [H, W, D]
           'output_shape': ens_prediction_prob.shape, # [N, H, W, D]
           'brain_location': np.where(brain_mask == 1),
           'affine': orig_flair_affine,
           'targets': target[brain_mask == 1], # ground truth...
           'pred_probs': ens_prediction_prob[np.broadcast_to(brain_mask, ens_prediction_prob.shape) == 1]
           }

visit = flair_path.parent.name
patient_id = flair_path.parent.parent.name
if not os.path.exists(output_pred_dir):
    os.makedirs(output_pred_dir, exist_ok=True)
new_filename = f'{patient_id}_{visit}_pred.npz'
if not os.path.exists(os.path.join(output_pred_dir, new_filename)):
    print(f"Creating {new_filename}")
    np.savez_compressed(os.path.join(output_pred_dir, new_filename), **to_save)


# %%

