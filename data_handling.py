import logging
import os
import torch
from monai.networks.utils import one_hot
import numpy as np
from pathlib import Path


class PredictorNpzEnsemble:
    def __init__(self, data_loader, activation, device, inferer, class_num: int, n_classes: int,
                 save_path: str, set_name: str, temperature: float = 1,
                 inputs_key: str = "inputs", targets_key: str = "targets", bm_key:str = "brain_mask"):
        """
        Save probability predictions for all models in the ensemble / single model in to npz files.
        :param data_loader:
        :param activation:
        :param device:
        :param inferer:
        :param prob_threshold:
        :param temperature: temperature scaling parameter, applied only if activation is not None
        :param class_num:
        :param save_path:
        :param set_name:
        :param save_pred:
        :param include_background:
        :param postprocessing:
        :param inputs_key:
        :param targets_key:
        """
        self.temperature = temperature
        self.data_loader = data_loader
        self.activation = activation
        self.inferer = inferer
        self.inputs_key = inputs_key
        self.targets_key = targets_key
        self.bm_key = bm_key
        self.device = device
        self.class_num = class_num
        self.n_classes = n_classes
        self.save_path_pred = os.path.join(save_path, f"predictions_{set_name}_npz")
        os.makedirs(self.save_path_pred, exist_ok=True)

    def __call__(self, network: list, *args, **kwargs):
        for n in network:
            n.eval()
        with torch.no_grad():
            for data in self.data_loader:
                filename = data['targets_meta_dict']['filename_or_obj'][0]
                filename = os.path.basename(filename)
                new_filename = filename.split('.')[0] + '_pred.npz'
                if not os.path.exists(os.path.join(self.save_path_pred, new_filename)):
                    logging.info(f"Creating {new_filename}")
                    inputs, targets, brain_mask = data[self.inputs_key].to(self.device), \
                                                data[self.targets_key].to(self.device), \
                                                data[self.bm_key].squeeze(0).squeeze(0).numpy()
                    outputs: list = [self.inferer(inputs=inputs, network=n) for n in network]
                    if self.activation is not None:
                        outputs: list = [self.activation(o / self.temperature) for o in outputs]     # list of [1, 2, H, W, D]
                        outputs: list = [o.squeeze(0).cpu().numpy()[self.class_num] for o in outputs]  # [H, W, D]
                    else:
                        outputs: list = [o.squeeze(0).cpu().numpy() for o in outputs] # [2, H, W, D]
                    outputs: np.ndarray = np.stack(outputs, axis=0)

                    targets = one_hot(targets, num_classes=self.n_classes)
                    targets = targets.squeeze(0).cpu().numpy()[self.class_num]  # [H, W, D]

                    to_save = {
                        'shape': targets.shape, 'output_shape': outputs.shape,
                        'brain_location': np.where(brain_mask == 1),
                        'affine': data['targets_meta_dict']['affine'][0],
                        'targets': targets[brain_mask == 1],
                        'pred_logits' if self.activation is None else 'pred_probs': outputs[np.broadcast_to(brain_mask, outputs.shape) == 1]
                    }

                    np.savez_compressed(os.path.join(self.save_path_pred, new_filename), **to_save)
                else:
                    logging.info(f"Already exists {new_filename}")
                    
                    
class NpzDataset:
    def __init__(self, pred_path:str, pred_prefix: str = 'pred.npz'):
        """

        :param pred_path: path to the directory with npz files
        :param pred_prefix: prefix of the npz files
        """
        self.pred_filepaths: list = sorted(list(Path(pred_path).glob(f"*{pred_prefix}")))

        logging.info(f"Initializing the dataset. Number of subjects {len(self.pred_filepaths)}")

    def __len__(self):
        return len(self.pred_filepaths)

    def __getitem__(self, idx):
        data: np.lib.npyio.NpzFile = np.load(self.pred_filepaths[idx])
        data_dict = {
            'shape': data['shape'],
            'affine': data['affine'],
            'filename': os.path.basename(self.pred_filepaths[idx])
        }

        # parse brain mask
        bm = np.zeros(data['shape'])
        bm[data['brain_location'][0], data['brain_location'][1], data['brain_location'][2]] = 1
        data_dict['brain_mask'] = bm

        # parse outputs
        output_name = 'pred_logits' if 'pred_logits' in data.files else 'pred_probs'
        output = np.zeros(shape=data['output_shape'])
        output[np.broadcast_to(bm, data['output_shape']) == 1] = data[output_name]
        data_dict[output_name] = output
        # data_dict[output_name] = data[output_name]

        # parse targets
        target = np.zeros(data['shape'])
        target[bm == 1] = data['targets']
        data_dict['targets'] = target

        return data_dict
