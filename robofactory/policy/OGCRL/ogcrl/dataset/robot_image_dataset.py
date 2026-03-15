from typing import Dict
import numba
import torch
import numpy as np
import copy
from ogcrl.common.pytorch_util import dict_apply
from ogcrl.common.replay_buffer import ReplayBuffer
from ogcrl.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from ogcrl.dataset.base_dataset import BaseImageDataset

class RobotImageDataset(BaseImageDataset):
    def __init__(self,
                 zarr_path, 
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 batch_size=64,
                 max_train_episodes=None):
        """
        Initialize the dataset object, configure sampler, data buffers, etc.

        :param zarr_path: Path to the dataset containing the replay buffer file.
        :param horizon: Length of the sampled sequence.
        :param pad_before: Padding length before the sequence.
        :param pad_after: Padding length after the sequence.
        :param seed: Random seed for generating the validation mask.
        :param val_ratio: Proportion of data used for validation.
        :param batch_size: Batch size.
        :param max_train_episodes: Maximum number of training episodes. No limit if None.
        """
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=['head_camera', 'state', 'action']
        )
            
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.batch_size = batch_size

        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_validation_dataset(self):
        """
        Get a validation dataset by copying the current dataset with an inverted episode mask.

        :return: Validation dataset object.
        """
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """
        Convert a raw sample to a standardized dictionary format.

        :param sample: A sample containing camera images, state, and action.
        :return: Standardized data dictionary.
        """
        agent_pos = sample['state'].astype(np.float32)
        head_cam = np.moveaxis(sample['head_camera'], -1, 1) / 255

        data = {
            'obs': {
                'head_cam': head_cam,  # T, 3, H, W
                'agent_pos': agent_pos,  # T, D
            },
            'action': sample['action'].astype(np.float32)  # T, D
        }
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get a sample by index and convert it to PyTorch tensor format.

        :param idx: Index (supports int, slice, and batch ndarray).
        :return: Converted data dictionary.
        """
        if isinstance(idx, slice):
            raise NotImplementedError
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(self.buffers[k], v, self.sampler.indices, idx, self.sampler.sequence_length)
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        """
        Transfer sampled data to the target device and normalize.

        :param samples: Sampled data.
        :param device: PyTorch device.
        :return: Normalized data dictionary.
        """
        agent_pos = samples['state'].to(device, non_blocking=True)
        head_cam = samples['head_camera'].to(device, non_blocking=True) / 255.0
        action = samples['action'].to(device, non_blocking=True)
        
        return {
            'obs': {
                'head_cam': head_cam,  # B, T, 3, H, W
                'agent_pos': agent_pos,  # B, T, D
            },
            'action': action  # B, T, D
        }

def _batch_sample_sequence(data: np.ndarray, input_arr: np.ndarray, indices: np.ndarray, idx: np.ndarray, sequence_length: int):
    """
    Batch sample sequences, copying data from the replay buffer to the target buffer.

    :param data: Target data array.
    :param input_arr: Input array (from replay buffer).
    :param indices: Index array determining start/end positions for each sequence.
    :param idx: Indices to sample.
    :param sequence_length: Sequence length.
    """
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]  # pad before
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]  # pad after

_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)

def batch_sample_sequence(data: np.ndarray, input_arr: np.ndarray, indices: np.ndarray, idx: np.ndarray, sequence_length: int):
    """
    Choose parallel or sequential batch sampling based on data size.

    :param data: Target data array.
    :param input_arr: Input data array.
    :param indices: Index array.
    :param idx: Current batch indices.
    :param sequence_length: Sequence length.
    """
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2 ** 16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
