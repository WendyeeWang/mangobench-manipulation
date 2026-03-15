from typing import Optional
import numpy as np
import numba
from ogcrl.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray, sequence_length: int, 
    episode_mask: np.ndarray,
    pad_before: int = 0, pad_after: int = 0,
    debug: bool = True) -> np.ndarray:
    """
    Create sampling indices for sequences. This function generates contiguous sampling
    windows (sub-sequences) for each episode from the replay buffer.

    :param episode_ends: Array of episode end positions.
    :param sequence_length: Length of each sampled sequence.
    :param episode_mask: Boolean array indicating which episodes are used for training.
    :param pad_before: Padding length before each sequence (default 0).
    :param pad_after: Padding length after each sequence (default 0).
    :param debug: Whether to enable debug assertions (default True).
    
    :return: Array of indices, each containing four values:
             [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
    """
    
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    """
    Generate a validation mask based on the specified validation ratio.

    :param n_episodes: Number of episodes in the training set.
    :param val_ratio: Proportion of episodes used for validation (between 0 and 1).
    :param seed: Random seed (default 0).
    
    :return: Boolean array indicating which episodes are in the validation set.
    """
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # Ensure at least 1 validation episode and 1 training episode
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    """
    Downsample the training mask so the number of training episodes does not exceed max_n.

    :param mask: Boolean training mask indicating which episodes are used for training.
    :param max_n: Maximum number of training episodes.
    :param seed: Random seed (default 0).
    
    :return: New training mask with at most max_n episodes.
    """
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask


class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray] = None,
        ):
        """
        Sequence sampler for sampling fixed-length sequences from the replay buffer.

        :param replay_buffer: Replay buffer containing training data.
        :param sequence_length: Length of the sampled sequence.
        :param pad_before: Padding length before the sequence (default 0).
        :param pad_after: Padding length after the sequence (default 0).
        :param keys: Keys to use (e.g., images, state, action). Uses all keys if None.
        :param key_first_k: Dict specifying to only load the first k entries per key for performance.
        :param episode_mask: Boolean array indicating which episodes to use. All episodes if None.
        """
        super().__init__()
        assert(sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        else:
            indices = np.zeros((0, 4), dtype=np.int64)

        self.indices = indices 
        self.keys = list(keys)  # Avoid OmegaConf list performance issues
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
    
    def __len__(self):
        return len(self.indices)
        
    def sample_sequence(self, idx):
        """
        Sample a sequence from the replay buffer by index.

        :param idx: Index of the sequence to sample.
        :return: Dictionary containing sequence data for each key.
        """
        
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # Only load the first k entries to avoid unnecessary memory allocation
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=np.nan, dtype=input_arr.dtype)
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                except Exception as e:
                    import pdb; pdb.set_trace()
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result
