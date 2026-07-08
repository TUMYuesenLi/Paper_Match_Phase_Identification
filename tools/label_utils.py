import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Batch
import itertools
import random
import sys
import warnings
warnings.filterwarnings('ignore')


def boundaries_from_phase_ids(phase_ids, label_at='current', ignore_index=None):
    phase = np.asarray(phase_ids)
    length = len(phase)
    y_hard = np.zeros(length, dtype=np.float32)
    valid = np.ones(length, dtype=bool)

    if length == 0:
        return y_hard, valid
    if label_at == 'current':
        valid[-1] = False
    elif label_at == 'next':
        valid[0] = False
    else:
        raise ValueError("label_at must be 'current' or 'next'")

    for idx in range(length - 1):
        current, next_value = phase[idx], phase[idx + 1]
        if ignore_index is not None and (current == ignore_index or next_value == ignore_index):
            valid[idx if label_at == 'current' else idx + 1] = False
            continue
        if current != next_value:
            y_hard[idx if label_at == 'current' else idx + 1] = 1.0

    return y_hard, valid


def _apply_window_max(dst, idx, val_fn, window):
    start = max(0, idx - window)
    end = min(len(dst), idx + window + 1)
    positions = np.arange(start, end)
    distances = np.abs(positions - idx)
    dst[start:end] = np.maximum(dst[start:end], val_fn(distances).astype(dst.dtype))


def soft_labels_from_boundaries(
    y_hard,
    window=5,
    scheme='flat',
    flat_value=0.5,
    min_floor=0.0,
    triangular_floor=0.0,
    gaussian_sigma=2.0,
):
    y_soft = np.zeros(len(y_hard), dtype=np.float32)

    if scheme == 'flat':
        def val_fn(distances):
            values = np.full_like(distances, fill_value=flat_value, dtype=np.float32)
            values[distances == 0] = 1.0
            return values
    elif scheme == 'triangular':
        def val_fn(distances):
            values = 1.0 - (distances / (window + 1.0))
            values = np.maximum(values, triangular_floor).astype(np.float32)
            values[distances == 0] = 1.0
            return values
    elif scheme == 'gaussian':
        def val_fn(distances):
            values = np.exp(-0.5 * (distances / max(1e-6, gaussian_sigma)) ** 2)
            return np.clip(values, 0.0, 1.0).astype(np.float32)
    else:
        raise ValueError("scheme must be one of: 'flat', 'triangular', 'gaussian'")

    for boundary_idx in np.flatnonzero(np.asarray(y_hard) > 0.5):
        _apply_window_max(y_soft, boundary_idx, val_fn=val_fn, window=window)

    if min_floor > 0.0:
        y_soft = np.maximum(y_soft, np.float32(min_floor))
    return np.clip(y_soft, 0.0, 1.0)


def cat_hetero(data_list):
    all_frames = []
    episode_offset = 0
    aim_offset = 0
    phase_offset = 0
    print("Combining Training Data")
    for batch in data_list:
        frames = batch.to_data_list()
        for frame in tqdm(frames):
            if 'Episode_indices' in frame and hasattr(frame['Episode_indices'], 'id'):
                frame['Episode_indices'].id += episode_offset
            if 'Aim_indices' in frame and hasattr(frame['Aim_indices'], 'id'):
                frame['Aim_indices'].id += aim_offset
            if 'Phase_indices' in frame and hasattr(frame['Phase_indices'], 'id'):
                frame['Phase_indices'].id += phase_offset
            all_frames.append(frame)

        max_ep = max([f['Episode_indices'].id.max().item() for f in frames])
        episode_offset = max(episode_offset, max_ep + 1)
        max_aim = max([f['Aim_indices'].id.max().item() for f in frames])
        aim_offset = max(aim_offset, max_aim + 1)
        max_phase = max([f['Phase_indices'].id.max().item() for f in frames])
        phase_offset = max(phase_offset, max_phase + 1)

    return Batch.from_data_list(all_frames)


def split_dataset(data, train_ratio=0.5, seed=42):
    random.seed(seed)
    data = data.copy()  # 避免原地打乱
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    train_set = data[:split_idx]
    val_set = data[split_idx:]
    return train_set, val_set


def pd_collate_fn(batch):
    max_length = max(len(episode) for episode in batch)
    # print(max_length)
    batch_episodes = []
    masks = []
    for i, episode in enumerate(batch):
        episode_original_len = len(episode)
        pad_len = max_length - len(episode)
        batch_episodes.append(episode)
        mask = torch.tensor([1] * episode_original_len + [0] * pad_len, dtype=torch.bool)
        masks.append(mask)
        # print(len(mask))
    masks = pad_sequence(masks, batch_first=True, padding_value=0)
    # print(masks.shape)
    return batch_episodes, masks


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def generate_indices(data, first_level, second_level):
    slices = [data[data[f'{first_level}_indices'] == index] for index in
              data[f'{first_level}_indices'].unique()]
    counts = []
    indices = []
    i = 0
    print(f'Generating {second_level}_indices')
    for df in tqdm(slices):
        aim_list = df[second_level]
        for k, v in enumerate(itertools.groupby(aim_list)):
            aims = list(v[1])
            length = [len(aims)] * len(aims)
            aim_indices = [k + i] * len(aims)
            counts.extend(length)
            indices.extend(aim_indices)
        i += k+1
    data[f'{second_level}_indices'] = indices
    return data


def aim_transform(x):
    aim_dict = {'Invade opponent space': 0,
                'Keep possession': 1,
                'Scoring': 2}

    return aim_dict[x]


def phase_transform(x):
    phase_dict = {'Build Up': 0,
                  'Progression': 1,
                  'Counter attack': 2,
                  'Maintenance': 3,
                  'Sustained Threat': 4,
                  'Finishing': 5}
    return phase_dict[x]


def phase2aim(data):
    aim_list = []
    for phase in data['Matchphases_new'].to_list():
        if phase == 'Maintenance':
            aim = 'Keep possession'
        elif phase == 'Sustained Threat' or phase == 'Finishing':
            aim = 'Scoring'
        elif phase == 'Build Up' or phase == 'Progression' or phase == 'Counter attack':
            aim = 'Invade opponent space'
        else:
            aim = np.nan
        aim_list.append(aim)
    data['Aim'] = aim_list
    return data
