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
