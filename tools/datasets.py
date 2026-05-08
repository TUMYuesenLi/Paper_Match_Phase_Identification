import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import copy
import random
import warnings
warnings.filterwarnings('ignore')


class SoccerHeteroDataset(Dataset):
    def __init__(self, data_list, cat
                 , aim_name=None
                 , from_new=True
                 , from_simi=False
                 , phase_interval=False
                 , batch_size=None
                 , shuffle=True
                 , down_sampler=False
                 , down_round=4
                 , max_length=100
                 , over_sampler=False
                 , minority_class=None
                 , oversample_times=None
                 ):
        """
        :param data_list: 一个 HeteroData 列表，每个 HeteroData 包含 'Episode indices'
        """
        self.cat = cat
        self.aim_name = aim_name
        self.batch_size = batch_size
        if from_new:
            self.intervals = self.group_by(data_list, from_simi
                                           # , phase_interval
                                           )
        else:
            self.intervals = data_list
        if down_sampler:
            self.down_sampler(down_round, max_length)
        if over_sampler:
            self.over_sampler(minority_class, oversample_times)
        if self.batch_size is not None:
            self.split_list_numpy(self.batch_size, shuffle)

    def group_by(self, data_list, from_simi
                 # , phase_interval
                 ):
        """
        根据 'Episode indices' 组织数据，每个 episode 作为一个完整的时间序列
        """
        interval_dict = {}
        if self.cat == 'Episode':
            interval_name = 'Episode_indices'
        elif self.cat == 'Aim':
            interval_name = 'Aim_indices'
        elif self.cat == 'Phase':
            interval_name = 'Phase_indices'
        else:
            raise ValueError('No such interval category')
        self.id_list = []
        for data in tqdm(data_list):
            interval_id = data[interval_name]['id'].item()  # 取 Episode ID
            # print(f"PhaseID:{interval_id}")
            # print(interval_id)
            if interval_id not in interval_dict:
                interval_dict[interval_id] = []
            if self.cat == 'Aim':
                if self.aim_name == 'Invade':
                    aim_list = [0]
                    phase_list = [0, 1, 2]
                elif self.aim_name == 'Keep':
                    aim_list = [1]
                    phase_list = [3]
                elif self.aim_name == 'Scoring':
                    aim_list = [2]
                    phase_list = [4, 5]
                elif self.aim_name == 'General':
                    aim_list = [0, 1, 2]
                    phase_list = [0, 1, 2, 3, 4, 5]
                else:
                    raise ValueError('No such aim category')
                if from_simi:
                    if (data['global'].y[0].item() in aim_list) and (data['global'].y[1].item() in phase_list):
                        interval_dict[interval_id].append(data)
                    else:
                        continue
                else:
                    if data['global'].y[0].item() in aim_list:
                        interval_dict[interval_id].append(data)
                        # if phase_interval:
                        #     phase_interval_name = 'Phase_indices'
                        #     phase_interval_id = data[phase_interval_name]['id'].item()
                        #     interval_dict[phase_interval_id].append(data)
                        # else:
                        #     interval_dict[interval_id].append(data)
                    else:
                        continue
            elif self.cat == 'Phase':

                # print(interval_name)
                if self.aim_name == 'Invade':
                    phase_list = [0, 1, 2]
                elif self.aim_name == 'Keep':
                    phase_list = [3]
                elif self.aim_name == 'Scoring':
                    phase_list = [4, 5]
                else:
                    raise ValueError('No such aim category')
                if data['global'].y[1].item() in phase_list:
                    self.id_list.append(interval_id)
                    interval_dict[interval_id].append(data)
                else:
                    continue
            else:
                interval_dict[interval_id].append(data)
        if self.cat == 'Aim' or self.cat == 'Phase':
            interval_list = []
            for interval in interval_dict.values():
                if len(interval) != 0:
                    interval_list.append(interval)
                else:
                    continue
            return interval_list
        else:
            return list(interval_dict.values())  # 返回按 Episode 分组的序列列表

    def split_list_numpy(self, batch_size, shuffle=True):
        if shuffle:
            random.shuffle(self.intervals)  # 直接对列表进行随机打乱
        self.intervals = [self.intervals[i:i + batch_size] for i in range(0, len(self.intervals), batch_size)]
        # return [self.intervals[i:i + self.batch_size] for i in range(0, len(self.intervals), self.batch_size)]

    def down_sampler(self, rounds, max_length):
        i = 0
        while i < rounds:
            intervals_list = []
            for interval in self.intervals:
                if len(interval) > 2*max_length:  # 只对长度大于 200 的 episode 进行下采样
                    # interval = interval[::2]  # 隔两帧取一个值
                    # intervals_list.append(interval)
                    first_half = interval[:len(interval)//2]
                    second_half = interval[len(interval)//2:]
                    # third_half = interval[(2*len(interval))//3:]
                    intervals_list.append(first_half)
                    intervals_list.append(second_half)
                    # intervals_list.append(third_half)
                else:
                    intervals_list.append(interval)
                              # else:
                #     interval = interval[::2]
            self.intervals = intervals_list
            i += 1
        # return intervals_list

    def check_ratio(self, interval, minority_class):
        if self.cat == 'Episode':
            y_key = 0
        else:
            y_key = 1
        frame_labels = [frame['global'].y[y_key].item() for frame in interval]
        count_minority = sum(1 for label in frame_labels if label == minority_class)
        return count_minority / len(frame_labels)

    def over_sampler(self, minority_class_list, oversample_times):
        interval_list = []
        for minority_class in minority_class_list:
            for interval in tqdm(self.intervals):
                ratio = self.check_ratio(interval, minority_class)
                if ratio != 0:
                    print(ratio)
                    interval_list.append(interval)
                    for i in range(oversample_times):
                        for frame in interval:
                            # 处理 player 节点特征
                            frame['player'].x[0:2] += torch.randn_like(frame['player'].x[0:2]) * 0.02
                            frame['player'].x[4:] += torch.randn_like(frame['player'].x[4:]) * 0.02

                            # 处理 global 特征
                            frame['global'].x += torch.randn_like(frame['global'].x) * 0.02
                            frame['player', 'teammate', 'player'].edge_attr[1:]\
                                += torch.randn_like(frame['player', 'teammate', 'player'].edge_attr[1:]) * 0.02
                        interval_list.append(interval)
                else:
                    # continue
                    interval_list.append(interval)
        self.intervals = interval_list

    def flatten(self):
        self.intervals = [item for sublist in self.intervals for item in sublist]

    def copy(self):
        return copy.deepcopy(self)

    def divide_by_breaks(self, breaks_list):
        interval_list = []
        if self.batch_size is not None:
            raise ValueError("Does not support batched intervals")
        else:
            for interval, breaks in zip(self.intervals, breaks_list):
                if len(breaks) == 0:
                    interval_list.append(interval)
                else:
                    parts = []
                    start = 0
                    for b in breaks:
                        part = interval[start:b+1]
                        parts.append(part)
                        start = b + 1
                    parts.append(interval[start:])
                    interval_list.extend(parts)
        self.intervals = interval_list

    def __len__(self):
        return len(self.intervals)

    def __getitem__(self, idx):
        """
        返回整个 Episode（多个 HeteroData 组成的时间序列）
        """
        return self.intervals[idx]  # 返回一个完整的时间序列


class SoccerDataset(Dataset):
    def __init__(self, data, labels
                 # , seq_len
                 ):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        label = self.labels[idx]

        return sequence, label
