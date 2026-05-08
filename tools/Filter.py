import pandas as pd
import numpy as np
from tqdm import tqdm
pd.set_option('display.max_columns', None)


class AimFilter:

    def __init__(self, data, gnn=False):
        self.data = data
        self.aim_pred = self.data['Aim_pred']
        self.episode = self.data['Episode']
        self.gnn = gnn

    def get_length(self, episode_data):
        counts = []
        indices = []
        start_indices = []
        stop_indices = []
        change_indices = episode_data['Aim_pred'].compare(episode_data['Aim_pred'].shift(1), align_axis=1).dropna(
            how='all').index
        # 增加第一个索引和最后一个索引
        change_indices = [0] + list(change_indices) + [len(episode_data)]
        # 使用这些索引进行切片
        slices = [episode_data.loc[idx: next_idx - 1] for idx, next_idx in zip(change_indices[:-1], change_indices[1:])]
        # 打印结果
        for k, slice_ in enumerate(slices):
            if slice_.empty:
                continue
            else:
                length = [len(slice_)] * len(slice_)
                # print(length, len(length))
                aim_indices = [k] * len(slice_)
                counts.extend(length)
                indices.extend(aim_indices)
                start = [slice_.index[0]] * len(slice_)
                stop = [slice_.index[-1]] * len(slice_)
                start_indices.extend(start)
                stop_indices.extend(stop)

        # for k, v in enumerate(itertools.groupby(self.aim_pred)):
        #     aims = list(v[1])
        #     length = [len(aims)] * len(aims)
        #     aim_indices = [k] * len(aims)
        #     counts.extend(length)
        #     indices.extend(aim_indices)
        # print(episode_data)
        episode_data['length'] = counts
        episode_data['Aim_indices'] = indices
        episode_data['Aim_start'] = start_indices
        episode_data['Aim_end'] = stop_indices
        return episode_data

    def get_aim_filtered_length(self, episode_data):
        counts = []
        indices = []
        start_indices = []
        stop_indices = []
        change_indices = episode_data['Aim_filtered'].compare(episode_data['Aim_filtered'].shift(1), align_axis=1).dropna(
            how='all').index
        # 增加第一个索引和最后一个索引
        change_indices = [0] + list(change_indices) + [len(episode_data)]
        # 使用这些索引进行切片
        slices = [episode_data.loc[idx: next_idx - 1] for idx, next_idx in zip(change_indices[:-1], change_indices[1:])]
        # 打印结果
        for k, slice_ in enumerate(slices):
            if slice_.empty:
                continue
            else:
            # print(slice_)
                length = [len(slice_)] * len(slice_)
                aim_indices = [k] * len(slice_)
                counts.extend(length)
                indices.extend(aim_indices)
                start = [slice_.index[0]] * len(slice_)
                stop = [slice_.index[-1]] * len(slice_)
                start_indices.extend(start)
                stop_indices.extend(stop)

        episode_data['Aim_filtered_length'] = counts
        episode_data['Aim_filtered_indices'] = indices
        episode_data['Aim_filtered_start'] = start_indices
        episode_data['Aim_filtered_end'] = stop_indices
        return episode_data

    def find_most(self, df, aim, aim_length, start_index, end_index, interval):
        if len(df) <= interval:
            aim_new = df.mode()['Aim_pred'].values[0]
            
        elif interval <= start_index and end_index <= len(df) - interval:
            if aim_length < interval:
                interval_data = df.iloc[start_index - interval:end_index + interval]
                # print(interval_data)
                interval_data = interval_data.dropna(subset=['Aim_pred'])
                # print(interval_data.mode()['Aim_pred'].values)
                aim_new = interval_data.mode()['Aim_pred'].values[0]
                # print(interval_data.mode()['Aim_pred'].values)
            else:
                aim_new = aim
        elif start_index < interval:
            if aim_length < interval:
                interval_data = df.iloc[0:end_index + interval]
                interval_data = interval_data.dropna(subset=['Aim_pred'])
                aim_new = interval_data.mode()['Aim_pred'].values[0]
            else:
                aim_new = aim
        else:
            if aim_length < interval:
                interval_data = df.iloc[start_index - interval:]
                # print(interval_data)
                interval_data = interval_data.dropna(subset=['Aim_pred'])
                aim_new = interval_data.mode()['Aim_pred'].values[0]
            else:
                aim_new = aim
        return aim_new

    def filter(self, general_interval, possession_interval, scoring_interval):
        # data_drop = self.data.dropna(subset='Episode').reset_index(drop=True)
        data_drop = self.data.copy()
        # self.get_start_stop()
        # 找到'key'列值发生变化的索引
        change_indices = data_drop['Episode'].ne(data_drop['Episode'].shift(1))

        # change_indices = self.data['Episode'].ne(self.data['Episode'].shift(1))
        # self.data['change_indices'] = change_indices
        # data_drop = self.data.dropna(subset='Episode')
        # change_indices = data_drop['change_indices']

        if self.gnn:
            slices = [data_drop[data_drop['Episode_indices'] == index] for index in data_drop['Episode_indices'].unique()]
        else:
            # 使用这些索引进行切片
            slices = [data_drop.loc[idx: next_idx - 1] for idx, next_idx in zip(change_indices.index[change_indices],
                                                                         change_indices.index[change_indices][1:].append(
                                                                            pd.Index([len(data_drop)], dtype='int64')))]
        # data_copy = self.data.copy()
        print('Start Filtering Aims')
        episode_list = []
        for episode_df in tqdm(slices):
            if episode_df['Episode'].isna().any():
                continue
            else:
                episode_df = episode_df.reset_index(drop=True)
                # print(episode_df)
                episode_df = self.get_length(episode_df)
                # print(episode_df)
                episode_df_copy = episode_df.copy()
                aim_filtered = []
                for i, rows in enumerate(episode_df_copy.itertuples()):
                    aim_length = rows.length
                    aim = rows.Aim_pred
                    start_index = rows.Aim_start
                    end_index = rows.Aim_end
                    if aim is not np.nan:
                        if aim == 'Scoring':
                            aim_new = self.find_most(episode_df_copy, aim, aim_length
                                            , start_index, end_index, scoring_interval)
                        elif aim == 'Keep possession':
                            aim_new = self.find_most(episode_df_copy, aim, aim_length
                                                     , start_index, end_index, possession_interval)
                        else:
                            aim_new = self.find_most(episode_df_copy, aim, aim_length
                                            , start_index, end_index, general_interval)
                        episode_df_copy.loc[i, 'Aim_pred'] = aim_new
                        aim_filtered.append(aim_new)
                    else:
                        aim_filtered.append(aim)
                episode_df['Aim_filtered'] = aim_filtered
                episode_df = self.get_aim_filtered_length(episode_df)
                episode_list.append(episode_df)
        data_drop = pd.concat(episode_list, axis=0).reset_index(drop=True)
        self.data = self.data.merge(data_drop, how='left')

        return self.data


class PhaseFilter:

    def __init__(self, data, gnn=False):
        self.data = data
        self.phase_pred = self.data['Phase_pred_filtered_aim']
        self.gnn = gnn

    def get_length(self, episode_data):
        counts = []
        indices = []
        start_indices = []
        stop_indices = []
        change_indices = episode_data['Phase_pred_filtered_aim'].compare(episode_data['Phase_pred_filtered_aim'].shift(1), align_axis=1).dropna(
            how='all').index
        # 增加第一个索引和最后一个索引
        change_indices = [0] + list(change_indices) + [len(episode_data)]
        # 使用这些索引进行切片
        slices = [episode_data.loc[idx: next_idx - 1] for idx, next_idx in zip(change_indices[:-1], change_indices[1:])]
        # 打印结果
        for k, slice_ in enumerate(slices):
            if slice_.empty:
                continue
            else:
            # print(slice_)
                length = [len(slice_)] * len(slice_)
                aim_indices = [k] * len(slice_)
                counts.extend(length)
                indices.extend(aim_indices)
                start = [slice_.index[0]] * len(slice_)
                stop = [slice_.index[-1]] * len(slice_)
                start_indices.extend(start)
                stop_indices.extend(stop)

        episode_data['Phase_length'] = counts
        episode_data['Phase_indices'] = indices
        episode_data['Phase_start'] = start_indices
        episode_data['Phase_end'] = stop_indices
        return episode_data

    def find_most(self, df, aim, aim_length, start_index, end_index, interval):
        if len(df) <= interval:
            aim_new = df.mode()['Phase_pred_filtered_aim'].values[0]
        elif interval <= start_index and end_index <= len(df) - interval:
            if aim_length < interval:
                interval_data = df.iloc[start_index - interval:end_index + interval]
                # print(interval_data)
                interval_data = interval_data.dropna(subset=['Phase_pred_filtered_aim'])
                # print(interval_data.mode()['Aim_pred'].values)
                aim_new = interval_data.mode()['Phase_pred_filtered_aim'].values[0]
                # print(interval_data.mode()['Aim_pred'].values)
            else:
                aim_new = aim
        elif start_index < interval:
            if aim_length < interval:
                interval_data = df.iloc[0:end_index + interval]
                interval_data = interval_data.dropna(subset=['Phase_pred_filtered_aim'])
                aim_new = interval_data.mode()['Phase_pred_filtered_aim'].values[0]
            else:
                aim_new = aim
        else:
            if aim_length < interval:
                interval_data = df.iloc[start_index - interval:]
                interval_data = interval_data.dropna(subset=['Phase_pred_filtered_aim'])
                aim_new = interval_data.mode()['Phase_pred_filtered_aim'].values[0]
            else:
                aim_new = aim
        return aim_new

    def filter(self, general_interval, finishing_interval):
        # data_drop = self.data.dropna(subset='Episode').reset_index(drop=True)
        data_drop = self.data.copy()
        # # self.get_start_stop()
        # # 找到'key'列值发生变化的索引
        change_indices = data_drop['Episode'].ne(data_drop['Episode'].shift(1))
        print(change_indices)

        # change_indices = self.data['Episode'].ne(self.data['Episode'].shift(1))
        # self.data['change_indices'] = change_indices
        # data_drop = self.data.dropna(subset='Episode')
        # change_indices = data_drop['change_indices']

        # 使用这些索引进行切片
        if self.gnn:
            slices = [data_drop[data_drop['Aim_filtered_indices'] == index] for index in
                      data_drop['Aim_filtered_indices'].unique()]
        else:
            slices = [data_drop.loc[idx: next_idx - 1] for idx, next_idx in zip(change_indices.index[change_indices],
                                                                            change_indices.index[change_indices][
                                                                            1:].append(
                                                                                pd.Index([len(data_drop)], dtype='int64')))]
        # data_copy = self.data.copy()
        print('Start Filtering Phases')
        episode_list = []
        for episode_df in tqdm(slices):
            # print(episode_df)
            if episode_df['Aim_filtered'].isna().any():
                continue
            else:
                episode_df = episode_df.reset_index(drop=True)
                episode_df = self.get_length(episode_df)
                # print(episode_df)
                episode_df_copy = episode_df.copy()
                phase_filtered = []
                for i, rows in enumerate(episode_df_copy.itertuples()):
                    phase_length = rows.Phase_length
                    phase = rows.Phase_pred_filtered_aim
                    start_index = rows.Phase_start
                    end_index = rows.Phase_end
                    if phase is not np.nan:
                        if phase != 'Finishing':
                            phase_new = self.find_most(episode_df_copy, phase, phase_length
                                                       , start_index, end_index, general_interval)
                        else:
                            phase_new = self.find_most(episode_df_copy, phase, phase_length
                                                       , start_index, end_index, finishing_interval)
                        episode_df_copy.loc[i, 'Phase_pred_filtered_aim'] = phase_new
                        phase_filtered.append(phase_new)
                    else:
                        phase_filtered.append(phase)
                episode_df['Phase_pred_filtered_aim_filtered'] = phase_filtered
                # print(episode_df)
                episode_list.append(episode_df)
        data_drop = pd.concat(episode_list, axis=0).reset_index(drop=True)
        return data_drop
        # self.data = self.data.merge(data_drop, how='left')

        # return self.data

