import pandas as pd
from tqdm import tqdm
pd.set_option('display.max_columns', None)


class PhaseFixer:

    def __init__(self, data):
        self.data = data

    def replace_progression_after_counter_attack(self, group):
    # Convert the column to a list for easier manipulation
    #     print(group)
        phases = group['Phase_pred_filtered_aim_filtered'].tolist()

        for i in range(1, len(phases)):
            # Check if current phase is "Progression" and previous one is "Counter attack"
            if phases[i] == 'Progression' and phases[i - 1] == 'Counter attack':
                phases[i] = 'Counter attack'
            else:
                continue
        phases.reverse()
        for i in range(1, len(phases)):
            if phases[i-1] == 'Counter attack' and phases[i] in ['Progression']:
                phases[i] = 'Counter attack'
            else:
                continue
        phases.reverse()
    # print(phases[i])

        group['Phase_pred_filtered_aim_filtered_fixed'] = phases
        return group

    def replace_counter_after_phase(self, group):
        phases = group['Phase_pred_filtered_aim'].tolist()
        # print(phases)
        for i in range(1, len(phases)):
            # Check if current phase is "Counter attack" and previous one is "Other"
            if phases[i] == 'Counter attack' and phases[i - 1] in ['Build Up', 'Progression', 'Maintenance'] :
                print(phases[i])
                phases[i] = 'Progression'

        phases.reverse()

        # Assign the modified list back to the group
        group['Phase_pred_filtered_aim'] = phases
        return group


    def fix_finishing(self):
        for idx in self.data[self.data['Action'] == 'ShotAtGoal'].index:
            start_idx = max(0, idx - 10)
            self.data.loc[start_idx:idx, 'Phase_pred_filtered_aim_filtered_fixed'] = 'Finishing'

            # for idx in self.data[(self.data['Action'] == 'OtherEndAction')
            #                 & (self.data['Ball_x'] >= 36)
            #                 & (self.data['Ball_y'] >= -20.15)
            #                 & (self.data['Ball_y'] <= 20.15)].index:
            #     start_idx = max(0, idx - 10)
            #     self.data.loc[start_idx:idx, 'Phase_pred_filtered_aim'] = 'Finishing'

    def fix(self):
        #
        # data_modified = self.data.groupby('Episode').apply(self.replace_progression_after_counter_attack)
        # self.data['Phase_pred_filtered_aim'] = data_modified['Phase_pred_filtered_aim']
        # data_modified = self.data.groupby('Episode').apply(self.replace_counter_after_phase)
        # self.data['Phase_pred_filtered_aim'] = data_modified['Phase_pred_filtered_aim']
        # data_drop = self.data.dropna(subset='Episode').reset_index(drop=True)
        data_drop = self.data.copy()
        # self.get_start_stop()
        # 找到'key'列值发生变化的索引
        change_indices = data_drop['Episode'].ne(data_drop['Episode'].shift(1))

        # 使用这些索引进行切片
        slices = [data_drop.loc[idx: next_idx - 1] for idx, next_idx in zip(change_indices.index[change_indices],
                                                                            change_indices.index[change_indices][
                                                                            1:].append(
                                                                                pd.Index([len(data_drop)], dtype='int64')))]
        # data_copy = self.data.copy()
        episode_list = []
        for episode_df in tqdm(slices):
            if episode_df['Episode'].isna().any():
                continue
            else:
                episode_df = episode_df.reset_index(drop=True)
                # print(episode_df.columns)
                # print(episode_df)
                episode_df_modified = episode_df.groupby('Episode', group_keys=True).apply(self.replace_progression_after_counter_attack
                                                                          # , include_groups=False
                                                                          ).reset_index(drop=True)
                # print(episode_df_modified.columns)
                # print(len(episode_df))
                # print(len(episode_df_modified))
                # episode_df['Phase_pred_filtered_aim_filtered_fixed'] = episode_df_modified['Phase_pred_filtered_aim_filtered_fixed']
                # episode_df = episode_df.apply(self.replace_progression_after_counter_attack)
                episode_list.append(episode_df_modified)
        df = pd.concat(episode_list, axis=0).reset_index(drop=True)
        self.data = self.data.merge(df, how='left').reset_index(drop=True)
        self.fix_finishing()
        return self.data

