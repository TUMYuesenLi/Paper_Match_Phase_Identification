import pandas as pd
from tqdm import tqdm
pd.set_option('display.max_columns', None)


class PhaseFixer:

    def __init__(self, data):
        self.data = data

    def replace_progression_counter_attack_neighbors(self, group):
        phases = group['Phase_pred_filtered_aim_filtered'].tolist()
        incompatible = {'Progression', 'Counter attack'}

        for i in range(1, len(phases)):
            if {phases[i - 1], phases[i]} == incompatible and phases[i] != phases[i - 1]:
                phases[i] = phases[i - 1]

        group['Phase_pred_filtered_aim_filtered_fixed'] = phases
        return group

    def replace_progression_after_counter_attack(self, group):
        return self.replace_progression_counter_attack_neighbors(group)

    def replace_counter_after_phase(self, group):
        phases = group['Phase_pred_filtered_aim'].tolist()
        for i in range(1, len(phases)):
            if phases[i] == 'Counter attack' and phases[i - 1] in ['Build Up', 'Progression', 'Maintenance']:
                phases[i] = 'Progression'

        group['Phase_pred_filtered_aim'] = phases
        return group

    def fix_finishing(self):
        if 'Frames_to_Shot' in self.data.columns:
            frames_to_shot = pd.to_numeric(self.data['Frames_to_Shot'], errors='coerce')
            shot_window = frames_to_shot.between(0, 9, inclusive='both')
            self.data.loc[shot_window, 'Phase_pred_filtered_aim_filtered_fixed'] = 'Finishing'
            return

        if 'Ten_frames_to_shot' in self.data.columns:
            shot_window = self.data['Ten_frames_to_shot'].astype(str).str.lower().eq('true')
            self.data.loc[shot_window, 'Phase_pred_filtered_aim_filtered_fixed'] = 'Finishing'
            return

        if 'Action' not in self.data.columns:
            return

        for idx in self.data[self.data['Action'] == 'ShotAtGoal'].index:
            start_idx = max(0, idx - 9)
            self.data.loc[start_idx:idx, 'Phase_pred_filtered_aim_filtered_fixed'] = 'Finishing'

    def fix(self, fix_finishing=True):
        data_drop = self.data.copy()
        change_indices = data_drop['Episode'].ne(data_drop['Episode'].shift(1))

        slices = [data_drop.loc[idx: next_idx - 1] for idx, next_idx in zip(change_indices.index[change_indices],
                                                                            change_indices.index[change_indices][
                                                                            1:].append(
                                                                                pd.Index([len(data_drop)], dtype='int64')))]
        episode_list = []
        for episode_df in tqdm(slices):
            if episode_df['Episode'].isna().any():
                continue
            else:
                episode_df = episode_df.reset_index(drop=True)
                episode_df_modified = episode_df.groupby('Episode', group_keys=True).apply(
                    self.replace_progression_counter_attack_neighbors
                ).reset_index(drop=True)
                episode_list.append(episode_df_modified)
        df = pd.concat(episode_list, axis=0).reset_index(drop=True)
        self.data = df
        if fix_finishing:
            self.fix_finishing()
        return self.data


if __name__ == '__main__':
    data = pd.read_csv('Matchphases_BVB_FCA_Phase_pred.csv', sep=';')
    phase_fixer = PhaseFixer(data)
    data = phase_fixer.fix()
