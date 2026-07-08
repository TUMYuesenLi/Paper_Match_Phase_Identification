import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData, Batch
import torch.nn.functional as F
from tqdm import tqdm
import itertools
import re
from typing import Optional
from phase_model_pkg.tools.label_utils import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class Converter:

    def __init__(self, match_id, global_data, node_data, edge_data, action_data
                 , label_data, traditional_feature_data
                 ):
        self.tradition_features = ['Ball_x'
            , 'Ball_y'
            , 'Ball_z'
            , 'Ball_angle'
            , 'Ball_goalline_angle'
            , 'closest_opponent_speed'
            , 'closest_opponent_ball_angle'
            , 'ball_dist_to_closest_oppo'
            , 'dist_convex_x'
            , 'dist_convex_x_opponent'
            , 'team_ave_speed'
            , 'team_num_of_high_speed'
            , 'opponent_ave_speed'
            , 'opponent_num_of_high_speed'
            , 'opponents_behind_ball'
            , 'last_defender_x'
            , 'dist_ball_last_defender_x'
                         ]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.match_id = match_id
        slices = [global_data[global_data['Episode_indices'] == index] for index in
                  global_data['Episode_indices'].unique()]
        counts = []
        indices = []
        i = 0
        for episode_df in slices:
            label_episode_df = label_data[label_data['Frame'].isin(episode_df['Frame'])]
            aim_list = label_episode_df['Aim']
            for k, v in enumerate(itertools.groupby(aim_list)):
                aims = list(v[1])
                length = [len(aims)] * len(aims)
                aim_indices = [k+i] * len(aims)
                counts.extend(length)
                indices.extend(aim_indices)
            i+=k+1
        global_data['Aim_indices'] = indices

        aim_slices = [global_data[global_data['Aim_indices'] == index] for index in
                  global_data['Aim_indices'].unique()]
        phase_counts = []
        phase_indices_list = []
        j = 0
        for aim_df in aim_slices:
            label_aim_df = label_data[label_data['Frame'].isin(aim_df['Frame'])]
            phase_list = label_aim_df['Matchphases_new']
            for l, w in enumerate(itertools.groupby(phase_list)):
                phases = list(w[1])
                length = [len(phases)] * len(phases)
                phase_indices = [l + j] * len(phases)
                phase_counts.extend(length)
                phase_indices_list.extend(phase_indices)
            j += l+1
        global_data['Phase_indices'] = phase_indices_list


        global_data['Match_id'] = match_id
        # self.device = torch.device('cpu')
        label_data = phase2aim(label_data)
        self.label_data = label_data[['Episode', 'Matchphases_new', 'Aim']]
        self.label_data['Graph_id'] = self.label_data.index.values
        self.label_data = self.label_data.dropna(subset=['Episode', 'Matchphases_new', 'Aim']).reset_index(drop=True)
        self.label_data['Aim_encoded'] = self.label_data['Aim'].apply(self.aim_transform)
        self.label_data['Matchphases_new_encoded'] = self.label_data['Matchphases_new'].apply(self.phase_transform)
        # print(len(self.label_data))
        global_data['Graph_id'] = global_data.index.values
        global_data = global_data.dropna(subset='Episode').reset_index(drop=True)
        node_data = self.add_position_group(node_data, unknown_id=-1)
        traditional_feature_data['Graph_id'] = traditional_feature_data.index.values
        traditional_feature_data = traditional_feature_data.dropna(subset='Episode').reset_index(drop=True)
        self.traditional_feature_data = traditional_feature_data[traditional_feature_data['Graph_id'].isin(self.label_data['Graph_id'])].reset_index(drop=True)

        self.node_data = node_data.merge(global_data[['Frame_index', 'Graph_id']], how='left')
        self.edge_data = edge_data.merge(global_data[['Frame_index', 'Graph_id']], how='left')
        self.action_data = action_data.merge(global_data[['Frame_index', 'Graph_id']], how='left')

        self.global_data = global_data[global_data['Graph_id'].isin(self.label_data['Graph_id'])].reset_index(drop=True)
        # print(len(self.global_data))
        self.node_data = self.node_data[self.node_data['Graph_id'].isin(self.label_data['Graph_id'])].reset_index(drop=True)
        self.edge_data = self.edge_data[self.edge_data['Graph_id'].isin(self.label_data['Graph_id'])].reset_index(drop=True)
        self.action_data = self.action_data[self.action_data['Graph_id'].isin(self.label_data['Graph_id'])].reset_index(
            drop=True)

        traditional_filter_features = ['Ball_angle'
            , 'Ball_goalline_angle'
            , 'closest_opponent_ball_angle'
         ]
        self.traditional_feature_data\
            = self.low_pass_filter(self.traditional_feature_data, traditional_filter_features)
        traditional_standardized_features = ['Ball_x'
            , 'Ball_y'
            , 'Ball_angle'
            , 'last_defender_x'
                                             ]

        traditional_standardized_features_positive = list(set(self.tradition_features)
                                                          - set(traditional_standardized_features))
        traditional_feature_scaler = MinMaxScaler((-1, 1))
        traditional_feature_scaler_positive = MinMaxScaler((0, 1))
        self.traditional_feature_data[traditional_standardized_features] \
            = traditional_feature_scaler.fit_transform(self.traditional_feature_data[traditional_standardized_features])
        self.traditional_feature_data[traditional_standardized_features_positive] \
            = traditional_feature_scaler_positive.fit_transform(self.traditional_feature_data[traditional_standardized_features_positive])

        node_standardized_features = ['player_x', 'player_y', 'player_angle', 'player_ball_angle', 'player_a'
            , 'player_ball_relative_v']
        node_standardized_features_positive = ['player_v', 'player_ball_distance', 'player_voronoi_area']
        node_scaler = MinMaxScaler((-1, 1))
        node_scaler_positive = MinMaxScaler((0, 1))
        self.node_data[node_standardized_features]\
            = node_scaler.fit_transform(self.node_data[node_standardized_features])
        self.node_data[node_standardized_features_positive] \
            = node_scaler_positive.fit_transform(self.node_data[node_standardized_features_positive])
        # print(self.node_data['player_voronoi_area'])

        edge_standardized_features = ['player_distance', 'player_angle']
        edge_scaler = MinMaxScaler((-1, 1))
        edge_scaler_positive = MinMaxScaler((0, 1))
        # self.edge_data[['player_angle']] \
        #     = edge_scaler.fit_transform(self.edge_data[['player_angle']])
        self.edge_data[edge_standardized_features] \
            = edge_scaler_positive.fit_transform(self.edge_data[edge_standardized_features])

        self.global_data = self.global_data[['Frame_index', 'Graph_id', 'Episode_indices', 'Aim_indices',
                                                'Phase_indices',
                                                'ball_x',
                                                'ball_y',
                                                'ball_z',
                                                'ball_angle',
                                                'ball_v',
                                                'Last_episode_opponent',
                                                'Last_episode_length',
                                                'Episode_gap',
                                                'prev_action',
                                                'next_action',
                                            ]]
        self.global_data = self.low_pass_filter(self.global_data, 'ball_angle')
        self.global_data[['Last_episode_length', 'Episode_gap']] = self.global_data[['Last_episode_length', 'Episode_gap']].fillna(0)
        global_standardized_features = ['ball_x', 'ball_y', 'ball_angle']
        global_standardized_features_positive = ['ball_z', 'ball_v', 'Last_episode_length', 'Episode_gap']
        global_scaler = MinMaxScaler((-1, 1))
        global_scaler_positive = MinMaxScaler((0, 1))
        self.global_data[global_standardized_features]\
            = global_scaler.fit_transform(self.global_data[global_standardized_features])
        self.global_data[global_standardized_features_positive] \
            = global_scaler_positive.fit_transform(self.global_data[global_standardized_features_positive])

        # self.global_data['ball_x', 'ball_y', 'ball_z', 'ball_angle', 'ball_v', 'Last_episode_length', 'Episode_gap'] =
        self.global_data['prev_action'] = self.global_data['prev_action'].apply(self.action_transform)
        self.global_data['next_action'] = self.global_data['next_action'].apply(self.action_transform)
        self.action_data['action_type'] = self.action_data['action_type'].apply(self.action_transform)
        
        action_standardized_features = ['start_x', 'start_y', 'end_x', 'end_y', 'delta_x']
        action_standardized_features_positive = ['action_goal_angle']
        action_scaler = MinMaxScaler((-1, 1))
        action_scaler_positive = MinMaxScaler((0, 1))
        self.action_data[action_standardized_features]\
            = action_scaler.fit_transform(self.action_data[action_standardized_features])
        self.action_data[action_standardized_features_positive] \
            = action_scaler_positive.fit_transform(self.action_data[action_standardized_features_positive])
        self.action_data.loc[self.action_data["action_goal_angle"].notna(), "action_goal_angle"]\
            = action_scaler_positive.fit_transform(
            self.action_data.loc[self.action_data["action_goal_angle"].notna(), ["action_goal_angle"]]
        )

        self.action_data['action_goal_angle'] = self.action_data['action_goal_angle'].fillna(0)
        
        # self.traditional_feature_data['Ball_angle'] = self.global_data['ball_angle']
        self.traditional_feature_data[self.tradition_features] \
            = self.traditional_feature_data[self.tradition_features].fillna(0)
        self.generate_sequence_boundary_label()
        self.merge_global_data()
        self.merge_node_data()
        # print(self.graph_ids)

    def convert(self):
        global_data = self.global_data.set_index(['Graph_id'])
        self.traditional_feature_data['Action'] = self.traditional_feature_data['Action'].replace(np.nan, ' ')
        traditional_feature_data = self.traditional_feature_data.set_index(['Graph_id'])

        # print(global_data.columns.values)
        node_data = self.node_data.set_index(['Graph_id', 'player_on_field_index'])
        # print(node_data.columns.values)
        edge_data = self.edge_data.set_index(['Graph_id'])

        # self.graph_ids = self.node_data.index.get_level_values(['match_id', 'Frame_index']).unique()
        graph_ids = global_data.index.values
        data_list = []
        # self.label_data['Matchphases_new_encoded'] = self.label_data['Matchphases_new'].apply(self.phase_transform)
        # self.label_data['Aim_encoded'] = self.label_data['Aim'].apply(self.aim_transform)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        for graph_id in tqdm(graph_ids):
            # print(graph_id)
            # 选择当前图的节点特征
            node_df = node_data.loc[graph_id]
            # print(node_df.columns)
            node_possession_status = torch.tensor(node_df['team_in_possession'].values, dtype=torch.long).to(self.device)
            node_position_id = torch.tensor(node_df['player_position'].values, dtype=torch.long).to(
                self.device)
            # del node_df['team_in_possession']
            # print(node_df.columns)
            node_features = torch.tensor(node_df[['team_in_possession', 'is_IBA_player',
                                                  'player_x', 'player_y', 'player_angle', 'player_ball_angle'
            , 'player_ball_relative_v', 'player_v', 'player_a', 'player_ball_distance', 'player_voronoi_area'
                                                  ]].values
                                         , dtype=torch.float).to(self.device)
            # print(node_features)
            node_ids = torch.tensor(node_df.index.values, dtype=torch.long).to(self.device)

            # 选择当前图的边特征
            edge_df = edge_data.loc[graph_id]
            edge_index = torch.tensor(edge_df[['player_on_field_index', 'other_player_on_field_index']].values.T
                                      , dtype=torch.long).to(self.device)
            # print(edge_df.iloc[:, 2:].values)
            edge_features = torch.tensor(edge_df[['if_teammate', 'player_distance', 'player_angle']].values
                                         , dtype=torch.float).to(self.device)

            # 选择当前图的全局特征
            # print(self.global_data.loc[graph_id].values)
            global_df = global_data.loc[graph_id]
            global_features = torch.tensor(global_df.iloc[4:12].values, dtype=torch.float).unsqueeze(0).to(self.device)
            action_features = torch.tensor(global_df.iloc[12:14].values, dtype=torch.long).unsqueeze(0).to(self.device)
            # action_features = F.one_hot(action_features, num_classes=16).view(1, -1)
            # global_features = torch.cat((global_features, action_features), dim=1).to(self.device)
            traditional_df = traditional_feature_data.loc[graph_id]
            traditional_features = torch.tensor(traditional_df[self.tradition_features].values, dtype=torch.float).unsqueeze(0).to(self.device)
            # Labels
            global_labels = torch.tensor(global_df[['Aim_encoded', 'Matchphases_new_encoded']].values, dtype=torch.long).to(
                self.device)
            # print(global_df['Aim_boundary_hard'])
            global_boundary_hard_labels = torch.tensor(global_df[['Aim_boundary_hard', 'Phase_boundary_hard']].values, dtype=torch.long).to(
                self.device)
            global_boundary_soft_labels = torch.tensor(global_df[['Aim_boundary_soft', 'Phase_boundary_soft']].values,
                                                       dtype=torch.float).to(
                self.device)

            # nan_mask = node_df[['Aim_encoded', 'Matchphases_new_encoded']].isna()
            node_labels = torch.tensor(node_df[['Aim_encoded', 'Matchphases_new_encoded']].fillna(-1)
                                       .values, dtype=torch.long).to(
                self.device)

            # 预测时掩码处理nan
            # # 找到 NaN 值的位置
            # nan_mask = torch.isnan(node_labels)
            #
            # # 单独处理非 NaN 和 NaN 部分
            # valid_labels = node_labels[~nan_mask.any(dim=1)]

            # 将掩码也转换为张量，方便后续处理
            # nan_mask = torch.tensor(nan_mask.values, dtype=torch.bool).to(self.device)
            # global_labels, node_labels = self.generate_labels()
            # 创建HeteroData对象
            data = HeteroData()
            data['player'].x = node_features
            data['player'].id = node_ids
            data['player', 'teammate', 'player'].edge_index = edge_index
            data['player', 'teammate', 'player'].edge_attr = edge_features
            data['global'].x = global_features
            data['traditional'].x = traditional_features
            data['player'].y = node_labels
            data['player'].poss = node_possession_status
            data['player'].position = node_position_id
            data['global'].y = global_labels
            data['global'].bnd_hard = global_boundary_hard_labels
            data['global'].bnd_soft = global_boundary_soft_labels
            data['action'].x = action_features
            data['Episode_indices'].id = torch.tensor(global_df['Episode_indices'], dtype=torch.int).to(
                self.device)
            data['Aim_indices'].id = torch.tensor(global_df['Aim_indices'], dtype=torch.int).to(
                self.device)
            data['Phase_indices'].id = torch.tensor(global_df['Phase_indices'], dtype=torch.int).to(
                self.device)
            data['Match_id'].id = torch.tensor(self.match_id, dtype=torch.int).to(
                self.device)
            data['Frame_index'].id = torch.tensor(global_df['Frame_index'], dtype=torch.int).to(
                self.device)
            data['Episode'].id = torch.tensor(global_df['Episode'], dtype=torch.int).to(
                self.device)
            action = self.action_transform(traditional_df['Action'])
            data['Frame_action'] = torch.tensor(action, dtype=torch.int).to(
                self.device)

            if graph_id in self.action_data['Graph_id'].unique():
                action_df = self.action_data[self.action_data['Graph_id'] == graph_id]
                action_info = torch.tensor(action_df[['action_type', 'isSuccessful', 'is_possession_team'
                , 'start_x', 'start_y', 'end_x', 'end_y'
                , 'delta_x', 'action_goal_angle', 'angle_mask', 'event_mask']].values, dtype=torch.float).to(self.device)
                data['match_action'].x = action_info
                data['event_mask'].x = torch.tensor(
                    [1], dtype=torch.long, device=self.device
                )
            else:
                no_event_action = torch.zeros(
                    (1, 11),  # 10 = 你的 event feature 维度
                    dtype=torch.float,
                    device=self.device
                )

                # 如果你不用 0 作为 no_event 的 action_type，可以单独指定
                # no_event_action[0, 0] = no_event_id

                data['match_action'].x = no_event_action

                # 2️⃣ event_mask = 0，明确告诉模型：这一帧没有真实事件
                data['event_mask'].x = torch.tensor(
                    [0], dtype=torch.long, device=self.device
                )

            # data['global'].y = globel_labels

            data_list.append(data)
        # 使用Batch将多个图数据合并为一个批处理数据集
        print('Start batching')
        batch = Batch.from_data_list(data_list).to(self.device)
        print('Batching complete')
        return batch

    def merge_global_data(self):
        self.global_data = self.global_data.merge(self.label_data, how='left')

    def merge_node_data(self):
        node_data_possession = self.node_data[self.node_data['team_in_possession'] == 1]
        node_data_possession = node_data_possession.merge(self.label_data, how='left')
        self.node_data = self.node_data.merge(node_data_possession, how='left')

    def aim_transform(self, x):
        aim_dict = {'Invade opponent space': 0,
                    'Keep possession': 1,
                    'Scoring': 2}

        return aim_dict[x]

    def phase_transform(self, x):
        phase_dict = {'Build Up': 0,
                      'Progression': 1,
                      'Counter attack': 2,
                      'Maintenance': 3,
                      'Sustained Threat': 4,
                      'Finishing': 5}
        return phase_dict[x]

    def action_transform(self, x):
        action_list = ['Kickoff', 'Pass', 'Cross', 'Reception', 'Offside', 'FreeKick', 'BallClaiming',
         'TackleLost', 'OtherStartAction', 'ThrowIn', 'OtherEndAction', 'GoalKick',
         'ShotAtGoal', 'Foul', 'SelfPass', 'Corner', 'InCompetition', ' ']
        nums = list(range(len(action_list)))
        action_dict = dict(zip(action_list, nums))
        # if x == np.nan:
        #     return 99
        # else:
        return action_dict[x]

    def generate_labels(self):
        self.merge_global_data()
        self.merge_node_data()
        global_labels = torch.tensor(self.global_data[['Matchphases_new_encoded', 'Aim_encoded']].values
                                     , dtype=torch.float).unsqueeze(0).to(self.device)
        node_labels = torch.tensor(self.node_data[['Matchphases_new_encoded', 'Aim_encoded']].values
                                   , dtype=torch.float).unsqueeze(0).to(self.device)
        return global_labels, node_labels

    def add_position_group(self,
                           df: pd.DataFrame,
                           position_col: str = "player_position",
                           group_col: str = "player_position",
                           id_col: str = "player_position",
                           unknown_id: Optional[int] = None) -> pd.DataFrame:

        groups = {
            "GK": ["TW"],
            "Back": ["LV", "IVL", "IVZ", "IVR", "RV"],
            "Winger": ["LA", "RA", "LM", "RM", "OLM", "ORM", "ROM", "HL", "HR", "OHL"],
            "Mid": ["DMZ", "DML", "DMR", "DLM", "DRM", "MZ", "ZO"],
            "Forward": ["HST", "STZ", "STL", "STR"],
        }
        group_to_id = {"GK": 0, "Back": 1, "Winger": 2, "Mid": 3, "Forward": 4}
        # 当同一单元格出现多个代码时的优先顺序（与数字编码一致）
        priority = ["GK", "Back", "Winger", "Mid", "Forward"]

        # —— 2) 代码 -> 类别 的反查表 ——
        code_to_group = {}
        for g, codes in groups.items():
            for c in codes:
                code_to_group[c.upper()] = g

        # —— 3) 将单元格映射为类别（支持一个格子多个代码：/ , ; | 空格 分隔） ——
        def classify_position(value) -> str:
            if pd.isna(value):
                return "Unlabeled"
            s = str(value).strip().upper()
            if not s or s in {"NAN", "<NA>", "NONE", "NULL"}:
                return "Unlabeled"
            tokens = re.split(r"[\/,;|\s]+", s)
            cats = [code_to_group.get(tok) for tok in tokens if tok]
            cats = [c for c in cats if c is not None]
            if not cats:
                return "Unknown"
            for p in priority:
                if p in cats:
                    return p
            return cats[0]

        # —— 4) 生成新列并编码 ——
        out = df.copy()
        out[group_col] = out[position_col].apply(classify_position)
        out[id_col] = out[group_col].map(group_to_id)

        # 处理未知/未标注的编码
        if unknown_id is not None:
            out[id_col] = out[id_col].fillna(unknown_id).astype(int)

        return out

    def generate_sequence_boundary_label(self):
        episode_data_list = []
        for episode_index in self.global_data['Episode_indices'].unique():
            episode_data = self.global_data[self.global_data['Episode_indices'] == episode_index]
            aim = episode_data['Aim_indices'].values
            y_hard_aim, _ = boundaries_from_phase_ids(aim)  # 1 at t=2 and t=6
            y_soft_aim = soft_labels_from_boundaries(y_hard_aim, window=10, scheme='flat', flat_value=1)
            episode_data['Aim_boundary_hard'] = y_hard_aim.astype(dtype=np.int32)
            episode_data['Aim_boundary_soft'] = y_soft_aim
            episode_data_list.append(episode_data)
        data = pd.concat(episode_data_list).reset_index(drop=True)
        aim_data_list = []
        for aim_index in data['Aim_indices'].unique():
            aim_data = data[data['Aim_indices'] == aim_index]
            phase = aim_data['Phase_indices'].values
            y_hard_phase, _ = boundaries_from_phase_ids(phase)  # 1 at t=2 and t=6
            y_soft_phase = soft_labels_from_boundaries(y_hard_phase, window=10, scheme='flat', flat_value=1)
            aim_data['Phase_boundary_hard'] = y_hard_phase.astype(dtype=np.int32)
            aim_data['Phase_boundary_soft'] = y_soft_phase
            aim_data_list.append(aim_data)
        self.global_data = pd.concat(aim_data_list).reset_index(drop=True)

    def low_pass_filter(self, data, column):
        episode_data_list = []
        for episode_index in data['Episode_indices'].unique():
            episode_data = data[data['Episode_indices'] == episode_index]
            episode_data[column] = episode_data[column].ewm(alpha=0.3, adjust=False).mean()
            episode_data_list.append(episode_data)
        data = pd.concat(episode_data_list).reset_index(drop=True)
        return data



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


if __name__ == '__main__':
    match_list = [
                  'DFL-MAT-003AXI_Borussia Dortmund-FC Augsburg_2018-2019',
                  'DFL-MAT-003AXO_Fortuna Düsseldorf 1895 e.V.-FC Schalke 04_2018-2019',
                  'DFL-MAT-003AXK_SV Werder Bremen-VfL Wolfsburg_2018-2019'
                  ]
    match_team_list = [
                        'BVB_FCA',
                        'S04_DÜS',
                        'SVW_WOB',
                        ]
    match_team_dict = {
        'BVB_FCA': 1,
        'S04_DÜS': 2,
        'SVW_WOB': 3,
    }
    feature_folder = 'Demo_Datasets'
    global_data_list = []
    node_data_list = []
    edge_data_list = []
    label_data_list = []
    for i, match_id in enumerate(match_list):
        global_path = f'{feature_folder}/{match_id}/Global_demo_{match_id}.csv'
        node_path = f'{feature_folder}/{match_id}/Node_demo_{match_id}.csv'
        edge_path = f'{feature_folder}/{match_id}/Edge_demo_{match_id}.csv'
        action_path = f'{feature_folder}/{match_id}/Action_demo_{match_id}.csv'
        global_data = pd.read_csv(global_path)
        node_data = pd.read_csv(node_path)
        edge_data = pd.read_csv(edge_path)
        action_data = pd.read_csv(action_path)
        label_data = pd.read_csv(f'{feature_folder}/{match_id}/Label_demo_{match_id}.csv')
        feature_data = pd.read_csv(f'{feature_folder}/{match_id}/Feature_demo_{match_id}.csv'
                                   )
        label_data['Frame_index'] = label_data.index.values

        converter = Converter(match_team_dict[match_team_list[i]], global_data, node_data, edge_data, action_data
                              , label_data, feature_data
                              )
        gnn_batch = converter.convert()
        print(match_id)
        print("Saving")
        torch.save(gnn_batch, f'{feature_folder}/{match_id}/Hetero_demo_{match_id}.pt'
                   )
        print("Saving complete")

