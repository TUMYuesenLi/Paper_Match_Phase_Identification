import pandas as pd
import numpy as np
from sklearn.metrics import *


class PhasesClassifier:

    def __init__(self, clf, aim, new_features=False):
        self.clf = clf
        self.aim = aim
        self.new_features = new_features
        if self.new_features:
            base_features = ['Ball_x'
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
            self.features = []
            # 生成特征名称
            # for i in range(-25, 25 + 1):
            #     for feature in base_features:
            #         self.features.append(f"{feature}_window_{i}th_frame")
            fixer_columns = ['Frames_to_Shot'
                             # , 'Ten_frames_to_shot'
                                , 'Frames_to_Block_inbox'
                                , 'Last_episode_opponent'
                                , 'Last_episode_length'
                                , 'Episode_gap'
                                , 'Episode_begin_x'
                                , 'Zone_score']
            self.features.extend(fixer_columns)
        else:
            # self.features = ['Ball_x', 'Ball_y', 'Ball_z'
            #                  , 'Ball_angle'
            #                  , 'Ball_goalline_angle'
            #                  # , 'iba_x', 'iba_y', 'iba_speed', 'iba_angle', 'iba_pressure'
            #                  # , 'dist_convex_abs'
            #                  , 'dist_convex_x', 'dist_convex_x_opponent'
            #                  # , 'dist_to_closest_oppo'
            #                  , 'team_ave_speed'
            #                  , 'team_num_of_high_speed'
            #                  , 'mean_team_num_of_high_speed'
            #                  , 'opponent_ave_speed'
            #                  , 'mean_opponent_ave_speed'
            #                  , 'opponent_num_of_high_speed'
            #                  , 'mean_opponent_num_of_high_speed'
            #                  , 'ball_dist_to_closest_oppo', 'closest_opponent_ball_angle', 'closest_opponent_speed'
            #                  , 'opponents_behind_ball', 'last_defender_x', 'dist_ball_last_defender_x'
            #                  , 'mean_ball_angle'
            #                  , 'mean_ball_goalline_angle'
            #                  , 'x_distance_moved'
            #                  , 'start_x'
            #                  , 'end_x'
            #                  , 'mean_ball_dist_to_closest_oppo', 'mean_closest_opponent_ball_angle', 'mean_closest_opponent_speed'
            #                  # , 'mean_dist_convex_x'
            #                  # , 'mean_dist_convex_x_opponent'
            #                  # , 'mean_team_ave_speed'
            #                  # , 'mean_opponents_behind_ball'
            #                  # , 'mean_last_defender_x'
            #                  # , 'mean_dist_ball_last_defender_x'
            #                  , 'Frames_to_Shot'
            #                  # , 'Ten_frames_to_shot'
            #                  , 'Frames_to_Block_inbox'
            #                  , 'Last_episode_opponent'
            #                  , 'Last_episode_length'
            #                  , 'Episode_gap'
            #                  , 'Episode_begin_x'
            #                  , 'Zone_score'
            #                  ]
            self.features = ['Ball_x', 'Ball_y', 'Ball_z'
                , 'Ball_angle'
                , 'Ball_goalline_angle'
                , 'dist_convex_x', 'dist_convex_x_opponent'
                , 'team_ave_speed'
                , 'team_num_of_high_speed'
                , 'opponent_ave_speed'
                , 'opponent_num_of_high_speed'
                , 'ball_dist_to_closest_oppo', 'closest_opponent_ball_angle', 'closest_opponent_speed'
                , 'opponents_behind_ball', 'last_defender_x', 'dist_ball_last_defender_x'
                             ]

    def model_train(self, data, ratio):
        data = data.dropna(subset=['Episode', 'Matchphases_new', 'Aim']).reset_index(drop=True)
        data = data[data['Aim'] == self.aim].reset_index(drop=True)


        train_data = data.sample(frac=ratio)
        test_data = data[~data.index.isin(train_data.index)]

        Xtrain = train_data[self.features].fillna(-1)
        Ytrain = train_data['Matchphases_new']

        Xtest = test_data[self.features].fillna(-1)
        Ytest = test_data['Matchphases_new']
        print(f'Training {self.aim} model')
        self.clf.fit(Xtrain, Ytrain)
        probas_ = self.clf.predict_proba(Xtest)
        accuracy = self.clf.score(Xtest, Ytest)
        y_pred_test = self.clf.predict(Xtest)
        y_pred_train = self.clf.predict(Xtrain)
        pre = precision_score(Ytest, y_pred_test
                              , average='macro'
                              )
        rec = recall_score(Ytest, y_pred_test
                           , average='macro'
                           )
        f1 = f1_score(Ytest, y_pred_test
                      , average='macro'
                      )
        columns = ['acc',
                   # 'auc',
                   'Precision', 'Recall', 'F1']
        # print(mean_auc)
        values = np.array([accuracy,
                           # mean_auc,
                           pre, rec, f1]).reshape(1, 4)
        report_df = pd.DataFrame(values, columns=columns)
        C1 = confusion_matrix(Ytest, y_pred_test, labels=self.clf.classes_)  # True_label 真实标签 shape=(n,1);T_predict1 预测标签 shape=(n,1)
        print(C1)
        print(report_df)

        return Ytrain, y_pred_train


    def predict_original(self, data, report=False):
        data = data.dropna(subset=['Episode', 'Matchphases_new', 'Aim']).reset_index(drop=True)
        data = data[data['Aim'] == self.aim].reset_index(drop=True)
        X = data[self.features].fillna(-1)
        print(f'Predicting {self.aim} from original aim')
        pred = self.clf.predict(X)
        pred_proba = self.clf.predict_proba(X)
        data['Phase_pred_original_aim'] = pred
        if report:
            Y = data['Matchphases_new']
            accuracy = accuracy_score(Y, data['Phase_pred_original_aim'])
            pre = precision_score(Y, data['Phase_pred_original_aim']
                                  , average='macro'
                                  )
            rec = recall_score(Y, data['Phase_pred_original_aim']
                               , average='macro'
                               )
            f1 = f1_score(Y, data['Phase_pred_original_aim']
                          , average='macro'
                          )
            columns = ['acc',
                       # 'auc',
                       'Precision', 'Recall', 'F1']
            # print(mean_auc)
            values = np.array([accuracy,
                               # mean_auc,
                               pre, rec, f1]).reshape(1, 4)
            report_df = pd.DataFrame(values, columns=columns)
            print(report_df)

        return data

    def predict_unfiltered(self, data, report=False):
        data = data.dropna(subset=['Episode', 'Aim_pred']).reset_index(drop=True)
        data = data[data['Aim_pred'] == self.aim].reset_index(drop=True)
        X = data[self.features].fillna(-1)
        print(f'Predicting {self.aim} from unfiltered aim')
        pred = self.clf.predict(X)
        pred_proba = self.clf.predict_proba(X)
        data['Phase_pred_unfiltered_aim'] = pred
        if report:
            data_drop = data.dropna(subset=['Episode', 'Matchphases_new', 'Aim_pred']).reset_index(drop=True)
            Y = data_drop['Matchphases_new']
            accuracy = accuracy_score(Y, data_drop['Phase_pred_unfiltered_aim'])
            pre = precision_score(Y, data_drop['Phase_pred_unfiltered_aim']
                                  , average='weighted'
                                  )
            rec = recall_score(Y, data_drop['Phase_pred_unfiltered_aim']
                               , average='weighted'
                               )
            f1 = f1_score(Y, data_drop['Phase_pred_unfiltered_aim']
                          , average='weighted'
                          )
            columns = ['acc',
                       # 'auc',
                       'Precision', 'Recall', 'F1']
            # print(mean_auc)
            values = np.array([accuracy,
                               # mean_auc,
                               pre, rec, f1]).reshape(1, 4)
            report_df = pd.DataFrame(values, columns=columns)
            print(report_df)

        return data

    def predict_filtered(self, data, report=False, fix=False):
        data = data.dropna(subset=['Episode', 'Aim_filtered']).reset_index(drop=True)
        data = data[data['Aim_filtered'] == self.aim].reset_index(drop=True)
        X = data[self.features].fillna(-1)
        print(f'Predicting {self.aim} from filtered aim')
        pred = self.clf.predict(X)
        pred_proba = self.clf.predict_proba(X)
        classes = self.clf.classes_
        print(list(range(len(classes))), classes)
        class_to_index = dict(zip(classes, list(range(len(classes)))))
        predicted_probas = [pred_proba[i][class_to_index[pred[i]]] for i in range(len(pred))]

        data['Phase_pred_filtered_aim'] = pred
        data['Phase_pred_proba'] = predicted_probas

        if fix:
            for idx in data[data['Action'] == 'ShotAtGoal'].index:
                start_idx = max(0, idx - 10)
                data.loc[start_idx:idx, 'Phase_pred_filtered_aim'] = 'Finishing'
            if self.new_features:
                ball_angle_feature_name = 'Ball_angle_window_0th_frame'
            else:
                ball_angle_feature_name = 'Ball_angle'
            # data.loc[(data['Zone_score'] > 0.5) & (
            #             data[ball_angle_feature_name] < 90), 'Phase_pred_filtered_aim'] = 'Finishing'

            # for idx in data[(data['Action'] == 'OtherEndAction')
            #                 & (data['Ball_x'] >= 36)
            #                 & (data['Ball_y'] >= -20.15)
            #                 & (data['Ball_y'] <= 20.15)].index:
            #     start_idx = max(0, idx - 10)
            #     data.loc[start_idx:idx, 'Phase_pred_filtered_aim'] = 'Finishing'


        if report:
            data_drop = data.dropna(subset=['Episode', 'Matchphases_new', 'Aim_filtered']).reset_index(drop=True)
            Y = data_drop['Matchphases_new']
            accuracy = accuracy_score(Y, data_drop['Phase_pred_filtered_aim'])
            pre = precision_score(Y, data_drop['Phase_pred_filtered_aim']
                                  , average='weighted'
                                  )
            rec = recall_score(Y, data_drop['Phase_pred_filtered_aim']
                               , average='weighted'
                               )
            f1 = f1_score(Y, data_drop['Phase_pred_filtered_aim']
                          , average='weighted'
                          )
            columns = ['acc',
                       # 'auc',
                       'Precision', 'Recall', 'F1']
            # print(mean_auc)
            values = np.array([accuracy,
                               # mean_auc,
                               pre, rec, f1]).reshape(1, 4)
            report_df = pd.DataFrame(values, columns=columns)
            print(report_df)

        return data

