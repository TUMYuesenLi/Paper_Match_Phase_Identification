import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import *


class AimClassifier:

    def __init__(self, clf, xgb=False):
        self.clf = clf
        self.features = ['Ball_x'
            , 'dist_convex_x', 'dist_convex_x_opponent'
            , 'ball_dist_to_closest_oppo', 'closest_opponent_ball_angle', 'closest_opponent_speed'
            , 'opponents_behind_ball', 'last_defender_x', 'dist_ball_last_defender_x'
                         ]

        self.xgb = xgb


    def model_train(self, data, ratio):
        data_drop = data.dropna(subset=['Episode', 'Matchphases_new', 'Aim']).reset_index(drop=True)
        data_drop = data_drop.fillna(-1)
        if self.xgb:
            data_drop['Aim'] = data_drop['Aim'].replace(['Invade opponent space', 'Keep possession', 'Scoring'],
                                                          [0, 1, 2])
        train_data = data_drop.sample(frac=ratio)
        test_data = data_drop[~data_drop.index.isin(train_data.index)]
        Xtrain = train_data[self.features]
        Ytrain = train_data['Aim']
        print(Xtrain.columns)

        Xtest = test_data[self.features]
        Ytest = test_data['Aim']

        print(f'Training Aim model')
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
        df = pd.DataFrame(C1, index=self.clf.classes_,
                          columns=self.clf.classes_)
        sns.heatmap(df, annot=True)
        plt.title("Aim training")
        plt.show()
        print(report_df)

        return Ytrain, y_pred_train


    def predict(self, data, report=False):
        data_drop_episode = data.dropna(subset=['Episode']).reset_index(drop=True)
        if self.xgb:
            data_drop_episode['Aim'] = data_drop_episode['Aim'].replace(['Invade opponent space', 'Keep possession', 'Scoring'],
                                                          [0, 1, 2])
        X = data_drop_episode[self.features].fillna(-1)
        print(f'Predicting Aims')
        pred = self.clf.predict(X)
        pred_proba = self.clf.predict_proba(X)
        data_drop_episode['Aim_pred'] = pred
        data_drop_episode['pred_index'] = data_drop_episode.index
        if report:
            data_drop_episode = data_drop_episode.dropna(subset=['Episode', 'Aim', 'Aim_pred']).reset_index(drop=True)
            Y = data_drop_episode['Aim']
            accuracy = accuracy_score(Y, data_drop_episode['Aim_pred'])
            pre = precision_score(Y, data_drop_episode['Aim_pred']
                                  , average='macro'
                                  )
            rec = recall_score(Y, data_drop_episode['Aim_pred']
                               , average='macro'
                               )
            f1 = f1_score(Y, data_drop_episode['Aim_pred']
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
            C1 = confusion_matrix(Y, data_drop_episode['Aim_pred'], labels=self.clf.classes_)
            print(C1)
            df = pd.DataFrame(C1, index=self.clf.classes_,
                              columns=self.clf.classes_)
            sns.heatmap(df, annot=True)
            plt.title('Aim test')
            plt.show()
            print(report_df)

        data_drop_episode = data_drop_episode[['Frame', 'pred_index', 'Aim_pred']]
        data = data.merge(data_drop_episode, how='left').reset_index(drop=True)
        if self.xgb:
            data['Aim_pred'] = data['Aim_pred'].replace(
                [0, 1, 2],
                ['Invade opponent space', 'Keep possession', 'Scoring'])
            data['Aim'] = data['Aim'].replace(
                [0, 1, 2],
                ['Invade opponent space', 'Keep possession', 'Scoring'])
        return data



