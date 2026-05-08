import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import *

pd.set_option('display.max_columns', None)

class Tester:

    def __init__(self, data):
        self.data = data
        # self.ground_truth = ground_truth
        # self.pred = pred
        # self.start_indices = []
        # self.stop_indices = []

    def get_length(self, ordinary_column):
        counts = []
        indices = []
        for k, v in enumerate(itertools.groupby(self.data[ordinary_column])):
            aims = list(v[1])
            length = [len(aims)] * len(aims)
            aim_indices = [k] * len(aims)
            counts.extend(length)
            indices.extend(aim_indices)
        self.data[f'{ordinary_column}_length'] = counts
        self.data[f'{ordinary_column}_indices'] = indices

    def get_start_stop(self, ordinary_column):
        start_indices = []
        stop_indices = []
        data_dup = self.data.duplicated(subset=[ordinary_column, f'{ordinary_column}_length', f'{ordinary_column}_indices'])
        self.data[f'dup_{ordinary_column}'] = data_dup
        nan_r = np.where(~data_dup)
        start_stop = [index for index in nan_r[0]]

        for k, v in enumerate(itertools.groupby(self.data[ordinary_column])):
            aims = list(v[1])
            start_index = start_stop[k]
            try:
                stop_index = start_stop[k + 1] - 1
            except :
                stop_index = len(self.data) - 1
            start_indices.extend([start_index] * len(aims))
            stop_indices.extend([stop_index] * len(aims))

        self.data[f'{ordinary_column}_start'] = start_indices
        self.data[f'{ordinary_column}_stop'] = stop_indices

    def get_interval_data(self, column):
        interval_data = self.data[self.data[f'dup_{column}'] == False].reset_index(drop=True)
        return interval_data

    def test(self, ordinary_column, test_column, compute_start_stop=False, interval=False):
        if compute_start_stop:
            self.get_length(ordinary_column)
            self.get_start_stop(ordinary_column)
        if not interval:
            ordinary_interval = self.get_interval_data(ordinary_column)
        else:
            ordinary_interval = interval
        accuracy_list = []
        for index, row in ordinary_interval.iterrows():
            start_index = row[f'{ordinary_column}_start']
            stop_index = row[f'{ordinary_column}_stop']
            interval_data = self.data.iloc[start_index: stop_index+1]
            y_true = interval_data[ordinary_column]
            y_test = interval_data[test_column]
            accuracy = accuracy_score(y_true, y_test)
            accuracy_list.append(accuracy)
        mean_acc = np.mean(accuracy_list)
        return mean_acc

    def test_iou(self, ordinary_column, test_column, obs_label
                 , interval, pred_label=False, compute_start_stop=False, if_number=False, out_list=False):
        iou_num = 0
        if compute_start_stop:
            self.get_length(ordinary_column)
            self.get_start_stop(ordinary_column)
            self.get_length(test_column)
            self.get_start_stop(test_column)
        if interval is None:
            ordinary_interval = self.get_interval_data(ordinary_column)
        else:
            ordinary_interval = interval
        iou_list = []
        for index, row in ordinary_interval.iterrows():
            if row[ordinary_column] == obs_label:
                start_index = row[f'{ordinary_column}_start']
                stop_index = row[f'{ordinary_column}_stop']
                interval_data = self.data.iloc[start_index: stop_index+1]
                y_true = interval_data[ordinary_column]
                if pred_label:
                    y_test = interval_data[interval_data[test_column] == pred_label]
                    # print(y_test)
                    length_list = []
                    for k, v in enumerate(itertools.groupby(y_test[f'{test_column}_indices'])):
                        aims = list(v[1])
                        length = len(aims)
                        length_list.append(length)
                    # if len(length_list) > 1:
                    #     print(length_list)
                    # else:
                    #     continue
                    if length_list:
                        iou = max(length_list) / len(y_true)
                    else:
                        iou = 0
                    # iou = len(y_test) / len(y_true)
                else:
                    y_test = interval_data[test_column]
                    iou = accuracy_score(y_true, y_test)
                iou_list.append(iou)
                if iou >= 0.5:
                    iou_num += 1
                else:
                    continue
            else:
                continue
        if if_number:
            return iou_num
        elif out_list:
            return iou_list
        else:
            return np.mean(iou_list)

    def iou_matrix(self, ordinary_column, pred_column, if_number=False, ratio=False, recall=False):
        if recall:
            true_labels = self.data[pred_column].unique()
            validate_labels = self.data[ordinary_column].unique()
            # true_data = pred_data
            # validate_data = ordinary_data
            true_column = pred_column
            validate_column = ordinary_column
        else:
            true_labels = self.data[ordinary_column].unique()
            validate_labels = self.data[pred_column].unique()
            # true_data = ordinary_data
            # validate_data = pred_data
            true_column = ordinary_column
            validate_column = pred_column
        self.get_length(true_column)
        self.get_start_stop(true_column)
        self.get_length(validate_column)
        self.get_start_stop(validate_column)
        ordinary_interval = self.get_interval_data(true_column)
        iou_matrix = np.zeros((len(true_labels), len(true_labels)))
        length_list = []
        for i, obs in enumerate(true_labels):
            ordinary_df = ordinary_interval[ordinary_interval[true_column] == obs].reset_index(drop=True)
            ordinary_len = len(ordinary_df)
            length_list.append(ordinary_len)
            for j, pred in enumerate(true_labels):
                iou = self.test_iou(true_column, validate_column, obs, pred_label=pred
                                    , interval=ordinary_interval, compute_start_stop=False
                                    , if_number=if_number)
                # print(iou)
                iou_matrix[i][j] = iou
        length_series = pd.Series(length_list, index=true_labels)
        iou_matrix_df = pd.DataFrame(iou_matrix, columns=true_labels
                                     , index=true_labels)
        if ratio:
            return iou_matrix_df.divide(length_series, axis=0)
        else:
            return iou_matrix_df
