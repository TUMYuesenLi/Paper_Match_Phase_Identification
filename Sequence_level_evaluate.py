from phase_model_pkg.tools.Tester import Tester
import pandas as pd

PHASE_ORDER = [
    "Build Up",
    "Progression",
    "Counter attack",
    "Maintenance",
    "Sustained Threat",
    "Finishing",
]

AIM_ORDER = [
    "Invade opponent space",
    "Keep possession",
    "Scoring"
]


def reorder_to_fixed_phases(mat, order):
    mat = mat.copy()

    # add missing
    for ph in order:
        if ph not in mat.index:
            mat.loc[ph] = 0.0
        if ph not in mat.columns:
            mat[ph] = 0.0

    # ensure square alignment
    mat = mat.reindex(index=mat.index, columns=mat.columns, fill_value=0.0)

    extras = sorted(set([x for x in mat.index if x not in order] + [x for x in mat.columns if x not in order]))
    final_order = order + extras

    return mat.reindex(index=final_order, columns=final_order, fill_value=0.0)


def generate_iou_matrices(data, model_name):
    data = data.dropna(subset=['Episode', 'Matchphases_new', 'Aim', 'Aim_pred', 'Aim_filtered', 'Aim_filtered_indices',
            'Phase_pred_filtered_aim', 'Phase_pred_filtered_aim_filtered',
            'Phase_pred_filtered_aim_filtered_fixed',
            'Phase_pred_filtered_aim_filtered_fixed_indices']).reset_index(drop=True)
    tester = Tester(data)
    column_aim = 'Aim'
    column_pred_aim = 'Aim_pred'
    column_filtered_aim = 'Aim_filtered'

    column_phase = 'Matchphases_new'
    column_pred_phase = 'Phase_pred_filtered_aim'
    column_filtered_phase = 'Phase_pred_filtered_aim_filtered'
    column_fixed_phase = 'Phase_pred_filtered_aim_filtered_fixed'

    iou_matrix_aim = tester.iou_matrix(column_aim, column_pred_aim
                                       # , recall=True
                                       )
    
    iou_matrix_aim = reorder_to_fixed_phases(iou_matrix_aim, order=AIM_ORDER)
    iou_matrix_aim_filtered = tester.iou_matrix(column_aim, column_filtered_aim
                                                # , recall=True
                                                )
    iou_matrix_aim_filtered = reorder_to_fixed_phases(iou_matrix_aim_filtered, order=AIM_ORDER)


    iou_matrix_phase = tester.iou_matrix(column_phase, column_pred_phase
                                         # , recall=True
                                         )
    iou_matrix_phase = reorder_to_fixed_phases(iou_matrix_phase, order=PHASE_ORDER)
    iou_matrix_phase_filtered = tester.iou_matrix(column_phase, column_filtered_phase
                                                  # , recall=True
                                                  )
    iou_matrix_phase_filtered = reorder_to_fixed_phases(iou_matrix_phase_filtered, order=PHASE_ORDER)
    iou_matrix_phase_fixed = tester.iou_matrix(column_phase, column_fixed_phase
                                               # , recall=True
                                               )
    iou_matrix_phase_fixed = reorder_to_fixed_phases(iou_matrix_phase_fixed, order=PHASE_ORDER)



    iou_matrix_aim_recall = tester.iou_matrix(column_aim, column_pred_aim
                                              , recall=True
                                              )
    iou_matrix_aim_recall = reorder_to_fixed_phases(iou_matrix_aim_recall, order=AIM_ORDER)

    iou_matrix_aim_filtered_recall = tester.iou_matrix(column_aim, column_filtered_aim
                                                       , recall=True
                                                       )
    iou_matrix_aim_filtered_recall = reorder_to_fixed_phases(iou_matrix_aim_filtered_recall, order=AIM_ORDER)


    iou_matrix_phase_recall = tester.iou_matrix(column_phase, column_pred_phase
                                                , recall=True
                                                )
    iou_matrix_phase_recall = reorder_to_fixed_phases(iou_matrix_phase_recall, order=PHASE_ORDER)
    iou_matrix_phase_filtered_recall = tester.iou_matrix(column_phase, column_filtered_phase
                                                         , recall=True
                                                         )
    iou_matrix_phase_filtered_recall = reorder_to_fixed_phases(iou_matrix_phase_filtered_recall, order=PHASE_ORDER)
    iou_matrix_phase_fixed_recall = tester.iou_matrix(column_phase, column_fixed_phase
                                                      , recall=True
                                                      )
    iou_matrix_phase_fixed_recall = reorder_to_fixed_phases(iou_matrix_phase_fixed_recall, order=PHASE_ORDER)

    iou_matrix_aim_f1 = (2 * iou_matrix_aim * iou_matrix_aim_recall) / (
                iou_matrix_aim + iou_matrix_aim_recall)
    iou_matrix_aim_filtered_f1 = (2 * iou_matrix_aim_filtered * iou_matrix_aim_filtered_recall) / (
                iou_matrix_aim_filtered + iou_matrix_aim_filtered_recall)


    iou_matrix_phase_f1 = (2 * iou_matrix_phase * iou_matrix_phase_recall) / (
                iou_matrix_phase + iou_matrix_phase_recall)
    iou_matrix_phase_filtered_f1 = (2 * iou_matrix_phase_filtered * iou_matrix_phase_filtered_recall) / (
            iou_matrix_phase_filtered + iou_matrix_phase_filtered_recall)
    iou_matrix_phase_fixed_f1 = (2 * iou_matrix_phase_fixed * iou_matrix_phase_fixed_recall) / (
            iou_matrix_phase_fixed + iou_matrix_phase_fixed_recall)

    matrices_dict = {f"{model_name} Intention Unfiltered": iou_matrix_aim_f1.fillna(0).round(2)
                     , f"{model_name} Intention Processed": iou_matrix_aim_filtered_f1.fillna(0).round(2)
                     , f"{model_name} Phase Unfiltered": iou_matrix_phase_f1.fillna(0).round(2)
                     # , f"{model_name} Phase Filtered": iou_matrix_phase_filtered_f1.fillna(0)
                     , f"{model_name} Phase Processed": iou_matrix_phase_fixed_f1.fillna(0).round(2)}

    return matrices_dict


data_folder = "Demo_Datasets"

match_list = [
    'DFL-MAT-003AXI_Borussia Dortmund-FC Augsburg_2018-2019',
    'DFL-MAT-003AXO_Fortuna Düsseldorf 1895 e.V.-FC Schalke 04_2018-2019',
    'DFL-MAT-003AXK_SV Werder Bremen-VfL Wolfsburg_2018-2019',
]
match_team_list = [
    'BVB_FCA',
    'S04_DÜS',
    'SVW_WOB',
]

T_GNN_list = []
Transformer_list = []
GNN_list = []
Rule_based_list = []
RF_list = []

TGNN_dict_list = []
dict_list = []

for i in range(len(match_team_list)):
    TGNN_match = pd.read_csv(f'{data_folder}/Matchphases_{match_team_list[i]}_Phase_pred_T-GAN_demo')
    T_GNN_list.append(TGNN_match)
    Transformer_match = pd.read_csv(f'{data_folder}'
                                    f'/Matchphases_{match_team_list[i]}_Phase_pred_pure_transformer_demo.csv')
    Transformer_list.append(Transformer_match)
    GNN_match = pd.read_csv(f'{data_folder}'
                                    f'/Matchphases_{match_team_list[i]}_Phase_pred_pure_GNN_demo.csv')
    GNN_list.append(GNN_match)
    Rule_based_match = pd.read_csv(f'{data_folder}'
                                    f'/Matchphases_{match_team_list[i]}_Phase_pred_rule_based_demo.csv')
    Rule_based_list.append(Rule_based_match)
    RF_match = pd.read_csv(f'{data_folder}'
                                    f'/Matchphases_{match_list[i]}_Phase_pred_random_forest_demo.csv')
    RF_list.append(RF_match)
    TGNN_dict = generate_iou_matrices(TGNN_match, 'T-GAN')
    TGNN_dict_list.append(TGNN_dict)
    Transformer_dict = generate_iou_matrices(Transformer_match, 'Pure Transformer')
    GNN_dict = generate_iou_matrices(GNN_match, 'Pure GNN')
    Rule_dict = generate_iou_matrices(Rule_based_match, 'Rule-based')
    RF_dict = generate_iou_matrices(RF_match, 'Random Forest')
    merged_dict = TGNN_dict | Transformer_dict | GNN_dict | Rule_dict | RF_dict
    dict_list.append(merged_dict)



