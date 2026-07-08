from sklearn.ensemble import RandomForestClassifier
from Aim_classification import *
from phase_model_pkg.tools.Filter import *
from Phases_classification import *
from phase_model_pkg.tools.Tester import *
from phase_model_pkg.tools.Fixer import PhaseFixer


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

BVB_FCA_data = pd.read_csv(f'../Demo_Datasets/{match_list[0]}/Feature_demo_{match_list[0]}.csv')
S04_DUS_data = pd.read_csv(f'../Demo_Datasets/{match_list[1]}/Feature_demo_{match_list[1]}.csv')
SVW_WOB_data = pd.read_csv(f'../Demo_Datasets/{match_list[2]}/Feature_demo_{match_list[2]}.csv')

data_list = [
    BVB_FCA_data,
    S04_DUS_data,
    SVW_WOB_data,
]


param_grid_boost = {'n_estimators': [100, 500, 1000], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [4, 5, 6], 'num_leaves': [20, 30, 40]}
param_grid_boost_rdf = {'n_estimators': [100, 500, 1000], 'max_depth': [3, 4, 5, 6]}

rdf_aim = RandomForestClassifier(n_estimators=500, n_jobs=-1, max_depth=3)

aim_classifier = AimClassifier(rdf_aim
                               # , new_features=True
                               )

rdf_invade = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
rdf_scoring = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
# #
phase_classifier_invade = PhasesClassifier(rdf_invade, 'Invade opponent space'
                                           # , new_features=True
                                           )
phase_classifier_scoring = PhasesClassifier(rdf_scoring, 'Scoring'
                                            # , new_features=True
                                            )


aim_f1_train_list = []
aim_f1_test_list = []
aim_f1_invade_train_list = []
aim_f1_invade_test_list = []
aim_f1_keep_train_list = []
aim_f1_keep_test_list = []
aim_f1_scoring_train_list = []
aim_f1_scoring_test_list = []

invade_f1_train_list = []
invade_f1_test_list = []
invade_build_f1_train_list = []
invade_build_f1_test_list = []
invade_Progression_f1_train_list = []
invade_Progression_f1_test_list = []
invade_counter_f1_train_list = []
invade_counter_f1_test_list = []

scoring_f1_train_list = []
scoring_f1_test_list = []
scoring_sustained_f1_train_list = []
scoring_sustained_f1_test_list = []
scoring_finishing_f1_train_list = []
scoring_finishing_f1_test_list = []

last_episode_index = 0
last_aim_index = 0
last_phase_index = 0
final_data_list=[]
for i, test_data in enumerate(data_list):
    train_datas = data_list[:i] + data_list[i + 1:]
    train_data = pd.concat(train_datas).reset_index(drop=True)
    data_drop = train_data.dropna(subset=['Episode', 'Matchphases_new', 'Aim']).reset_index(drop=True)
    data_drop = data_drop.fillna(-1)

    train_Y_aim, train_Y_aim_pred = aim_classifier.model_train(data_drop, 0.9)
    train_Y_aim = train_Y_aim.replace(['Invade opponent space', 'Keep possession', 'Scoring'],
                                                          [0, 1, 2])
    train_Y_aim_pred = pd.Series(train_Y_aim_pred).replace(['Invade opponent space', 'Keep possession', 'Scoring'],
                                                          [0, 1, 2])
    aim_f1_train = f1_score(train_Y_aim, train_Y_aim_pred, average='macro')
    aim_f1_each_train = f1_score(train_Y_aim, train_Y_aim_pred, average=None)
    aim_matrix_train = confusion_matrix(train_Y_aim, train_Y_aim_pred)
    
    print(aim_f1_train)
    print(aim_f1_each_train)
    print(aim_matrix_train)
    aim_f1_train_list.append(aim_f1_train)
    aim_f1_invade_train_list.append(aim_f1_each_train[0])
    aim_f1_keep_train_list.append(aim_f1_each_train[1])
    aim_f1_scoring_train_list.append(aim_f1_each_train[2])
    
    test_data = aim_classifier.predict(test_data)
    test_data_drop = test_data.dropna(subset=['Episode', 'Matchphases_new', 'Aim', 'Aim_pred']).reset_index(drop=True)
    test_Y_aim = test_data_drop['Aim'].replace(['Invade opponent space', 'Keep possession', 'Scoring'],
                                                          [0, 1, 2])
    test_Y_aim_pred = test_data_drop['Aim_pred'].replace(['Invade opponent space', 'Keep possession', 'Scoring'],
                                                          [0, 1, 2])
    aim_f1_test = f1_score(test_Y_aim, test_Y_aim_pred, average='macro')
    aim_f1_each_test = f1_score(test_Y_aim, test_Y_aim_pred, average=None)
    aim_matrix_test = confusion_matrix(test_Y_aim, test_Y_aim_pred)
    print(aim_f1_test)
    print(aim_f1_each_test)
    print(aim_matrix_test)
    aim_f1_test_list.append(aim_f1_test)
    aim_f1_invade_test_list.append(aim_f1_each_test[0])
    aim_f1_keep_test_list.append(aim_f1_each_test[1])
    aim_f1_scoring_test_list.append(aim_f1_each_test[2])
    
    train_Y_invade, train_Y_invade_pred = phase_classifier_invade.model_train(data_drop, 0.9)
    train_Y_invade = train_Y_invade.replace(['Build Up', 'Progression', 'Counter attack'],
                                      [0, 1, 2])
    train_Y_invade_pred = pd.Series(train_Y_invade_pred).replace(['Build Up', 'Progression', 'Counter attack'],
                                                [0, 1, 2])
    invade_f1_train = f1_score(train_Y_invade, train_Y_invade_pred, average='macro')
    invade_f1_each_train = f1_score(train_Y_invade, train_Y_invade_pred, average=None)
    invade_matrix_train = confusion_matrix(train_Y_invade, train_Y_invade_pred)

    print(invade_f1_train)
    print(invade_f1_each_train)
    print(invade_matrix_train)
    invade_f1_train_list.append(invade_f1_train)
    invade_build_f1_train_list.append(invade_f1_each_train[0])
    invade_Progression_f1_train_list.append(invade_f1_each_train[1])
    invade_counter_f1_train_list.append(invade_f1_each_train[2])

    test_data_invade = phase_classifier_invade.predict_original(test_data)
    test_data_drop_invade = test_data_invade.dropna(subset=['Episode', 'Matchphases_new', 'Aim']).reset_index(drop=True)
    test_Y_invade = test_data_drop_invade['Matchphases_new']
    test_Y_invade_pred = test_data_drop_invade['Phase_pred_original_aim']
    test_Y_invade = test_Y_invade.replace(['Build Up', 'Progression', 'Counter attack'],
                                            [0, 1, 2])
    test_Y_invade_pred = test_Y_invade_pred.replace(['Build Up', 'Progression', 'Counter attack'],
                                                      [0, 1, 2])
    invade_f1_test = f1_score(test_Y_invade, test_Y_invade_pred, average='macro')
    invade_f1_each_test = f1_score(test_Y_invade, test_Y_invade_pred, average=None)
    invade_matrix_test = confusion_matrix(test_Y_invade, test_Y_invade_pred)

    print(invade_f1_test)
    print(invade_f1_each_test)
    print(invade_matrix_test)
    invade_f1_test_list.append(invade_f1_test)
    invade_build_f1_test_list.append(invade_f1_each_test[0])
    invade_Progression_f1_test_list.append(invade_f1_each_test[1])
    invade_counter_f1_test_list.append(invade_f1_each_test[2])
    
    
    train_Y_scoring, train_Y_scoring_pred = phase_classifier_scoring.model_train(data_drop, 0.9)
    train_Y_scoring = train_Y_scoring.replace(['Sustained Threat', 'Finishing'],
                                            [0, 1])
    train_Y_scoring_pred = pd.Series(train_Y_scoring_pred).replace(['Sustained Threat', 'Finishing'],
                                            [0, 1])
    scoring_f1_train = f1_score(train_Y_scoring, train_Y_scoring_pred, average='macro')
    scoring_f1_each_train = f1_score(train_Y_scoring, train_Y_scoring_pred, average=None)
    scoring_matrix_train = confusion_matrix(train_Y_scoring, train_Y_scoring_pred)

    print(scoring_f1_train)
    print(scoring_f1_each_train)
    print(scoring_matrix_train)
    scoring_f1_train_list.append(scoring_f1_train)
    scoring_sustained_f1_train_list.append(scoring_f1_each_train[0])
    scoring_finishing_f1_train_list.append(scoring_f1_each_train[1])
    
    test_data_scoring = phase_classifier_scoring.predict_original(test_data)
    test_data_drop_scoring = test_data_scoring.dropna(subset=['Episode', 'Matchphases_new', 'Aim']).reset_index(drop=True)
    test_Y_scoring = test_data_scoring['Matchphases_new']
    test_Y_scoring_pred = test_data_scoring['Phase_pred_original_aim']
    test_Y_scoring = test_Y_scoring.replace(['Sustained Threat', 'Finishing'],
                                            [0, 1])
    test_Y_scoring_pred = test_Y_scoring_pred.replace(['Sustained Threat', 'Finishing'],
                                            [0, 1])
    scoring_f1_test = f1_score(test_Y_scoring, test_Y_scoring_pred, average='macro')
    scoring_f1_each_test = f1_score(test_Y_scoring, test_Y_scoring_pred, average=None)
    scoring_matrix_test = confusion_matrix(test_Y_scoring, test_Y_scoring_pred)

    print(scoring_f1_test)
    print(scoring_f1_each_test)
    print(scoring_matrix_test)
    scoring_f1_test_list.append(scoring_f1_test)
    scoring_sustained_f1_test_list.append(scoring_f1_each_test[0])
    scoring_finishing_f1_test_list.append(scoring_f1_each_test[1])

    pred_data = aim_classifier.predict(test_data, report=True)
    # Filter Aims
    aim_pred_data = pred_data[['Frame', 'Episode', 'Aim', 'Aim_pred']]
    aim_filter = AimFilter(aim_pred_data)
    aim_pred_data = aim_filter.filter(75, 50, 25)
    pred_data['Aim_filtered'] = aim_pred_data['Aim_filtered']
    pred_data = generate_indices(pred_data, 'Episode', 'Aim_filtered')

    invade_data = phase_classifier_invade.predict_original(pred_data
                                                           , report=True
                                                           )
    scoring_data = phase_classifier_scoring.predict_original(pred_data
                                                             , report=True
                                                           )
    keep_data = pred_data[pred_data['Aim'] == 'Keep possession'].reset_index(drop=True)
    keep_data['Phase_pred_original_aim'] = keep_data['Matchphases_new']
    new_data = pd.concat([invade_data, scoring_data, keep_data]).sort_values(by='Frame').reset_index(drop=True)
    pred_data = pred_data.merge(new_data, how='left').reset_index(drop=True)

    invade_data = phase_classifier_invade.predict_unfiltered(pred_data
                                                             , report=True
                                                           )
    scoring_data = phase_classifier_scoring.predict_unfiltered(pred_data
                                                               , report=True
                                                           )
    keep_data = pred_data[pred_data['Aim_pred'] == 'Keep possession'].reset_index(drop=True)
    keep_data['Phase_pred_unfiltered_aim'] = ['Maintenance'] * len(keep_data)
    new_data = pd.concat([invade_data, scoring_data, keep_data]).sort_values(by='Frame').reset_index(drop=True)
    pred_data = pred_data.merge(new_data, how='left').reset_index(drop=True)


    invade_data = phase_classifier_invade.predict_filtered(pred_data
                                                           , report=True
                                                           # , fix=True
                                                           )
    scoring_data = phase_classifier_scoring.predict_filtered(pred_data
                                                             , report=True
                                                             # , fix=True
                                                             )
    keep_data = pred_data[pred_data['Aim_filtered'] == 'Keep possession'].reset_index(drop=True)
    keep_data['Phase_pred_filtered_aim'] = ['Maintenance'] * len(keep_data)
    new_data = pd.concat([invade_data, scoring_data, keep_data]).sort_values(by='Frame').reset_index(drop=True)
    pred_data = pred_data.merge(new_data, how='left').reset_index(drop=True)


    # phase filter
    phase_pred_data = pred_data[['Frame', 'Episode', 'Episode_indices'
        , 'Aim', 'Aim_filtered', 'Aim_filtered_indices'
        , 'Matchphases_new', 'Phase_pred_filtered_aim', 'Action'
                                 ]]
    phase_filter = PhaseFilter(phase_pred_data)
    phase_pred_data = phase_filter.filter(50, 10)

    # phase fixer
    phase_fixer = PhaseFixer(phase_pred_data)
    phase_pred_data = phase_fixer.fix()
    phase_pred_data = generate_indices(phase_pred_data, 'Aim_filtered', 'Phase_pred_filtered_aim_filtered_fixed')
    pred_data = pred_data.merge(phase_pred_data, how='left')
    pred_data.to_csv(f'Matchphases_{match_list[i]}_Phase_pred_random_forest_demo.csv')
    last_episode_index += phase_pred_data['Episode_indices'].unique()[-1]
    last_aim_index += phase_pred_data['Aim_filtered_indices'].unique()[-1]
    last_phase_index += phase_pred_data['Phase_pred_filtered_aim_filtered_fixed_indices'].unique()[-1]
    final_data_list.append(pred_data)

f1_df = pd.DataFrame({
    'Intention_model_test_f1': aim_f1_test_list,
    'Intention_model_invade_test_f1': aim_f1_invade_test_list,
    'Intention_model_keep_test_f1': aim_f1_keep_test_list,
    'Intention_model_scoring_test_f1': aim_f1_scoring_test_list,
    'Invade_model_test_f1': invade_f1_test_list,
    'Invade_model_build_test_f1': invade_build_f1_test_list,
    'Invade_model_Progression_test_f1': invade_Progression_f1_test_list,
    'Invade_model_counter_test_f1': invade_counter_f1_test_list,
    'Scoring_model_test_f1': scoring_f1_test_list,
    'Scoring_model_sustained_test_f1': scoring_sustained_f1_test_list,
    'Scoring_model_finishing_test_f1': scoring_finishing_f1_test_list
})
f1_df.to_csv('Test_f1_per_match_random_forest.csv', index=False)

mean = f1_df.mean()
std = f1_df.std()

formatted = mean.round(2).astype(str) + ' ± ' + std.round(2).astype(str)

result_df = pd.DataFrame(formatted).T
result_df.index = ['Mean ± Std']

print("\nF1 results：")
print(result_df)
