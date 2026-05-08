from phase_model_pkg.tools.Filter import *
from phase_model_pkg.tools.Tester import *
from phase_model_pkg.tools.Fixer import PhaseFixer
from phase_model_pkg.tools.label_utils import generate_indices, aim_transform, phase_transform


class RuleIdentifier:

    def __init__(self):
        self.court_left = -52.5
        self.court_right = 52.5
        self.defensive_third = -17.5
        self.attacking_third = 17.5
        self.box_left = 36
        self.box_bottom = -20.16
        self.box_top = 20.16

    def set_rules_invade(self, x):
        if -75 <= x['Ball_angle'] <= 75:
            if (x['Last_episode_opponent'] == 1) and (x['Last_episode_length'] >= 150) and (x['Episode_gap'] <= 25):
                return 'Counter attack'
            else:
                return 'Progression'

        else:
            return 'Build Up'

    def set_rules_scoring(self, x):
        if (x['Ball_x'] > 36) and (-20.16 < x['Ball_y'] < 20.16):
            return 'Finishing'
        else:
            return 'Sustained Threat'

    def set_rules_aim(self, x):

        if -52.5 <= x['Ball_x'] < -17.5:
            return 'Keep possession'
        elif -17.5 < x['Ball_x'] <= 17.5:
            return 'Invade opponent space'
        else:
            return 'Scoring'

    def predict_aim(self, data):
        data['Aim_pred'] = data.apply(self.set_rules_aim, axis=1)
        return data

    def predict_invade(self, data, phase_column):
        data[phase_column] = data.apply(self.set_rules_invade, axis=1)
        return data
    def predict_scoring(self, data, phase_column):
        data[phase_column] = data.apply(self.set_rules_scoring, axis=1)
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

BVB_FCA_data = pd.read_csv(f'../Demo_Datasets/{match_list[0]}/Feature_demo_{match_team_list[0]}.csv')
S04_DUS_data = pd.read_csv(f'../Demo_Datasets/{match_list[1]}/Feature_demo_{match_team_list[1]}.csv')
SVW_WOB_data = pd.read_csv(f'../Demo_Datasets/{match_list[2]}/Feature_demo_{match_team_list[2]}.csv')

pred_data_list = [
    BVB_FCA_data
    , S04_DUS_data
    , SVW_WOB_data
]

last_episode_index = 0
last_aim_index = 0
last_phase_index = 0
final_data_list = []
identifier = RuleIdentifier()

aim_f1_test_list = []
aim_f1_invade_test_list = []
aim_f1_keep_test_list = []
aim_f1_scoring_test_list = []

invade_f1_test_list = []
invade_build_f1_test_list = []
invade_Progression_f1_test_list = []
invade_counter_f1_test_list = []

scoring_f1_test_list = []
scoring_sustained_f1_test_list = []
scoring_finishing_f1_test_list = []
for i, pred_data in enumerate(pred_data_list):
    pred_data = pred_data.dropna(subset=['Episode', 'Matchphases_new', 'Aim']).reset_index(drop=True)
    aim_pred_data = identifier.predict_aim(pred_data)
    pred_data["Aim_number"] = pred_data['Aim'].apply(aim_transform)
    pred_data["Aim_pred_number"] = pred_data['Aim_pred'].apply(aim_transform)
    aim_f1 = f1_score(pred_data['Aim_number'], pred_data['Aim_pred_number'], average='macro')
    aim_f1_each = f1_score(pred_data['Aim_number'], pred_data['Aim_pred_number'], average=None)
    aim_matrix = confusion_matrix(pred_data['Aim_number'], pred_data['Aim_pred_number'])
    print(aim_f1)
    print(aim_f1_each)
    print(aim_matrix)
    aim_f1_test_list.append(aim_f1)
    aim_f1_invade_test_list.append(aim_f1_each[0])
    aim_f1_keep_test_list.append(aim_f1_each[1])
    aim_f1_scoring_test_list.append(aim_f1_each[2])

    invade_data = pred_data[pred_data['Aim'] == 'Invade opponent space']
    invade_pred_data = identifier.predict_invade(invade_data, 'Phase_pred')
    invade_pred_data["Phase_number"] = invade_pred_data['Matchphases_new'].apply(phase_transform)
    invade_pred_data["Phase_pred_number"] = invade_pred_data['Phase_pred'].apply(phase_transform)
    invade_f1 = f1_score(invade_pred_data['Phase_number'], invade_pred_data['Phase_pred_number'], average='macro')
    invade_f1_each = f1_score(invade_pred_data['Phase_number'], invade_pred_data['Phase_pred_number'], average=None)
    invade_matrix = confusion_matrix(invade_pred_data['Phase_number'], invade_pred_data['Phase_pred_number'])
    print(invade_f1)
    print(invade_f1_each)
    print(invade_matrix)
    invade_f1_test_list.append(invade_f1)
    invade_build_f1_test_list.append(invade_f1_each[0])
    invade_Progression_f1_test_list.append(invade_f1_each[1])
    invade_counter_f1_test_list.append(invade_f1_each[2])
    
    scoring_data = pred_data[pred_data['Aim'] == 'Scoring']
    scoring_pred_data = identifier.predict_scoring(scoring_data, 'Phase_pred')
    scoring_pred_data["Phase_number"] = scoring_pred_data['Matchphases_new'].apply(phase_transform)
    scoring_pred_data["Phase_pred_number"] = scoring_pred_data['Phase_pred'].apply(phase_transform)
    scoring_f1 = f1_score(scoring_pred_data['Phase_number'], scoring_pred_data['Phase_pred_number'], average='macro')
    scoring_f1_each = f1_score(scoring_pred_data['Phase_number'], scoring_pred_data['Phase_pred_number'], average=None)
    scoring_matrix = confusion_matrix(scoring_pred_data['Phase_number'], scoring_pred_data['Phase_pred_number'])
    print(scoring_f1)
    print(scoring_f1_each)
    print(scoring_matrix)
    scoring_f1_test_list.append(scoring_f1)
    scoring_sustained_f1_test_list.append(scoring_f1_each[0])
    scoring_finishing_f1_test_list.append(scoring_f1_each[1])

    # final prediction process

    # Aim filter
    aim_filter = AimFilter(pred_data)
    pred_data = aim_filter.filter(50, 50, 25)
    pred_data = generate_indices(pred_data, 'Episode', 'Aim_filtered')



    invade_data = pred_data[pred_data['Aim_filtered'] == 'Invade opponent space']
    invade_data = identifier.predict_invade(invade_data, 'Phase_pred_filtered_aim')
    scoring_data = pred_data[pred_data['Aim_filtered'] == 'Scoring']
    scoring_data = identifier.predict_scoring(scoring_data, 'Phase_pred_filtered_aim')
    keep_data = pred_data[pred_data['Aim_filtered'] == 'Keep possession'].reset_index(drop=True)
    keep_data['Phase_pred_filtered_aim'] = 'Maintenance'

    new_data = pd.concat([invade_data, scoring_data, keep_data]).sort_values(by='Frame').reset_index(drop=True)
    pred_data = pred_data.merge(new_data, how='left').reset_index(drop=True)

    # phase filter
    phase_pred_data = pred_data[['Frame', 'Episode', 'Episode_indices'
        , 'Aim', 'Aim_pred', 'Aim_filtered', 'Aim_filtered_indices'
        , 'Matchphases_new', 'Phase_pred_filtered_aim', 'Action'
                                 ]]
    phase_filter = PhaseFilter(phase_pred_data)
    phase_pred_data = phase_filter.filter(50, 10)

    # phase fixer
    phase_fixer = PhaseFixer(phase_pred_data)
    phase_pred_data = phase_fixer.fix()
    phase_pred_data = generate_indices(phase_pred_data, 'Aim_filtered', 'Phase_pred_filtered_aim_filtered_fixed')
    phase_pred_data.to_csv(f'Matchphases_{match_team_list[i]}_Phase_pred_rule-based_demp.csv')
    # phase_pred_data = pred_data.merge(phase_pred_data, how='left')
    phase_pred_data['Episode_indices'] = phase_pred_data['Episode_indices'] + last_episode_index
    phase_pred_data['Phase_pred_filtered_aim_filtered_fixed_indices'] \
        = phase_pred_data['Phase_pred_filtered_aim_filtered_fixed_indices'] + last_phase_index
    final_data_list.append(phase_pred_data)
    last_episode_index += phase_pred_data['Episode_indices'].unique()[-1]
    last_phase_index += phase_pred_data['Phase_pred_filtered_aim_filtered_fixed_indices'].unique()[-1]


pred_data = pd.concat(final_data_list).reset_index(drop=True)
# pred_data = pred_data.sort_values(by='Frame').reset_index(drop=True)
pred_data = pred_data[['Frame', 'Episode'
                       , 'Matchphases_new', 'Aim', 'Aim_pred', 'Aim_filtered'
                       , 'Phase_pred_filtered_aim'
                       , 'Phase_pred_filtered_aim_filtered'
                       , 'Phase_pred_filtered_aim_filtered_fixed']]
pred_data_drop_episode = pred_data.dropna(subset=['Episode', 'Matchphases_new',
                                                  'Aim', 'Aim_pred', 'Aim_filtered',
                                                  'Phase_pred_filtered_aim',
                                                   'Phase_pred_filtered_aim_filtered',
                                                   'Phase_pred_filtered_aim_filtered_fixed'
                                                  ]).reset_index(drop=True)

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
f1_df.to_csv('Test_f1_per_match_rule-based.csv', index=False)

mean = f1_df.mean()
std = f1_df.std()

formatted = mean.round(2).astype(str) + ' ± ' + std.round(2).astype(str)

result_df = pd.DataFrame(formatted).T
result_df.index = ['Mean ± Std']

print("\nF1 results：")
print(result_df)

