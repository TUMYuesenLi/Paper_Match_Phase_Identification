from phase_model_pkg.tools.trainer import *
from phase_model_pkg.tools.label_utils import cat_hetero, phase2aim, generate_indices, aim_transform, phase_transform
from phase_model_pkg.tools.datasets import SoccerHeteroDataset
from Pure_GNN_model import SoccerGNN
from phase_model_pkg.tools.losses import FocalLoss, BinaryFocalLoss
from Filter import AimFilter, PhaseFilter
from Fixer import PhaseFixer

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

match_list = [
    'DFL-MAT-003AXI_Borussia Dortmund-FC Augsburg_2018-2019',
    'DFL-MAT-003AXO_Fortuna Düsseldorf 1895 e.V.-FC Schalke 04_2018-2019',
    'DFL-MAT-003AXK_SV Werder Bremen-VfL Wolfsburg_2018-2019',
]
match_team_list = [
    'BVB_FCA',
    'S04_DÜS',
    'SVW_WOB',
    # , ''
]

BVB_FCA_data = torch.load(f'../Demo_Datasets/{match_list[0]}/Hetero_demo_{match_list[0]}.pt', map_location=device)
S04_DUS_data = torch.load(f'../Demo_Datasets/{match_list[1]}/Hetero_demo_{match_list[1]}.pt', map_location=device)
SVW_WOB_data = torch.load(f'../Demo_Datasets/{match_list[2]}/Hetero_demo_{match_list[2]}.pt', map_location=device)

data_list = [BVB_FCA_data
    , S04_DUS_data
    , SVW_WOB_data
             ]



last_episode_index = 0
last_aim_index = 0
last_phase_index = 0
pred_data_list = []

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

for i, test_data in enumerate(data_list):

    train_datas = data_list[:i] + data_list[i + 1:]
    train_data = cat_hetero(train_datas)

    train_data_list = train_data.to_data_list()
    test_data_list = test_data.to_data_list()


    """
    Train Aims
    """
    #
    train_dataset = SoccerHeteroDataset(train_data_list
                                        , cat='Episode'
                                        # , cat='Aim'
                                        # , aim_name='General'
                                        , from_simi=True
                                        , batch_size=16
                                        , shuffle=True
                                        , down_sampler=True
                                        , down_round=5
                                        , max_length=100
                                        )

    test_dataset = SoccerHeteroDataset(test_data_list
                                       , cat='Episode'
                                       # , cat='Aim'
                                       # , aim_name='General'
                                       , shuffle=False
                                       , from_simi=True
                                       , down_sampler=True
                                       , down_round=5
                                       , max_length=100
                                       )

    print('Data load complete')

    print('Start training')

    aim_model = SoccerGNN(
        node_dim=11,
        edge_dim=3,
        global_dim=28,
        num_classes=3,
        d_model=128,
        nhead=1,
        dim_feedforward=256,
        dropout=0.5,
        gnn_layers=1,
        trans_layers=1
        # device=device
    ).to(device)
    #
    optimizer = torch.optim.AdamW(aim_model.parameters(), lr=2e-4, weight_decay=5e-4)
    # alpha = torch.tensor([0.2, 0.6, 1.2], device=device)
    alpha = torch.tensor([0.2, 0.4, 0.6], device=device)
    criterion = FocalLoss(alpha=alpha, gamma=1, reduction='mean')
    num_epochs = 30
    early_stopping = EarlyStopping(patience=10, mode='max'
                                   , save_path=f'aim_pure_GNN_demo_{match_team_list[i]}.pt'
                                   )
    aim_f1_train, aim_f1_train_each = train_model(aim_model, train_dataset
                    , test_dataset
                    , criterion, optimizer, num_epochs=num_epochs, device=device
                    , early_stopping=early_stopping, y_key=0
                    )
    aim_model.load_state_dict(torch.load(
        f'aim_pure_GNN_demo_{match_team_list[i]}.pt'
    ))

    aim_pred_full, aim_full, aim_f1_test, aim_f1_test_each = test_model(aim_model, test_dataset, device=device
                                                                        , y_key=0
                                                                        )

    aim_f1_test_list.append(aim_f1_test)
    aim_f1_invade_test_list.append(aim_f1_test_each[0])
    aim_f1_keep_test_list.append(aim_f1_test_each[1])
    aim_f1_scoring_test_list.append(aim_f1_test_each[2])

    """
    Train Invade
    """
    train_dataset = SoccerHeteroDataset(train_data_list
                                        , cat='Aim'
                                   # , cat='Phase'
                                        , aim_name='Invade'
                                        , from_simi=True
                                        , batch_size=16
                                        , shuffle=True
                                        , down_sampler=True
                                        , down_round=5
                                        , max_length=100
                                        )


    test_dataset = SoccerHeteroDataset(test_data_list
                                       , cat='Aim'
                                       # , cat='Phase'
                                       , aim_name='Invade'
                                       , shuffle=False
                                       , from_simi=True
                                       , down_sampler=True
                                       , down_round=5
                                       , max_length=100
                                       )

    print('Data load complete')

    print('Start training')
    invade_model = SoccerGNN(
        node_dim=11,
        edge_dim=3,
        global_dim=28,
        num_classes=3,
        d_model=128,
        nhead=1,
        dim_feedforward=256,
        dropout=0.1,
        gnn_layers=1,
        trans_layers=1
        # device=device
    ).to(device)

    optimizer = torch.optim.AdamW(invade_model.parameters(), lr=2e-4, weight_decay=5e-4)
    criterion = FocalLoss(alpha=torch.tensor([0.1, 0.2, 0.4], device=device), gamma=1, reduction='mean')
    num_epochs = 40
    early_stopping = EarlyStopping(patience=20, mode='max'
                                   , save_path=f'Invade_pure_GNN_demo_{match_team_list[i]}.pt'
                                   , max_f1_gap=0.2)
    Invade_f1_train, Invade_f1_train_each = train_model(invade_model, train_dataset
                , test_dataset
                , criterion, optimizer, num_epochs=num_epochs, device=device
                , early_stopping=early_stopping, y_key=1
                )
    invade_model.load_state_dict(torch.load(
        f'Invade_pure_transformer_demo_{match_team_list[i]}.pt'
    ))

    Invade_pred_full, Invade_phases_full, Invade_f1_test, Invade_f1_test_each \
        = test_model(invade_model, test_dataset, device=device, y_key=1
                     )

    C1 = confusion_matrix(Invade_phases_full, Invade_pred_full)  # True_label 真实标签 shape=(n,1);T_predict1 预测标签 shape=(n,1)
    print(C1)

    invade_f1_test_list.append(Invade_f1_test)
    invade_build_f1_test_list.append(Invade_f1_test_each[0])
    invade_Progression_f1_test_list.append(Invade_f1_test_each[1])
    invade_counter_f1_test_list.append(Invade_f1_test_each[2])

    """
    Train Scoring
    """
    train_dataset = SoccerHeteroDataset(train_data_list
                                        , cat='Aim'
                                        # , cat='Phase'
                                        , aim_name='Scoring'
                                        , from_simi=True
                                        , batch_size=16
                                        , shuffle=True
                                        , down_sampler=True
                                        , down_round=6
                                        , max_length=20
                                        )

    test_dataset = SoccerHeteroDataset(test_data_list
                                       , cat='Aim'
                                       # , cat='Phase'
                                       , aim_name='Scoring'
                                       , shuffle=False
                                       , from_simi=True
                                       , down_sampler=True
                                       , down_round=6
                                       , max_length=20
                                       )
    print(np.mean([len(aim) for aim in test_dataset]))
    print('Data load complete')

    print('Start training')
    Scoring_model = SoccerGNN(
        node_dim=11,
        edge_dim=3,
        global_dim=28,
        num_classes=2,
        d_model=128,
        nhead=1,
        dim_feedforward=256,
        dropout=0.1,
        gnn_layers=1,
        trans_layers=1,
        # device=device
    ).to(device)
    optimizer = torch.optim.AdamW(Scoring_model.parameters(), lr=1e-4, weight_decay=5e-4)
    class_weights = torch.tensor([0.1, 1]).to(device)
    criterion = BinaryFocalLoss(alpha=0.8, gamma=2, reduction='mean', ignore_index=None)
    num_epochs = 40
    early_stopping = EarlyStopping(patience=20, mode='max', max_f1_gap=0.2
                                   # , save_path='Scoring_2_GNN+16Batchsize+attention+balance_game.pt'
                                   # , save_path='Scoring_GNN_counter_features_down.pt'
                                   , save_path=f'Scoring_pure_GNN_demo_{match_team_list[i]}.pt'
                                   )
    Scoring_f1_train, Scoring_f1_train_each = train_model(Scoring_model, train_dataset
                # , val_dataset
                , test_dataset
                , criterion, optimizer, num_epochs=num_epochs, device=device
                , early_stopping=early_stopping, y_key=1
                )
    Scoring_model.load_state_dict(torch.load(
        f'Scoring_pure_GNN_demo_{match_team_list[i]}.pt'
    ))
    pred_ratios_full, target_ratios_full, Scoring_f1_test, Scoring_f1_test_each \
        = test_model(Scoring_model, test_dataset, device=device, y_key=1
                     )

    scoring_f1_test_list.append(Scoring_f1_test)
    scoring_sustained_f1_test_list.append(Scoring_f1_test_each[0])
    scoring_finishing_f1_test_list.append(Scoring_f1_test_each[1])
    match_hetero_data = test_data

    match_original_data = pd.read_csv(f'../Demo_Datasets/{match_list[i]}/Feature_demo_{match_list[i]}.csv')
    match_original_data['Frame_index'] = match_original_data.index.values
    match_original_data = phase2aim(match_original_data)
    global_data = pd.read_csv(f'../Demo_Datasets/{match_list[i]}/Global_demo_{match_list[i]}.csv')
    # global_data = pd.read_csv(f'{feature_folder}/Random_episodes/Random_episodes_test.csv')
    match_original_data = match_original_data[
        ['Frame', 'Frame_index', 'Episode', 'Matchphases_new', 'Aim', 'Action']]
    match_data = match_original_data.dropna(subset=['Episode', 'Matchphases_new', 'Aim']).reset_index(drop=True)

    global_data = global_data.dropna(subset='Episode').reset_index(drop=True)

    global_data = global_data[global_data['Frame_index'].isin(match_data['Frame_index'])].reset_index(drop=True)
    global_data = global_data[['Frame_index', 'Episode_indices']]
    global_data = global_data.merge(match_data, how='left')

    match_hetero_datalist = match_hetero_data.to_data_list()
    match_hetero_dataset = SoccerHeteroDataset(match_hetero_datalist, cat='Episode'
                                               , shuffle=False
                                               , down_sampler=True
                                               , down_round=5
                                               , max_length=100
                                               )

    aim_pred, aim, _, _ = test_model(aim_model, match_hetero_dataset, device=device, y_key=0
                               )

    pred_data = global_data.dropna(subset=['Aim']).reset_index(drop=True)
    pred_data['Aim_pred'] = aim_pred
    pred_data['Aim_pred'] = pred_data['Aim_pred'].replace(
        [0, 1, 2],
        ['Invade opponent space', 'Keep possession', 'Scoring'])
    pred_data['pred_index'] = pred_data.index

    aim_pred_data = pred_data[['Frame_index',
                               'Episode',
                               'Episode_indices', 'Aim', 'Aim_pred']]
    aim_filter = AimFilter(aim_pred_data, gnn=True)
    aim_pred_data = aim_filter.filter(50, 50, 25)
    pred_data['Aim_filtered'] = aim_pred_data['Aim_filtered']
    f1 = f1_score(pred_data['Aim'], pred_data['Aim_filtered']
                  , average='macro'
                  )
    print(f"F1-score filtered: {f1}")
    C1 = confusion_matrix(pred_data['Aim'], pred_data['Aim_filtered'])
    print(C1)

    pred_data = generate_indices(pred_data, 'Episode', 'Aim_filtered')

    pred_data['Aim_filtered'] = pred_data['Aim_filtered'].apply(aim_transform)
    pred_data['Matchphases_new'] = pred_data['Matchphases_new'].apply(phase_transform)


    # match_hetero_data['player'].y = pred_data['Aim_filtered'].apply()
    y = match_hetero_data['global'].y
    new_y = torch.tensor(pred_data[['Aim_filtered', 'Matchphases_new']].values
                         , dtype=torch.long).to(device).view(-1)
    match_hetero_data['global'].y = torch.tensor(pred_data[['Aim_filtered', 'Matchphases_new']].values
                                                 , dtype=torch.long).to(device).view(-1)
    match_hetero_data['Aim_indices'].id = torch.tensor(pred_data['Aim_filtered_indices'], dtype=torch.int).to(
        device)

    match_hetero_datalist = match_hetero_data.to_data_list()
    match_hetero_dataset_invade = SoccerHeteroDataset(match_hetero_datalist, cat='Aim', aim_name='Invade',
                                                      from_simi=False
                                                      , shuffle=False
                                                      , down_sampler=True
                                                      , down_round=5
                                                      , max_length=100
                                                      )

    Invade_pred, Invade_phases, _, _ = test_model(invade_model, match_hetero_dataset_invade, device=device, y_key=1
                                            )

    Invade_pred_data = pred_data[pred_data['Aim_filtered'] == 0].reset_index(drop=True)
    Invade_pred_data['Phase_pred_filtered_aim'] = Invade_pred
    Invade_pred_data['Phase_pred_filtered_aim'] = Invade_pred_data['Phase_pred_filtered_aim'].replace(
        [0, 1, 2],
        ['Build Up', 'Progression', 'Counter attack'])
    # Invade_pred_data = Invade_pred_data[['Frame', 'Episode', 'Matchphases_new', 'Phase_pred_filtered_aim']]

    match_hetero_dataset_scoring = SoccerHeteroDataset(match_hetero_datalist, cat='Aim', aim_name='Scoring',
                                                       from_simi=False
                                                       , shuffle=False
                                                       , down_sampler=True
                                                       , down_round=6
                                                       , max_length=20
                                                       )
    Scoring_pred, Scoring_phases, _, _ = test_model(Scoring_model
                                                    , match_hetero_dataset_scoring
                                                    , device=device
                                                    , y_key=1)

    # phase_pred_data = pred_data[['Frame', 'Episode', 'Matchphases_new', 'Phase_pred']]
    Scoring_pred_data = pred_data[pred_data['Aim_filtered'] == 2].reset_index(drop=True)
    Scoring_pred_data['Phase_pred_filtered_aim'] = Scoring_pred
    Scoring_pred_data['Phase_pred_filtered_aim'] = Scoring_pred_data['Phase_pred_filtered_aim'].replace(
        [0, 1],
        ['Sustained Threat', 'Finishing'])
    # Scoring_pred_data = Scoring_pred_data[['Frame', 'Episode', 'Matchphases_new', 'Phase_pred_filtered_aim']]

    Keep_pred_data = pred_data[pred_data['Aim_filtered'] == 1].reset_index(drop=True)
    Keep_pred_data['Phase_pred_filtered_aim'] = 'Maintenance'
    phase_pred_data = pd.concat([Invade_pred_data, Keep_pred_data, Scoring_pred_data]).sort_values(
        'Frame_index').reset_index(drop=True)

    phase_pred_data['Matchphases_new'] = phase_pred_data['Matchphases_new'].replace(
        [0, 1, 2, 3, 4, 5],
        ['Build Up', 'Progression', 'Counter attack', 'Maintenance', 'Sustained Threat', 'Finishing'])

    f1 = f1_score(phase_pred_data['Matchphases_new'], phase_pred_data['Phase_pred_filtered_aim']
                  , average='macro'
                  )
    print(f"F1-score unfiltered: {f1}")
    C1 = confusion_matrix(phase_pred_data['Matchphases_new'], phase_pred_data['Phase_pred_filtered_aim'])
    print(C1)

    # pred_data = pred_data.merge(phase_pred_data, how='left').reset_index(drop=True)
    phase_filter = PhaseFilter(phase_pred_data
                               # , gnn=True
                               )
    phase_pred_data = phase_filter.filter(50, 25)

    # phase_pred_data['Phase_pred_filtered_aim_filtered'] = phase_pred_data['Phase_pred_filtered_aim_filtered']
    f1 = f1_score(phase_pred_data['Matchphases_new'], phase_pred_data['Phase_pred_filtered_aim_filtered']
                  , average='macro'
                  )
    print(f"F1-score filtered: {f1}")
    C1 = confusion_matrix(phase_pred_data['Matchphases_new'], phase_pred_data['Phase_pred_filtered_aim_filtered'])
    print(C1)

    phase_pred_data['Matchphases_new'] = phase_pred_data['Matchphases_new'].replace(
        [0, 1, 2, 3, 4, 5],
        ['Build Up', 'Progression', 'Counter attack', 'Maintenance', 'Sustained Threat', 'Finishing'])
    phase_pred_data['Aim_filtered'] = phase_pred_data['Aim_filtered'].replace(
        [0, 1, 2],
        ['Invade opponent space', 'Keep possession', 'Scoring'])

    phase_fixer = PhaseFixer(phase_pred_data)
    phase_pred_data = phase_fixer.fix()
    phase_pred_data = phase_pred_data.sort_values(by='Frame_index').reset_index(drop=True)
    phase_pred_data = generate_indices(phase_pred_data, 'Aim_filtered', 'Phase_pred_filtered_aim_filtered_fixed')
    # phase_pred_data.to_csv('Matchphases_Random_episodes_phase_pred_gnn.csv', index=False)

    final_data = global_data.merge(phase_pred_data, how='left').reset_index(drop=True)
    final_data = final_data[[
        'Frame',
        'Frame_index',
        'Episode',
        'Episode_indices',
        'Matchphases_new', 'Aim', 'Aim_pred', 'Aim_filtered', 'Aim_filtered_indices',
        'Phase_pred_filtered_aim', 'Phase_pred_filtered_aim_filtered',
        'Phase_pred_filtered_aim_filtered_fixed',
        'Phase_pred_filtered_aim_filtered_fixed_indices'
    ]]
    final_data.to_csv(f'Matchphases_{match_team_list[i]}_Phase_pred_pure_GNN_demo.csv')

    phase_pred_data['Episode_indices'] = phase_pred_data['Episode_indices'] + last_episode_index
    phase_pred_data['Aim_filtered_indices'] = phase_pred_data['Aim_filtered_indices'] + last_aim_index
    phase_pred_data['Phase_pred_filtered_aim_filtered_fixed_indices'] \
        = phase_pred_data['Phase_pred_filtered_aim_filtered_fixed_indices'] + last_phase_index
    pred_data_list.append(phase_pred_data)
    last_episode_index += phase_pred_data['Episode_indices'].unique()[-1]
    last_aim_index += phase_pred_data['Aim_filtered_indices'].unique()[-1]
    last_phase_index += phase_pred_data['Phase_pred_filtered_aim_filtered_fixed_indices'].unique()[-1]

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

f1_df.to_csv('Test_f1_per_match_Pure_GNN.csv', index=False)
mean = f1_df.mean()
std = f1_df.std()

formatted = mean.round(2).astype(str) + ' ± ' + std.round(2).astype(str)

result_df = pd.DataFrame(formatted).T
result_df.index = ['Mean ± Std']

print("\nF1 results：")
print(result_df)
