import sys
from pathlib import Path

import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
TOOLS_DIR = REPO_ROOT / "phase_model_pkg" / "tools"
for path in (REPO_ROOT, SCRIPT_DIR, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase_model_pkg.tools.trainer import predict_model
from phase_model_pkg.tools.label_utils import generate_indices, aim_transform
from phase_model_pkg.tools.datasets import SoccerHeteroDataset
from phase_model_pkg.tools.Filter import AimFilter, PhaseFilter
from phase_model_pkg.tools.Fixer import PhaseFixer
from phase_model_pkg.T_GAN.T_GAN_model import SoccerGNNTransformer


# -------------------- Change these paths for a new match --------------------
match_name = "DFL-MAT-003AXI_Borussia Dortmund-FC Augsburg_2018-2019"
match_team = "BVB_FCA"

data_root = SCRIPT_DIR.parent / "Demo_Datasets"
hetero_path = data_root / match_name / f"Hetero_demo_{match_name}.pt"
feature_path = data_root / match_name / f"Feature_demo_{match_name}.csv"
global_path = data_root / match_name / f"Global_demo_{match_name}.csv"

aim_model_path = SCRIPT_DIR / f"aim_TGAN_demo_{match_team}.pt"
invade_model_path = SCRIPT_DIR / f"Invade_TGAN_demo_{match_team}.pt"
scoring_model_path = SCRIPT_DIR / f"Scoring_TGAN_demo_{match_team}.pt"

output_path = SCRIPT_DIR / f"Matchphases_{match_team}_Phase_pred_T-GAN_demo.csv"
# ----------------------------------------------------------------------------


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

match_hetero_data = torch.load(hetero_path, map_location=device, weights_only=False)

aim_model = SoccerGNNTransformer(
    node_dim=11,
    edge_dim=3,
    global_dim=18,
    action_dim=18,
    num_classes=3,
    d_model=128,
    nhead=2,
    dim_feedforward=256,
    dropout=0.1,
    gnn_layers=1,
    trans_layers=1,
).to(device)
aim_model.load_state_dict(torch.load(aim_model_path, map_location=device, weights_only=False))

invade_model = SoccerGNNTransformer(
    node_dim=11,
    edge_dim=3,
    global_dim=18,
    action_dim=17,
    num_classes=3,
    d_model=128,
    nhead=2,
    dim_feedforward=256,
    dropout=0.1,
    gnn_layers=1,
    trans_layers=1,
).to(device)
invade_model.load_state_dict(torch.load(invade_model_path, map_location=device, weights_only=False))

scoring_model = SoccerGNNTransformer(
    node_dim=11,
    edge_dim=3,
    global_dim=18,
    action_dim=17,
    num_classes=2,
    d_model=128,
    nhead=2,
    dim_feedforward=256,
    dropout=0.1,
    gnn_layers=1,
    trans_layers=1,
).to(device)
scoring_model.load_state_dict(torch.load(scoring_model_path, map_location=device, weights_only=False))

match_original_data = pd.read_csv(feature_path)
match_original_data["Frame_index"] = match_original_data.index.values

global_data = pd.read_csv(global_path)
required_feature_columns = ["Frame", "Frame_index", "Episode", "Action"]
missing_feature_columns = [col for col in required_feature_columns if col not in match_original_data.columns]
if missing_feature_columns:
    raise ValueError(f"Missing required feature columns: {missing_feature_columns}")

match_original_data = match_original_data[required_feature_columns]
match_data = match_original_data.dropna(subset=["Episode"]).reset_index(drop=True)

global_data = global_data.dropna(subset="Episode").reset_index(drop=True)
global_data = global_data[global_data["Frame_index"].isin(match_data["Frame_index"])].reset_index(drop=True)
global_data = global_data[["Frame_index", "Episode_indices"]]
global_data = global_data.merge(match_data, how="left")

match_hetero_datalist = match_hetero_data.to_data_list()
match_hetero_dataset = SoccerHeteroDataset(
    match_hetero_datalist,
    cat="Episode",
    shuffle=False,
    down_sampler=True,
    down_round=5,
    max_length=100,
)

aim_pred, aim_event_importance_df = predict_model(
    aim_model,
    match_hetero_dataset,
    device=device,
    y_key=0,
    traditional_features=True,
    return_event_importance=True,
)

pred_data = global_data.dropna(subset=["Episode"]).reset_index(drop=True)
pred_data["Aim_pred"] = aim_pred
pred_data["Aim_pred"] = pred_data["Aim_pred"].replace(
    [0, 1, 2],
    ["Invade opponent space", "Keep possession", "Scoring"],
)
pred_data["pred_index"] = pred_data.index

aim_pred_data = pred_data[["Frame_index", "Episode", "Episode_indices", "Aim_pred"]]
aim_pred_data = AimFilter(aim_pred_data, gnn=True).filter(50, 50, 25)
pred_data["Aim_filtered"] = aim_pred_data["Aim_filtered"]

pred_data = generate_indices(pred_data, "Episode", "Aim_filtered")
pred_data["Aim_filtered"] = pred_data["Aim_filtered"].apply(aim_transform)

pred_data["global_event_id"] = pred_data.index
pred_data = pred_data.merge(aim_event_importance_df, how="left").reset_index(drop=True)

match_hetero_data["global"].y = torch.tensor(
    pd.DataFrame({
        "Aim_filtered": pred_data["Aim_filtered"],
        "phase_placeholder": 0,
    }).values,
    dtype=torch.long,
).to(device).reshape(-1)
match_hetero_data["Aim_indices"].id = torch.tensor(
    pred_data["Aim_filtered_indices"],
    dtype=torch.int,
).to(device)

match_hetero_datalist = match_hetero_data.to_data_list()
match_hetero_dataset_invade = SoccerHeteroDataset(
    match_hetero_datalist,
    cat="Aim",
    aim_name="Invade",
    from_simi=False,
    shuffle=False,
    down_sampler=True,
    down_round=5,
    max_length=100,
)

Invade_pred, invade_event_importance_df = predict_model(
    invade_model,
    match_hetero_dataset_invade,
    device=device,
    y_key=1,
    traditional_features=True,
    counter_features=True,
    return_event_importance=True,
)

Invade_pred_data = pred_data[pred_data["Aim_filtered"] == 0].reset_index(drop=True)
Invade_pred_data["Phase_pred_filtered_aim"] = Invade_pred
Invade_pred_data["Phase_pred_filtered_aim"] = Invade_pred_data["Phase_pred_filtered_aim"].replace(
    [0, 1, 2],
    ["Build Up", "Progression", "Counter attack"],
)
Invade_pred_data["global_event_id"] = Invade_pred_data.index
Invade_pred_data = Invade_pred_data.merge(invade_event_importance_df, how="left").reset_index(drop=True)

match_hetero_dataset_scoring = SoccerHeteroDataset(
    match_hetero_datalist,
    cat="Aim",
    aim_name="Scoring",
    from_simi=False,
    shuffle=False,
    down_sampler=True,
    down_round=6,
    max_length=20,
)

Scoring_pred, scoring_event_importance_df = predict_model(
    scoring_model,
    match_hetero_dataset_scoring,
    device=device,
    y_key=1,
    traditional_features=True,
    return_event_importance=True,
)

Scoring_pred_data = pred_data[pred_data["Aim_filtered"] == 2].reset_index(drop=True)
Scoring_pred_data["Phase_pred_filtered_aim"] = Scoring_pred
Scoring_pred_data["Phase_pred_filtered_aim"] = Scoring_pred_data["Phase_pred_filtered_aim"].replace(
    [0, 1],
    ["Sustained Threat", "Finishing"],
)
Scoring_pred_data["global_event_id"] = Scoring_pred_data.index
Scoring_pred_data = Scoring_pred_data.merge(scoring_event_importance_df, how="left").reset_index(drop=True)

Keep_pred_data = pred_data[pred_data["Aim_filtered"] == 1].reset_index(drop=True)
Keep_pred_data["Phase_pred_filtered_aim"] = "Maintenance"

phase_pred_data = pd.concat([Invade_pred_data, Keep_pred_data, Scoring_pred_data]).sort_values(
    "Frame_index"
).reset_index(drop=True)

phase_pred_data = PhaseFilter(phase_pred_data).filter(50, 25)
phase_pred_data["Aim_filtered"] = phase_pred_data["Aim_filtered"].replace(
    [0, 1, 2],
    ["Invade opponent space", "Keep possession", "Scoring"],
)

phase_pred_data = PhaseFixer(phase_pred_data).fix()
phase_pred_data = phase_pred_data.sort_values(by="Frame_index").reset_index(drop=True)
phase_pred_data = generate_indices(
    phase_pred_data,
    "Aim_filtered",
    "Phase_pred_filtered_aim_filtered_fixed",
)

final_data = global_data.merge(phase_pred_data, how="left").reset_index(drop=True)
output_columns = [
    "Frame",
    "Frame_index",
    "Episode",
    "Episode_indices",
    "Aim_pred",
    "Aim_filtered",
    "Aim_filtered_indices",
    "Phase_pred_filtered_aim",
    "Phase_pred_filtered_aim_filtered",
    "Phase_pred_filtered_aim_filtered_fixed",
    "Phase_pred_filtered_aim_filtered_fixed_indices",
    "event_importance_intention",
    "event_importance_phase",
]
final_data = final_data[[col for col in output_columns if col in final_data.columns]]
final_data.to_csv(output_path, index=False)
print(f"Saved predictions to {output_path}")
