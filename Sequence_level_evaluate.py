from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


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
    "Scoring",
]

REQUIRED_COLUMNS = [
    "Episode",
    "Matchphases_new",
    "Aim",
    "Aim_pred",
    "Aim_filtered",
    "Aim_filtered_indices",
    "Phase_pred_filtered_aim",
    "Phase_pred_filtered_aim_filtered",
    "Phase_pred_filtered_aim_filtered_fixed",
    "Phase_pred_filtered_aim_filtered_fixed_indices",
]

KEEP_COLUMNS = [
    "Frame",
    "Frame_index",
    *REQUIRED_COLUMNS,
]

MATCH_KEY_COL = "__match_key__"
ROW_ORDER_COL = "__row_order__"
ORIGINAL_FRAME_PREFIX = "OriginalFrameF1__"


@dataclass(frozen=True)
class MatchInfo:
    short_name: str
    team_key: str
    mat_id: str


MATCHES = [
    MatchInfo("BVB_FCA", "BVB_FCA", "DFL-MAT-003AXI"),
    MatchInfo("S04_DUS", "S04_D*S", "DFL-MAT-003AXO"),
    MatchInfo("SVW_WOB", "SVW_WOB", "DFL-MAT-003AXK"),
    MatchInfo("FCB_SCP", "FCB_SCP", "DFL-MAT-003BL4"),
    MatchInfo("HOF_BSC", "HOF_BSC", "DFL-MAT-0027A5"),
    MatchInfo("SVD_BSC", "SVD_BSC", "DFL-MAT-0027G3"),
    MatchInfo("M05_FCI", "M05_FCI", "DFL-MAT-00279Z"),
]


@dataclass(frozen=True)
class ModelConfig:
    output_name: str
    folder_name: str
    prediction_path: Callable[[Path, MatchInfo], Path]


@dataclass(frozen=True)
class EvalSpec:
    name: str
    gt_col: str
    pred_col: str
    order: list[str]
    table_prefix: str


def find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        preview = "\n".join(str(path) for path in matches[:20])
        raise FileNotFoundError(
            f"Expected exactly one file for {directory / pattern}, found {len(matches)}.\n{preview}"
        )
    return matches[0]


def find_first(root: Path, candidates: list[tuple[str, str]]) -> Path:
    errors = []
    for directory_name, pattern in candidates:
        directory = root / directory_name
        matches = sorted(directory.glob(pattern))
        if len(matches) == 1:
            return matches[0]
        errors.append(f"{directory / pattern}: found {len(matches)}")
    raise FileNotFoundError("No unique prediction file found. Tried:\n" + "\n".join(errors))


MODELS = [
    ModelConfig(
        output_name="T-GNN",
        folder_name="T-GNN",
        prediction_path=lambda root, match: find_first(root, [
            ("T-GAN", f"{match.mat_id}*_phase_with_gnn_features.csv"),
            ("T_GAN", f"Matchphases_{match.team_key}_Phase_pred_T-GAN_demo.csv"),
        ]),
    ),
    ModelConfig(
        output_name="Pure Transformer",
        folder_name="Pure_Transformer",
        prediction_path=lambda root, match: find_first(root, [
            ("Transformer_only", f"Matchphases_{match.team_key}_Phase_pred_leave_out_7_pure_transformer.csv"),
            ("Pure_Transformer", f"Matchphases_{match.team_key}_Phase_pred_pure_transformer_demo.csv"),
        ]),
    ),
    ModelConfig(
        output_name="Pure_GNN",
        folder_name="Pure_GNN",
        prediction_path=lambda root, match: find_first(root, [
            ("GNN_only", f"Matchphases_{match.team_key}_Phase_pred_leave_out_7_pure_GNN.csv"),
            ("Pure_GNN", f"Matchphases_{match.team_key}_Phase_pred_pure_GNN_demo.csv"),
        ]),
    ),
    ModelConfig(
        output_name="Random Forest",
        folder_name="Random_Forest",
        prediction_path=lambda root, match: find_first(root, [
            ("Classic_ML_frame_level", f"Matchphases_{match.mat_id}*_Phase_pred_leave_out_7_light_GBM.csv"),
            ("Random_Forest", f"Matchphases_{match.mat_id}*_Phase_pred_random_forest_demo.csv"),
        ]),
    ),
    ModelConfig(
        output_name="Rule-based",
        folder_name="Rule_based",
        prediction_path=lambda root, match: find_first(root, [
            ("Rule-based", f"Matchphases_{match.team_key}_Phase_pred_rule_based.csv"),
            ("Rule_based", f"Matchphases_{match.team_key}_Phase_pred_rule_based_demo.csv"),
        ]),
    ),
]

EVAL_SPECS = [
    EvalSpec("Intention_Unfiltered", "Aim", "Aim_pred", AIM_ORDER, "IntentionMat__Unfiltered__"),
    EvalSpec("Intention_Processed", "Aim", "Aim_filtered", AIM_ORDER, "IntentionMat__Processed__"),
    EvalSpec("Phase_Unfiltered", "Matchphases_new", "Phase_pred_filtered_aim", PHASE_ORDER, "PhaseMat__Unfiltered__"),
    EvalSpec("Phase_Filtered", "Matchphases_new", "Phase_pred_filtered_aim_filtered", PHASE_ORDER, "PhaseMat__Filtered__"),
    EvalSpec(
        "Phase_Processed",
        "Matchphases_new",
        "Phase_pred_filtered_aim_filtered_fixed",
        PHASE_ORDER,
        "PhaseMat__Processed__",
    ),
]

ORIGINAL_FRAME_F1_FILES = {
    "T-GNN": Path("Test_f1_per_match_T-GNN.csv"),
    "Pure Transformer": Path("Baseline_models/Transformer_only/Test_f1_per_match_pure_transformer.csv"),
    "Pure_GNN": Path("Baseline_models/GNN_only/Test_f1_per_match_pure_GNN.csv"),
    "Random Forest": Path("Baseline_models/Classic_ML_frame_level/Test_f1_per_match_random_forest.csv"),
    "Rule-based": Path("Baseline_models/Rule-based/Test_f1_per_match_rule-based.csv"),
}


def read_prediction(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep=";", low_memory=False)
    if data.shape[1] == 1:
        data = pd.read_csv(path, sep=None, engine="python")
    keep = [col for col in KEEP_COLUMNS if col in data.columns]
    return data.loc[:, keep].copy()


def clean_label_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned.str.lower().isin(["", "nan", "none"]))


def reorder_to_fixed_labels(matrix: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    matrix = matrix.copy()
    for label in order:
        if label not in matrix.index:
            matrix.loc[label] = 0.0
        if label not in matrix.columns:
            matrix[label] = 0.0
    extras = sorted(set([x for x in matrix.index if x not in order] + [x for x in matrix.columns if x not in order]))
    final_order = order + extras
    return matrix.reindex(index=final_order, columns=final_order, fill_value=0.0)


def harmonic_f1(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    a, b = a.align(b, join="outer", axis=None, fill_value=0.0)
    denom = a + b
    return ((2.0 * a * b) / denom.replace(0, np.nan)).fillna(0.0)


def get_tester_class():
    try:
        from phase_model_pkg.tools.Tester import Tester
    except ModuleNotFoundError:
        from tools.Tester import Tester
    return Tester


def original_iotd_f1_matrix(tester, gt_col: str, pred_col: str, order: list[str]) -> pd.DataFrame:
    iotd = reorder_to_fixed_labels(tester.iou_matrix(gt_col, pred_col), order)
    iotd_recall = reorder_to_fixed_labels(tester.iou_matrix(gt_col, pred_col, recall=True), order)
    return harmonic_f1(iotd, iotd_recall).round(2)


def build_iotd_matrices(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    data = data.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    Tester = get_tester_class()
    tester = Tester(data)
    return {
        spec.name: original_iotd_f1_matrix(tester, spec.gt_col, spec.pred_col, spec.order)
        for spec in EVAL_SPECS
    }


def extract_segments(labels: pd.Series) -> list[tuple[str, int, int, int]]:
    values = labels.astype(str).to_numpy()
    if len(values) == 0:
        return []
    segments = []
    start = 0
    label = values[0]
    for idx, current in enumerate(values[1:], start=1):
        if current != label:
            end = idx - 1
            segments.append((label, start, end, end - start + 1))
            start = idx
            label = current
    end = len(values) - 1
    segments.append((label, start, end, end - start + 1))
    return segments


def prepare_tiou_data(df: pd.DataFrame, gt_col: str, pred_col: str) -> pd.DataFrame:
    needed = [MATCH_KEY_COL, "Episode", gt_col, pred_col]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for tIoU: {missing}")

    eval_cols = [MATCH_KEY_COL, "Episode", gt_col, pred_col]
    if "Frame_index" in df.columns:
        eval_cols.append("Frame_index")
    elif "Frame" in df.columns:
        eval_cols.append("Frame")

    data = df.loc[:, list(dict.fromkeys(eval_cols))].copy()
    data[ROW_ORDER_COL] = np.arange(len(data))
    data[gt_col] = clean_label_series(data[gt_col])
    data[pred_col] = clean_label_series(data[pred_col])
    data = data.dropna(subset=needed)

    sort_cols = [MATCH_KEY_COL, "Episode"]
    if "Frame_index" in data.columns:
        sort_cols.append("Frame_index")
    elif "Frame" in data.columns:
        sort_cols.append("Frame")
    sort_cols.append(ROW_ORDER_COL)
    return data.sort_values(sort_cols).reset_index(drop=True)


def temporal_iou(
    ref_start: int,
    ref_end: int,
    ref_len: int,
    val_start: int,
    val_end: int,
    val_len: int,
) -> tuple[int, float]:
    overlap = max(0, min(ref_end, val_end) - max(ref_start, val_start) + 1)
    if overlap == 0:
        return 0, 0.0
    union = ref_len + val_len - overlap
    return overlap, overlap / union if union else 0.0


def directional_tiou_matrix(df: pd.DataFrame, ref_col: str, val_col: str, order: list[str]) -> pd.DataFrame:
    data = prepare_tiou_data(df, ref_col, val_col)
    sums = pd.DataFrame(0.0, index=order, columns=order)
    counts = pd.Series(0.0, index=order)

    for _, episode_df in data.groupby([MATCH_KEY_COL, "Episode"], sort=False):
        ref_segments = extract_segments(episode_df[ref_col])
        val_segments = extract_segments(episode_df[val_col])
        val_start = 0

        for ref_label, ref_start, ref_end, ref_len in ref_segments:
            if ref_label not in counts.index:
                continue
            counts.loc[ref_label] += 1
            best_by_label: dict[str, float] = {}

            while val_start < len(val_segments) and val_segments[val_start][2] < ref_start:
                val_start += 1

            cursor = val_start
            while cursor < len(val_segments) and val_segments[cursor][1] <= ref_end:
                val_label, val_seg_start, val_seg_end, val_len = val_segments[cursor]
                if val_label in sums.columns:
                    overlap, tiou = temporal_iou(ref_start, ref_end, ref_len, val_seg_start, val_seg_end, val_len)
                    if overlap > 0:
                        best_by_label[val_label] = max(best_by_label.get(val_label, 0.0), tiou)
                cursor += 1

            for val_label, score in best_by_label.items():
                sums.loc[ref_label, val_label] += score

    return sums.div(counts.replace(0, np.nan), axis=0).fillna(0.0)


def tiou_f1_matrix(df: pd.DataFrame, spec: EvalSpec) -> pd.DataFrame:
    precision_like = directional_tiou_matrix(df, spec.gt_col, spec.pred_col, spec.order)
    recall_like = directional_tiou_matrix(df, spec.pred_col, spec.gt_col, spec.order).T
    return reorder_to_fixed_labels(harmonic_f1(precision_like, recall_like), spec.order)


def build_tiou_matrices(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {spec.name: tiou_f1_matrix(data, spec) for spec in EVAL_SPECS}


def add_match_key(data: pd.DataFrame, match: MatchInfo) -> pd.DataFrame:
    out = data.copy()
    out[MATCH_KEY_COL] = match.short_name
    return out


def label_f1(y_true: pd.Series, y_pred: pd.Series, labels: list[str], average: str | None):
    cleaned = pd.DataFrame({"true": clean_label_series(y_true), "pred": clean_label_series(y_pred)}).dropna()
    if cleaned.empty:
        return [0.0 for _ in labels] if average is None else 0.0
    return f1_score(cleaned["true"], cleaned["pred"], labels=labels, average=average, zero_division=0)


def compute_frame_f1(data: pd.DataFrame) -> dict[str, float]:
    aim_scores = label_f1(data["Aim"], data["Aim_pred"], AIM_ORDER, average=None)
    aim_macro = label_f1(data["Aim"], data["Aim_pred"], AIM_ORDER, average="macro")

    invade_data = data[data["Aim"] == "Invade opponent space"]
    invade_labels = ["Build Up", "Progression", "Counter attack"]
    invade_scores = label_f1(invade_data["Matchphases_new"], invade_data["Phase_pred_filtered_aim"], invade_labels, None)
    invade_macro = label_f1(invade_data["Matchphases_new"], invade_data["Phase_pred_filtered_aim"], invade_labels, "macro")

    scoring_data = data[data["Aim"] == "Scoring"]
    scoring_labels = ["Sustained Threat", "Finishing"]
    scoring_scores = label_f1(scoring_data["Matchphases_new"], scoring_data["Phase_pred_filtered_aim"], scoring_labels, None)
    scoring_macro = label_f1(scoring_data["Matchphases_new"], scoring_data["Phase_pred_filtered_aim"], scoring_labels, "macro")

    return {
        "FrameF1__Intention_model_invade_test_f1": float(aim_scores[0]),
        "FrameF1__Intention_model_keep_test_f1": float(aim_scores[1]),
        "FrameF1__Intention_model_scoring_test_f1": float(aim_scores[2]),
        "FrameF1__Intention_model_test_f1": float(aim_macro),
        "FrameF1__Invade_model_Progression_test_f1": float(invade_scores[1]),
        "FrameF1__Invade_model_build_test_f1": float(invade_scores[0]),
        "FrameF1__Invade_model_counter_test_f1": float(invade_scores[2]),
        "FrameF1__Invade_model_test_f1": float(invade_macro),
        "FrameF1__Scoring_model_finishing_test_f1": float(scoring_scores[1]),
        "FrameF1__Scoring_model_sustained_test_f1": float(scoring_scores[0]),
        "FrameF1__Scoring_model_test_f1": float(scoring_macro),
    }


def build_original_frame_f1_table(metrics_root: Path) -> pd.DataFrame | None:
    frames = []
    for model_name, relative_path in ORIGINAL_FRAME_F1_FILES.items():
        path = metrics_root / relative_path
        if not path.exists():
            return None
        source = pd.read_csv(path)
        source.insert(0, "match_id", range(len(source)))
        source.insert(1, "base_model", model_name)
        metric_cols = [col for col in source.columns if col not in {"match_id", "base_model"}]
        source = source.rename(columns={col: f"{ORIGINAL_FRAME_PREFIX}{col}" for col in metric_cols})
        frames.append(source)
    return pd.concat(frames, ignore_index=True)


def diag_columns(matrices: dict[str, pd.DataFrame]) -> dict[str, float]:
    out: dict[str, float] = {}
    for spec in EVAL_SPECS:
        matrix = matrices[spec.name]
        values = []
        for label in spec.order:
            value = float(matrix.loc[label, label])
            out[f"{spec.table_prefix}diag__{label}"] = value
            values.append(value)
        out[f"{spec.table_prefix}diag_mean"] = float(np.mean(values)) if values else 0.0
    return out


def write_matrices(matrices: dict[str, pd.DataFrame], output_dir: Path, metric_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in matrices.items():
        suffix = f"{metric_name}_f1_matrix.csv"
        matrix.to_csv(output_dir / f"{name}_{suffix}", encoding="utf-8-sig")


def order_columns(table: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["match_id", "base_model"]
    original_cols = [col for col in table.columns if col.startswith(ORIGINAL_FRAME_PREFIX)]
    recomputed_frame_cols = [col for col in table.columns if col.startswith("FrameF1__")]
    matrix_cols = [col for col in table.columns if col.startswith("IntentionMat__") or col.startswith("PhaseMat__")]
    other_cols = [col for col in table.columns if col not in set(id_cols + original_cols + recomputed_frame_cols + matrix_cols)]
    return table[id_cols + original_cols + recomputed_frame_cols + other_cols + matrix_cols]


def attach_original_frame_f1(table: pd.DataFrame, original_f1: pd.DataFrame | None) -> pd.DataFrame:
    if original_f1 is None:
        return order_columns(table)
    merged = table.merge(original_f1, on=["match_id", "base_model"], how="left", validate="one_to_one")
    return order_columns(merged)


def evaluate(pred_root: Path, output_root: Path, variant: str, metrics_root: Path | None = None) -> None:
    iotd_output_root = output_root / f"{variant}_iotd_f1_matrices_by_model"
    tiou_output_root = output_root / f"{variant}_tiou_f1_matrices_by_model"
    iotd_table_path = output_root / f"model_performance_every_match_{variant}_iotd_f1_with_filtered.csv"
    tiou_table_path = output_root / f"model_performance_every_match_{variant}_tiou_f1_with_filtered.csv"

    iotd_rows = []
    tiou_rows = []

    for model in MODELS:
        print(f"Model: {model.output_name}", flush=True)
        raw_model_frames = []
        keyed_model_frames = []

        for match_id, match in enumerate(MATCHES):
            print(f"  Match {match_id}: {match.short_name}", flush=True)
            data = read_prediction(model.prediction_path(pred_root, match))
            keyed_data = add_match_key(data, match)
            raw_model_frames.append(data)
            keyed_model_frames.append(keyed_data)

            iotd_matrices = build_iotd_matrices(data)
            tiou_matrices = build_tiou_matrices(keyed_data)
            write_matrices(iotd_matrices, iotd_output_root / model.folder_name / "per_match" / match.short_name, "iotd")
            write_matrices(tiou_matrices, tiou_output_root / model.folder_name / "per_match" / match.short_name, "tiou")

            base_row = {"match_id": match_id, "base_model": model.output_name, **compute_frame_f1(keyed_data)}
            iotd_rows.append({**base_row, **diag_columns(iotd_matrices)})
            tiou_rows.append({**base_row, **diag_columns(tiou_matrices)})

        print(f"  Overall matrices: {model.output_name}", flush=True)
        write_matrices(
            build_iotd_matrices(pd.concat(raw_model_frames, ignore_index=True)),
            iotd_output_root / model.folder_name,
            "iotd",
        )
        write_matrices(
            build_tiou_matrices(pd.concat(keyed_model_frames, ignore_index=True)),
            tiou_output_root / model.folder_name,
            "tiou",
        )

    original_f1 = build_original_frame_f1_table(metrics_root) if metrics_root else None
    iotd_table = pd.DataFrame(iotd_rows).sort_values(["match_id", "base_model"]).reset_index(drop=True)
    tiou_table = pd.DataFrame(tiou_rows).sort_values(["match_id", "base_model"]).reset_index(drop=True)
    iotd_table = attach_original_frame_f1(iotd_table, original_f1)
    tiou_table = attach_original_frame_f1(tiou_table, original_f1)

    output_root.mkdir(parents=True, exist_ok=True)
    iotd_table.to_csv(iotd_table_path, index=False, encoding="utf-8-sig")
    tiou_table.to_csv(tiou_table_path, index=False, encoding="utf-8-sig")

    print(f"Wrote {iotd_table_path}")
    print(f"Wrote {tiou_table_path}")
    print(f"Wrote {iotd_output_root}")
    print(f"Wrote {tiou_output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute sequence-level IoT-D F1 and tIoU-F1 tables/matrices from postprocessed predictions."
    )
    parser.add_argument("--pred-root", required=True, type=Path, help="Root folder containing model prediction subfolders.")
    parser.add_argument("--output-root", type=Path, default=None, help="Folder where result tables and matrices are written.")
    parser.add_argument("--variant", default=None, help="Prefix used in output file/folder names.")
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=None,
        help="Folder containing original Test_f1_per_match*.csv files. If omitted, OriginalFrameF1 columns are skipped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_root = args.pred_root.resolve()
    output_root = args.output_root.resolve() if args.output_root else pred_root.parent.resolve()
    variant = args.variant or pred_root.name.replace("postprocessed_predictions_", "")
    metrics_root = args.metrics_root.resolve() if args.metrics_root else None
    evaluate(pred_root=pred_root, output_root=output_root, variant=variant, metrics_root=metrics_root)


if __name__ == "__main__":
    main()
