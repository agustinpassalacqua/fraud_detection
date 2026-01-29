# src/make_plots.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import load
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# -------- config (igual a tu logica) --------
COST_FP = 1
RECALL_AT_K_LIST = [100, 500, 1000]


# -------- helpers --------
def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError("El modelo no soporta predict_proba ni decision_function.")


def recall_at_k(y_true, y_score, k):
    y_true_np = np.asarray(y_true)
    idx = np.argsort(-y_score)[:k]
    total_pos = y_true_np.sum()
    if total_pos == 0:
        return 0.0
    return float(y_true_np[idx].sum() / total_pos)


def precision_at_k(y_true, y_score, k):
    y_true_np = np.asarray(y_true)
    idx = np.argsort(-y_score)[:k]
    return float(y_true_np[idx].mean())


def amount_recovered_at_k(y_true, y_score, amount, k):
    y_true_np = np.asarray(y_true)
    amt_np = np.asarray(amount)
    idx = np.argsort(-y_score)[:k]
    return float(amt_np[idx][y_true_np[idx] == 1].sum())


def cost_at_threshold(y_true, y_score, amount, thr, cost_fp=COST_FP):
    y_true_np = np.asarray(y_true)
    amt = np.asarray(amount)

    y_pred = (y_score >= thr).astype(int)
    fp_mask = (y_pred == 1) & (y_true_np == 0)
    fn_mask = (y_pred == 0) & (y_true_np == 1)

    fp = int(fp_mask.sum())
    fn = int(fn_mask.sum())

    fn_amount_sum = float(amt[fn_mask].sum())
    cost = fp * cost_fp + fn_amount_sum
    return float(cost), fp, fn, float(fn_amount_sum)


def find_best_threshold_by_cost(y_true, y_score, amount):
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    candidate_thresholds = np.unique(np.concatenate([thresholds, np.array([0.0, 1.0])]))

    best_thr = 0.5
    best_cost = float("inf")
    best_fp = best_fn = 0
    best_fn_amount = 0.0

    for thr in candidate_thresholds:
        cost, fp, fn, fn_amount_sum = cost_at_threshold(y_true, y_score, amount, float(thr))
        if cost < best_cost:
            best_cost = cost
            best_thr = float(thr)
            best_fp = fp
            best_fn = fn
            best_fn_amount = fn_amount_sum

    return {"threshold": best_thr, "cost": best_cost, "fp": best_fp, "fn": best_fn, "fn_amount_sum": best_fn_amount, "candidate_thresholds": candidate_thresholds}


def save_roc(y_true, y_score, out_path, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_pr(y_true, y_score, out_path, title):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure()
    plt.plot(rec, prec, label=f"AP(PR-AUC) = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_cm(y_true, y_score, thr, out_path, title):
    y_pred = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Fraud", "Fraud"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d")
    plt.title(title + f" (thr={thr:.4f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_cost_curve(y_true, y_score, amount, chosen_thr, out_path, title):
    diag = find_best_threshold_by_cost(y_true, y_score, amount)
    thresholds = diag["candidate_thresholds"]

    costs = []
    for thr in thresholds:
        c, _, _, _ = cost_at_threshold(y_true, y_score, amount, float(thr))
        costs.append(c)
    costs = np.array(costs)

    best_idx = int(np.argmin(costs))
    best_thr = float(thresholds[best_idx])
    best_cost = float(costs[best_idx])

    plt.figure()
    plt.plot(thresholds, costs)
    plt.axvline(chosen_thr, linestyle="--", label=f"chosen thr={chosen_thr:.4f}")
    plt.axvline(best_thr, linestyle=":", label=f"best thr={best_thr:.4f}")
    plt.title(title + f" | best_cost={best_cost:.2f}")
    plt.xlabel("threshold")
    plt.ylabel("cost = FP*1 + sum(Amount of FN)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return {"best_thr": best_thr, "best_cost": best_cost}


def write_topk_report(y_true, y_score, amount, out_path, header):
    lines = []
    lines.append(header)
    total_fraud_amt = float(np.asarray(amount)[np.asarray(y_true) == 1].sum())
    lines.append(f"Total fraud amount in split: {total_fraud_amt:.2f}")
    for k in RECALL_AT_K_LIST:
        r = recall_at_k(y_true, y_score, k)
        p = precision_at_k(y_true, y_score, k)
        amt_rec = amount_recovered_at_k(y_true, y_score, amount, k)
        lines.append(f"K={k:4d} | recall@k={r:.4f} | precision@k={p:.4f} | fraud_amount_captured={amt_rec:.2f}")
    lines.append("")

    with open(out_path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    data_dir = project_root / "data" / "splits"

    report_path = models_dir / "report.json"
    model_cost_path = models_dir / "best_model.joblib"
    model_rank_path = models_dir / "best_model_rank.joblib"

    out_dir = project_root / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load report
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    best_cost_name = report["best_model_cost"]["name"]
    best_rank_name = report["best_model_rank"]["name"]
    thr_cost = float(report["best_model_cost"]["threshold_from_val"])
    thr_rank = float(report["best_model_rank"]["threshold_from_val"])

    # Load splits
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    for df in [val_df, test_df]:
        df["Class"] = df["Class"].astype(int)

    X_val = val_df.drop(columns=["Class"])
    y_val = val_df["Class"]
    amt_val = val_df["Amount"]

    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"]
    amt_test = test_df["Amount"]

    # Load models
    model_cost = load(model_cost_path)
    model_rank = load(model_rank_path)

    # Scores
    val_score_cost = get_scores(model_cost, X_val)
    test_score_cost = get_scores(model_cost, X_test)

    val_score_rank = get_scores(model_rank, X_val)
    test_score_rank = get_scores(model_rank, X_test)

    # -------- plots (ROC/PR) --------
    save_roc(y_val, val_score_cost, out_dir / "roc_val_best_cost.png", f"ROC - VAL - winner(min cost): {best_cost_name}")
    save_roc(y_test, test_score_cost, out_dir / "roc_test_best_cost.png", f"ROC - TEST - winner(min cost): {best_cost_name}")
    save_roc(y_val, val_score_rank, out_dir / "roc_val_best_rank.png", f"ROC - VAL - winner(best rank): {best_rank_name}")
    save_roc(y_test, test_score_rank, out_dir / "roc_test_best_rank.png", f"ROC - TEST - winner(best rank): {best_rank_name}")

    save_pr(y_val, val_score_cost, out_dir / "pr_val_best_cost.png", f"PR - VAL - winner(min cost): {best_cost_name}")
    save_pr(y_test, test_score_cost, out_dir / "pr_test_best_cost.png", f"PR - TEST - winner(min cost): {best_cost_name}")
    save_pr(y_val, val_score_rank, out_dir / "pr_val_best_rank.png", f"PR - VAL - winner(best rank): {best_rank_name}")
    save_pr(y_test, test_score_rank, out_dir / "pr_test_best_rank.png", f"PR - TEST - winner(best rank): {best_rank_name}")

    # -------- confusion matrices on TEST (threshold chosen on VAL) --------
    save_cm(y_test, test_score_cost, thr_cost, out_dir / "cm_test_best_cost.png", f"Confusion Matrix - TEST - winner(min cost): {best_cost_name}")
    save_cm(y_test, test_score_rank, thr_rank, out_dir / "cm_test_best_rank.png", f"Confusion Matrix - TEST - winner(best rank): {best_rank_name}")

    # -------- cost curves (VAL and TEST) --------
    save_cost_curve(y_val, val_score_cost, amt_val, thr_cost, out_dir / "costcurve_val_best_cost.png", f"Cost curve - VAL - {best_cost_name}")
    save_cost_curve(y_test, test_score_cost, amt_test, thr_cost, out_dir / "costcurve_test_best_cost.png", f"Cost curve - TEST - {best_cost_name}")

    save_cost_curve(y_val, val_score_rank, amt_val, thr_rank, out_dir / "costcurve_val_best_rank.png", f"Cost curve - VAL - {best_rank_name}")
    save_cost_curve(y_test, test_score_rank, amt_test, thr_rank, out_dir / "costcurve_test_best_rank.png", f"Cost curve - TEST - {best_rank_name}")

    # -------- top-k report (append) --------
    topk_path = project_root / "reports" / "topk_report.txt"
    topk_path.parent.mkdir(parents=True, exist_ok=True)

    # overwrite previous report for cleanliness
    with open(topk_path, "w", encoding="utf-8") as f:
        f.write("")

    write_topk_report(y_test, test_score_cost, amt_test, topk_path, f"Top-K - TEST - winner(min cost): {best_cost_name}")
    write_topk_report(y_test, test_score_rank, amt_test, topk_path, f"Top-K - TEST - winner(best rank): {best_rank_name}")

    print("Listo. Figuras en:", out_dir)
    print("Reporte top-k en:", topk_path)


if __name__ == "__main__":
    main()
