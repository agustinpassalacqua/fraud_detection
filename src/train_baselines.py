"""
train_baselines.py
==================

Objetivo:
- Entrenar varios modelos supervisados  para fraude en dataset desbalanceado.
- Hacer una búsqueda de hiperparámetros con PR-AUC.


Puntos conceptuales importantes en fraude:
- Accuracy no sirve con 0.172% de clase positiva: el modelo aprende a predecir la clase mayoritaria.
- Se usa AUPRC / Average Precision (PR-AUC) y métricas como Recall@K (lo deseable es que no haya FN -> alto recall).
- El umbral se elige según costos (FN cuesta mucho más que FP) o capacidad de revisión (top K).

Modelos:
- Regresion logistica (baseline fuerte en datos PCA + interpretabilidad)
- Linear SVM (como SGDClassifier en log-loss/hinge, eficiente en alta dimensión)
- Random Forest (baseline no lineal y puede requerir cuidado con desbalance)
- HistGradientBoostingClassifier

output:
- Guarda el mejor modelo (por costo o PR-AUC, según config) en models/best_model.joblib
- Guarda un reporte JSON con métricas de val y test + threshold elegido

Requisitos:
pip install pandas numpy scikit-learn matplotlib joblib
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


# -----------------------------
# config general
# -----------------------------

RANDOM_STATE = 2026

# En fraude, normalmente FN es más caro que FP (quiero evitar marcar como no fraude a casos de fraude)
# ponderamos el costo: si se clasifica mas un fraude (FN) cuesta 20x más que bloquear/alertar por error (FP).
COST_FP = 1
COST_FN = 20

# Métricas operativas: "si puedo revisar K alertas, cuanto fraude recupero?"
RECALL_AT_K_LIST = [100, 500, 1000]

# Cross validation estratificada para mantener ratio de fraude en folds
CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# Cantidad de iteraciones de RandomizedSearch por modelo, a criterio del usuario
N_ITER = 25


# -----------------------------
# Utilidades de metricas
# -----------------------------

def recall_at_k(y_true: pd.Series, y_score: np.ndarray, k: int) -> float:
    """
    Recall@K:
    - Ordenamos por score descendente
    - Tomamos top K como casos a revisar
    - Calculamos qué proporción del total de fraudes cae en esos top K.
    """
    y_true_np = y_true.to_numpy()
    idx = np.argsort(-y_score)[:k]
    total_pos = y_true_np.sum()
    if total_pos == 0:
        return 0.0
    return float(y_true_np[idx].sum() / total_pos)


def cost_at_threshold(y_true: pd.Series,
                      y_score: np.ndarray,
                      amount: pd.Series,
                      thr: float,
                      cost_fp: float = COST_FP) -> Tuple[float, int, int, float]:
    """
    Costo operativo:
      - FP: costo fijo por alerta falsa
      - FN: costo monetario proporcional al Amount (fraude no detectado)
            -> sum(Amount) de los falsos negativos

    Devuelve:
      (costo_total, fp_count, fn_count, fn_amount_sum)
    """
    y_true_np = y_true.to_numpy()
    amt = amount.to_numpy()

    y_pred = (y_score >= thr).astype(int)

    fp_mask = (y_pred == 1) & (y_true_np == 0)
    fn_mask = (y_pred == 0) & (y_true_np == 1)

    fp = int(fp_mask.sum())
    fn = int(fn_mask.sum())

    fn_amount_sum = float(amt[fn_mask].sum())
    cost = fp * cost_fp + fn_amount_sum
    return float(cost), fp, fn, fn_amount_sum


def find_best_threshold_by_cost(y_true: pd.Series,
                                y_score: np.ndarray,
                                amount: pd.Series,
                                cost_fp: float = COST_FP) -> Dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    candidate_thresholds = np.unique(np.concatenate([thresholds, np.array([0.0, 1.0])]))

    best = {"threshold": 0.5, "cost": float("inf"), "fp": None, "fn": None, "fn_amount_sum": None}
    for thr in candidate_thresholds:
        cost, fp, fn, fn_amount_sum = cost_at_threshold(y_true, y_score, amount, float(thr), cost_fp)
        if cost < best["cost"]:
            best = {
                "threshold": float(thr),
                "cost": float(cost),
                "fp": fp,
                "fn": fn,
                "fn_amount_sum": float(fn_amount_sum),
            }
    return best


def evaluate_scores(y_true: pd.Series, y_score: np.ndarray, amount: pd.Series, threshold: float) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))

    for k in RECALL_AT_K_LIST:
        metrics[f"recall@{k}"] = recall_at_k(y_true, y_score, k)

    cost, fp, fn, fn_amount_sum = cost_at_threshold(y_true, y_score, amount, threshold, COST_FP)
    metrics["threshold"] = float(threshold)
    metrics["cost_fp_per_fp"] = float(COST_FP)
    metrics["cost"] = float(cost)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["fn_amount_sum"] = float(fn_amount_sum)
    return metrics


# -----------------------------
# Carga de datos
# -----------------------------

def load_split(project_root: Path, split_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carga un split guardado en data/splits/{split}.csv
    """
    path = project_root / "data" / "splits" / f"{split_name}.csv"
    df = pd.read_csv(path)
    df["Class"] = df["Class"].astype(int)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y

#Para ponderar por la perdida del fraude
def load_split_with_amount(project_root: Path, split_name: str):
    path = project_root / "data" / "splits" / f"{split_name}.csv"
    df = pd.read_csv(path)
    df["Class"] = df["Class"].astype(int)

    y = df["Class"]
    amount = df["Amount"]
    X = df.drop(columns=["Class"])  # Amount queda como feature

    return X, y, amount

# -----------------------------
# Construcción de pipelines + búsqueda de hiperparámetros
# -----------------------------

def build_preprocess(columns: List[str]) -> ColumnTransformer:
    """
    En este dataset:
    - V1..V28 son PCA osea que ya están en escala similar, pero no siempre perfecta.
    - Amount suele beneficiarse de escalado (y a veces Time también).
    Para ser consistente entre modelos lineales, escalamos Amount y Time.
    El resto pasa 'as is'.
    """
    scale_cols = [c for c in ["Amount", "Time"] if c in columns]
    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), scale_cols)],
        remainder="passthrough"
    )


@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline
    param_distributions: Dict[str, Any]


def get_model_specs(feature_columns: List[str]) -> List[ModelSpec]:
    """
    Definimos:
    - Pipeline(preprocess + modelo)
    - espacio de búsqueda de hiperparams


    """
    preprocess = build_preprocess(feature_columns)

    specs: List[ModelSpec] = []

    #) reg logistica
    #    - C: regularización
    #    - class_weight balanced para desbalance
    logreg_pipe = Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(
            max_iter=10000,
            class_weight="balanced",
            solver="lbfgs"  # estable
        ))
    ])
    logreg_params = {
        "clf__C": np.logspace(-3, 2, 30),
    }
    specs.append(ModelSpec("logreg", logreg_pipe, logreg_params))

    #  Linear SVMcon SGDClassifier:
    #    - eficiente en alta dimensión
    #    - class_weight balanced
    #    Conceptualmente esto cubre el espacio linear margin/linear classifier.
    sgd_pipe = Pipeline([
        ("prep", preprocess),
        ("clf", SGDClassifier(
            loss="log_loss",  # nos da probabilidades
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ))
    ])
    sgd_params = {
        "clf__alpha": np.logspace(-6, -2, 30),   # regularizcion
        "clf__penalty": ["l2", "l1", "elasticnet"],
        "clf__l1_ratio": np.linspace(0.05, 0.95, 10),  # solo aplica si elasticnet
    }
    specs.append(ModelSpec("linear_svm_like_sgd", sgd_pipe, sgd_params))

    #  Random Forest:
    #    - no lineal, robusto
    #    - usamos class_weight=balanced_subsample
    rf_pipe = Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    rf_params = {
        "clf__max_depth": [None, 4, 6, 8, 12, 16],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
    }
    specs.append(ModelSpec("random_forest", rf_pipe, rf_params))

    #  HistGradientBoosting:
    #    - suele rendir muy bien en tabular
    #    - no tiene class_weight, así que compensamos con learning rate,
    #      y controlando complejidad (una mejora futura sería sample_weight)
    hgb_pipe = Pipeline([
        ("prep", preprocess),
        ("clf", HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ))
    ])
    hgb_params = {
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "clf__max_depth": [3, 4, 6, 8, None],
        "clf__max_iter": [200, 400, 600],
        "clf__min_samples_leaf": [20, 50, 100, 200],
        "clf__l2_regularization": [0.0, 0.1, 0.5, 1.0],
    }
    specs.append(ModelSpec("hist_gradient_boosting", hgb_pipe, hgb_params))

    return specs


def run_search(spec: ModelSpec, X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    """
    RandomizedSearchCV:
    - Mejor que Grid para espacios grandes: explora más variedad con menos costo.
    - scoring: average_precision (PR-AUC) -> recomendado para desbalance extremo.
    - CV estratificada.
    """
    search = RandomizedSearchCV(
        estimator=spec.pipeline,
        param_distributions=spec.param_distributions,
        n_iter=N_ITER,
        scoring="average_precision",
        cv=CV,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,  # al final refitea con mejores params en train
    )
    search.fit(X_train, y_train)
    return search


# -----------------------------
# Main: entrenamiento + selección + evaluación
# -----------------------------

def get_scores(model, X: pd.DataFrame) -> np.ndarray:
    """
    Unifica obtención de scores (proba de clase positiva).
    - Si tiene predict_proba, usamos proba.si no, usamos decision_function y lo pasamos por sigmoid
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError("El modelo no soporta predict_proba ni decision_function.")


def main():
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Cargar splits fijos (misma partición para todos los modelos)
    X_train, y_train, amt_train  = load_split_with_amount(project_root, "train")
    X_val, y_val, amt_val     = load_split_with_amount(project_root, "val")
    X_test, y_test, amt_test   = load_split_with_amount(project_root, "test")

    specs = get_model_specs(X_train.columns.tolist())

    results: Dict[str, Any] = {"models": {}, "selection_rule": "min_val_cost + best_cv_pr_auc"}

    # criterio 1: modelo que minimiza costo en validación
    best_overall = {"name": None, "val_cost": float("inf"), "search": None, "threshold": None}

    # criterio 2: mejor ranking (mejor PR-AUC en CV) + threshold por costo (en validación)
    best_rank = {"name": None, "cv_pr_auc": float("-inf"), "search": None, "threshold": None, "val_cost": None}

    for spec in specs:
        print("\n" + "="*80)
        print(f"Modelo: {spec.name}")
        print("="*80)

        search = run_search(spec, X_train, y_train)
        best_model = search.best_estimator_

        # Score para
        # elegir umbral por costo
        y_val_score = get_scores(best_model, X_val)

        thr_info = find_best_threshold_by_cost(y_val, y_val_score, amt_val, COST_FP)
        threshold = thr_info["threshold"]
        val_metrics = evaluate_scores(y_val, y_val_score, amt_val, threshold)

        # Guardamos info del modelo
        results["models"][spec.name] = {
            "best_params": search.best_params_,
            "cv_best_pr_auc": float(search.best_score_),
            "val_metrics": val_metrics,
        }

        print(f"Best CV PR-AUC: {search.best_score_:.6f}")
        print(f"VAL PR-AUC:     {val_metrics['pr_auc']:.6f}")
        print(f"VAL cost:       {val_metrics['cost']:.2f} (FP={val_metrics['fp']}, FN={val_metrics['fn']})")
        print(f"Chosen thr:     {val_metrics['threshold']:.6f}")
        for k in RECALL_AT_K_LIST:
            print(f"VAL recall@{k}: {val_metrics[f'recall@{k}']:.6f}")

        # Regla de seleccion final
        # sel elije el modelo que minimiza el costo en validación, no el que maximiza la metrica (este fue mi criterio, se podria tomar otro approach)
        if val_metrics["cost"] < best_overall["val_cost"]:
            best_overall = {
                "name": spec.name,
                "val_cost": val_metrics["cost"],
                "search": search,
                "threshold": threshold,
            }

        # criterio 2: mejor ranking (PR-AUC en CV). el threshold igual lo elijo por costo en validación.
        if float(search.best_score_) > float(best_rank["cv_pr_auc"]):
            best_rank = {
                "name": spec.name,
                "cv_pr_auc": float(search.best_score_),
                "search": search,
                "threshold": threshold,
                "val_cost": val_metrics["cost"],
            }

    # -----------------------------
    # Evaluación FINAL FINAL (criterio 1: minimo costo en validación)
    # -----------------------------
    best_name = best_overall["name"]
    best_search = best_overall["search"]
    best_threshold = float(best_overall["threshold"])

    best_model = best_search.best_estimator_

    y_test_score = get_scores(best_model, X_test)
    test_metrics = evaluate_scores(y_test, y_test_score, amt_test, best_threshold)

    results["best_model_cost"] = {
        "name": best_name,
        "threshold_from_val": best_threshold,
        "test_metrics": test_metrics,
        "best_params": results["models"][best_name]["best_params"],
        "cv_best_pr_auc": results["models"][best_name]["cv_best_pr_auc"],
        "val_metrics": results["models"][best_name]["val_metrics"],
    }

    print("\n" + "#"*80)
    print("MEJOR MODELO (según costo en validación)")
    print("#"*80)
    print("Best:", best_name)
    print("Threshold (from VAL):", best_threshold)
    print("TEST PR-AUC:", f"{test_metrics['pr_auc']:.6f}")
    print("TEST cost:", f"{test_metrics['cost']:.2f}", f"(FP={test_metrics['fp']}, FN={test_metrics['fn']})")
    for k in RECALL_AT_K_LIST:
        print(f"TEST recall@{k}: {test_metrics[f'recall@{k}']:.6f}")

    # Guardar modelo (criterio 1)
    dump(best_model, models_dir / "best_model.joblib")

    # -----------------------------
    # Evaluación FINAL FINAL (criterio 2: mejor ranking en CV + threshold por costo)
    # -----------------------------
    best_rank_name = best_rank["name"]
    best_rank_search = best_rank["search"]
    best_rank_threshold = float(best_rank["threshold"])

    best_rank_model = best_rank_search.best_estimator_

    y_test_score_rank = get_scores(best_rank_model, X_test)
    test_metrics_rank = evaluate_scores(y_test, y_test_score_rank, amt_test, best_rank_threshold)

    results["best_model_rank"] = {
        "name": best_rank_name,
        "threshold_from_val": best_rank_threshold,
        "test_metrics": test_metrics_rank,
        "best_params": results["models"][best_rank_name]["best_params"],
        "cv_best_pr_auc": results["models"][best_rank_name]["cv_best_pr_auc"],
        "val_metrics": results["models"][best_rank_name]["val_metrics"],
    }

    print("\n" + "#"*80)
    print("MEJOR MODELO (según ranking: mejor PR-AUC en CV + threshold por costo en validación)")
    print("#"*80)
    print("Best:", best_rank_name)
    print("Threshold (from VAL):", best_rank_threshold)
    print("TEST PR-AUC:", f"{test_metrics_rank['pr_auc']:.6f}")
    print("TEST cost:", f"{test_metrics_rank['cost']:.2f}", f"(FP={test_metrics_rank['fp']}, FN={test_metrics_rank['fn']})")
    for k in RECALL_AT_K_LIST:
        print(f"TEST recall@{k}: {test_metrics_rank[f'recall@{k}']:.6f}")

    # Guardar modelo (criterio 2) aparte
    dump(best_rank_model, models_dir / "best_model_rank.joblib")

    # Guardar reporte
    report_path = models_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nGuardado:")
    print(" - Modelo (min costo):", models_dir / "best_model.joblib")
    print(" - Modelo (mejor ranking):", models_dir / "best_model_rank.joblib")
    print(" - Reporte:", report_path)


if __name__ == "__main__":
    main()

