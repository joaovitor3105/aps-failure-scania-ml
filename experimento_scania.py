"""
Experimento Completo de Manutenção Preditiva - Scania APS Failure Dataset
Disciplina: Tópicos Especiais em Sistemas Inteligentes: Aprendizado de Máquina
Autor: João Vitor Lobato Romualdo (CEFET-MG)

Executa integralmente o protocolo da Etapa III (com as recomendações docentes):
1. Validação Cruzada Estratificada (5 dobras) e Busca de Hiperparâmetros sem vazamento
2. Estabilidade sob 5 Estratégias de Tratamento de Atributos Ausentes (E1 a E5)
3. Calibração de Probabilidades (Platt Scaling Sigmoidal e Regressão Isotônica) + Limiar de Bayes vs. OOF
4. Análise de Sensibilidade à Razão de Custos (c_FN / c_FP in {10, 25, 50, 100, 200})
5. Geração de Figuras (PDF e PNG) e Tabelas Resumo (JSON/CSV) para o Artigo IEEE
"""

import json
import os
import time
import warnings
import zipfile
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sklearn
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(BASE_DIR, "dados", "aps+failure+at+scania+trucks.zip")
OUT_DIR = os.path.join(BASE_DIR, "Praticas")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
THRESHOLDS = np.linspace(0.01, 0.99, 99)


def carregar_dados():
    print("=" * 72)
    print("1. CARREGANDO DATASET SCANIA APS FAILURE...")
    print("=" * 72)
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        with z.open("aps_failure_training_set.csv") as f:
            df_train = pd.read_csv(f, skiprows=20, na_values="na")
        with z.open("aps_failure_test_set.csv") as f:
            df_test = pd.read_csv(f, skiprows=20, na_values="na")

    y_train = (df_train["class"] == "pos").astype(int).values
    y_test = (df_test["class"] == "pos").astype(int).values
    X_train_df = df_train.drop(columns=["class"])
    X_test_df = df_test.drop(columns=["class"])

    print(
        f"Treino: {X_train_df.shape} (pos={y_train.sum()}, {y_train.mean()*100:.2f}%) | "
        f"Teste: {X_test_df.shape} (pos={y_test.sum()}, {y_test.mean()*100:.2f}%) "
        f"[{time.time() - t0:.2f}s]"
    )
    return X_train_df, y_train, X_test_df, y_test


def calcular_custo_vetorizado(y_true, y_prob, thresholds=THRESHOLDS, c_fp=10, c_fn=500):
    """Calcula o custo C = c_fp * FP + c_fn * FN para todos os limiares simultaneamente."""
    # y_true: (N,), y_prob: (N,), thresholds: (T,)
    preds = y_prob[:, None] >= thresholds[None, :]  # (N, T)
    y_col = y_true[:, None].astype(bool)
    fp = np.sum(preds & (~y_col), axis=0)
    fn = np.sum((~preds) & y_col, axis=0)
    custos = c_fp * fp + c_fn * fn
    return custos, fp, fn


def encontrar_melhor_limiar_oof(y_true, oof_probs, folds, c_fp=10, c_fn=500):
    """Encontra o limiar que minimiza o custo total OOF e calcula média/DP por dobra."""
    custos_totais, _, _ = calcular_custo_scania_curva(y_true, oof_probs, THRESHOLDS, c_fp, c_fn)
    best_idx = int(np.argmin(custos_totais))
    best_th = float(THRESHOLDS[best_idx])
    best_cost_total = int(custos_totais[best_idx])

    # Custo nas 5 dobras usando o limiar ótimo (escalado para o tamanho do teste = 16.000 para comparabilidade, e bruto por dobra)
    fold_costs_raw = []
    fold_costs_scaled16k = []
    fold_recalls = []
    fold_f1s = []
    for tr_idx, val_idx in folds:
        y_v = y_true[val_idx]
        p_v = (oof_probs[val_idx] >= best_th).astype(int)
        cm = confusion_matrix(y_v, p_v)
        tn, fp, fn, tp = cm.ravel()
        c_raw = int(c_fp * fp + c_fn * fn)
        fold_costs_raw.append(c_raw)
        fold_costs_scaled16k.append(c_raw * (16000.0 / len(y_v)))
        fold_recalls.append(recall_score(y_v, p_v))
        fold_f1s.append(f1_score(y_v, p_v))

    # Validação Cruzada Aninhada (Nested CV): seleciona limiar nas 4 dobras internas e avalia na dobra retida
    nested_costs_raw = []
    nested_ths = []
    for tr_idx, val_idx in folds:
        custos_inner, _, _ = calcular_custo_scania_curva(y_true[tr_idx], oof_probs[tr_idx], THRESHOLDS, c_fp, c_fn)
        th_k = float(THRESHOLDS[int(np.argmin(custos_inner))])
        nested_ths.append(round(th_k, 2))
        y_v = y_true[val_idx]
        p_v = (oof_probs[val_idx] >= th_k).astype(int)
        cm_k = confusion_matrix(y_v, p_v)
        _, fp_k, fn_k, _ = cm_k.ravel()
        nested_costs_raw.append(int(c_fp * fp_k + c_fn * fn_k))

    return {
        "best_th": round(best_th, 4),
        "oof_cost_total_60k": best_cost_total,
        "oof_cost_fold_mean": round(float(np.mean(fold_costs_raw)), 1),
        "oof_cost_fold_std": round(float(np.std(fold_costs_raw, ddof=1)), 1),
        "nested_cost_fold_mean": round(float(np.mean(nested_costs_raw)), 1),
        "nested_cost_fold_std": round(float(np.std(nested_costs_raw, ddof=1)), 1),
        "nested_ths": nested_ths,
        "oof_cost_equiv16k_mean": round(float(np.mean(fold_costs_scaled16k)), 1),
        "oof_cost_equiv16k_std": round(float(np.std(fold_costs_scaled16k, ddof=1)), 1),
        "oof_recall_mean": round(float(np.mean(fold_recalls)), 4),
        "oof_recall_std": round(float(np.std(fold_recalls, ddof=1)), 4),
        "oof_f1_mean": round(float(np.mean(fold_f1s)), 4),
        "oof_f1_std": round(float(np.std(fold_f1s, ddof=1)), 4),
        "curva_custos_oof": custos_totais.tolist(),
    }


def bootstrap_custo_teste(y_te, pred_vec, n_boot=10000, seed=RANDOM_STATE, c_fp=10, c_fn=500):
    """Calcula IC 95% via reamostragem bootstrap (10.000 amostras) no conjunto de teste."""
    rng = np.random.default_rng(seed)
    n = len(y_te)
    fp_i = ((pred_vec == 1) & (y_te == 0)).astype(np.int32)
    fn_i = ((pred_vec == 0) & (y_te == 1)).astype(np.int32)
    c_vec = c_fp * fp_i + c_fn * fn_i
    tots = np.zeros(n_boot, dtype=np.int64)
    chunk = 500
    for s in range(0, n_boot, chunk):
        e = min(s + chunk, n_boot)
        idx = rng.integers(0, n, size=(e - s, n))
        tots[s:e] = c_vec[idx].sum(axis=1)
    ci_low, ci_high = np.percentile(tots, [2.5, 97.5])
    return [int(round(ci_low)), int(round(ci_high))], round(float(np.std(tots, ddof=1)), 1), tots


def calcular_custo_scania_curva(y_true, y_prob, thresholds=THRESHOLDS, c_fp=10, c_fn=500):
    return calcular_custo_vetorizado(y_true, y_prob, thresholds, c_fp, c_fn)


def avaliar_no_teste(y_test, test_probs, th, c_fp=10, c_fn=500):
    preds = (test_probs >= th).astype(int)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    custo = int(c_fp * fp + c_fn * fn)
    return {
        "th": round(float(th), 4),
        "custo": custo,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "acc": round(float(accuracy_score(y_test, preds)), 4),
        "prec": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, test_probs)), 4),
        "pr_auc": round(float(average_precision_score(y_test, test_probs)), 4),
        "brier": round(float(brier_score_loss(y_test, np.clip(test_probs, 0, 1))), 4),
    }


def transformar_estrategia(X_tr_df, X_te_df, estrategia="E1"):
    """
    Aplica a estratégia de tratamento de dados ausentes ajustando estritamente em X_tr_df
    e aplicando em X_te_df (sem vazamento).
    E1: Corte >50% nulos + Mediana (Referência)
    E2: Corte >50% nulos + Média
    E3: Corte >50% nulos + Constante Zero
    E4: Corte >50% nulos + Mediana + Indicadores Binários de Ausência (MissingIndicator)
    E5: Sem descarte de colunas (170 atributos) + Mediana
    """
    if estrategia in ("E1", "E2", "E3", "E4"):
        missing_pct = X_tr_df.isnull().mean() * 100.0
        keep_cols = missing_pct[missing_pct <= 50.0].index.tolist()
        X_tr = X_tr_df[keep_cols].values
        X_te = X_te_df[keep_cols].values
    elif estrategia == "E5":
        keep_cols = X_tr_df.columns.tolist()
        X_tr = X_tr_df.values
        X_te = X_te_df.values
    else:
        raise ValueError(f"Estratégia desconhecida: {estrategia}")

    if estrategia in ("E1", "E5"):
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_tr)
        X_te_imp = imp.transform(X_te)
    elif estrategia == "E2":
        imp = SimpleImputer(strategy="mean")
        X_tr_imp = imp.fit_transform(X_tr)
        X_te_imp = imp.transform(X_te)
    elif estrategia == "E3":
        imp = SimpleImputer(strategy="constant", fill_value=0.0)
        X_tr_imp = imp.fit_transform(X_tr)
        X_te_imp = imp.transform(X_te)
    elif estrategia == "E4":
        imp = SimpleImputer(strategy="median")
        X_tr_num = imp.fit_transform(X_tr)
        X_te_num = imp.transform(X_te)
        ind = MissingIndicator(features="missing-only")
        X_tr_ind = ind.fit_transform(X_tr).astype(float)
        X_te_ind = ind.transform(X_te).astype(float)
        X_tr_imp = np.hstack([X_tr_num, X_tr_ind])
        X_te_imp = np.hstack([X_te_num, X_te_ind])

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_imp)
    X_te_scaled = scaler.transform(X_te_imp)

    return X_tr_imp, X_te_imp, X_tr_scaled, X_te_scaled, X_tr_imp.shape[1]


def main():
    t_global = time.time()
    X_train_df, y_train, X_test_df, y_test = carregar_dados()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(X_train_df, y_train))

    # Pré-computa as matrizes por dobra para E1 (evita reprocessar durante a busca em grade)
    print("\n" + "=" * 72)
    print("2. PREPARANDO DOBRAS DE VALIDAÇÃO CRUZADA (5-FOLD, ESTRATÉGIA E1)...")
    print("=" * 72)
    t0 = time.time()
    e1_folds_data = []
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        X_tr_imp, X_val_imp, X_tr_sc, X_val_sc, n_feats = transformar_estrategia(
            X_train_df.iloc[tr_idx], X_train_df.iloc[val_idx], "E1"
        )
        e1_folds_data.append((tr_idx, val_idx, X_tr_imp, X_val_imp, X_tr_sc, X_val_sc))
    X_tr_full_imp, X_te_full_imp, X_tr_full_sc, X_te_full_sc, n_feats_e1 = transformar_estrategia(
        X_train_df, X_test_df, "E1"
    )
    print(f"Dobras E1 prontas ({n_feats_e1} atributos ativos) em {time.time() - t0:.2f}s")

    # =========================================================================
    # BLOCO 1: BUSCA DE HIPERPARÂMETROS (5-FOLD CV)
    # =========================================================================
    print("\n" + "=" * 72)
    print("3. BUSCA DE HIPERPARÂMETROS EM 5-FOLD CV (LR E RF)...")
    print("=" * 72)

    lr_grid_C = [0.001, 0.01, 0.1, 1.0, 10.0]
    lr_search_results = []
    best_lr_C = None
    best_lr_oof_probs = None
    best_lr_cost = float("inf")

    for c_val in lr_grid_C:
        oof_probs = np.zeros(len(y_train))
        for tr_idx, val_idx, X_tr_imp, X_val_imp, X_tr_sc, X_val_sc in e1_folds_data:
            clf_lr = LogisticRegression(
                C=c_val, class_weight="balanced", max_iter=500, random_state=RANDOM_STATE, n_jobs=-1
            )
            clf_lr.fit(X_tr_sc, y_train[tr_idx])
            oof_probs[val_idx] = clf_lr.predict_proba(X_val_sc)[:, 1]

        oof_summary = encontrar_melhor_limiar_oof(y_train, oof_probs, folds)
        roc_oof = round(float(roc_auc_score(y_train, oof_probs)), 4)
        pr_oof = round(float(average_precision_score(y_train, oof_probs)), 4)
        entry = {
            "C": c_val,
            "best_th": oof_summary["best_th"],
            "oof_cost_total_60k": oof_summary["oof_cost_total_60k"],
            "oof_cost_fold_mean": oof_summary["oof_cost_fold_mean"],
            "oof_cost_fold_std": oof_summary["oof_cost_fold_std"],
            "oof_roc_auc": roc_oof,
            "oof_pr_auc": pr_oof,
        }
        lr_search_results.append(entry)
        print(
            f"  [LR C={c_val:<5}] th_OOF={entry['best_th']:.2f} | "
            f"Custo/Dobra=${entry['oof_cost_fold_mean']:,.1f} ± {entry['oof_cost_fold_std']:,.1f} | "
            f"ROC-AUC={roc_oof:.4f} | PR-AUC={pr_oof:.4f}"
        )
        if oof_summary["oof_cost_total_60k"] < best_lr_cost:
            best_lr_cost = oof_summary["oof_cost_total_60k"]
            best_lr_C = c_val
            best_lr_oof_probs = oof_probs.copy()

    rf_grid_cfgs = [
        {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 1},
        {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 5},
        {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 1},
        {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 5},
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 5},
    ]
    rf_search_results = []
    best_rf_cfg = None
    best_rf_oof_probs = None
    best_rf_cost = float("inf")

    for cfg in rf_grid_cfgs:
        oof_probs = np.zeros(len(y_train))
        for tr_idx, val_idx, X_tr_imp, X_val_imp, X_tr_sc, X_val_sc in e1_folds_data:
            clf_rf = RandomForestClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                min_samples_leaf=cfg["min_samples_leaf"],
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            clf_rf.fit(X_tr_imp, y_train[tr_idx])
            oof_probs[val_idx] = clf_rf.predict_proba(X_val_imp)[:, 1]

        oof_summary = encontrar_melhor_limiar_oof(y_train, oof_probs, folds)
        roc_oof = round(float(roc_auc_score(y_train, oof_probs)), 4)
        pr_oof = round(float(average_precision_score(y_train, oof_probs)), 4)
        entry = {
            "n_estimators": cfg["n_estimators"],
            "max_depth": str(cfg["max_depth"]),
            "min_samples_leaf": cfg["min_samples_leaf"],
            "best_th": oof_summary["best_th"],
            "oof_cost_total_60k": oof_summary["oof_cost_total_60k"],
            "oof_cost_fold_mean": oof_summary["oof_cost_fold_mean"],
            "oof_cost_fold_std": oof_summary["oof_cost_fold_std"],
            "oof_roc_auc": roc_oof,
            "oof_pr_auc": pr_oof,
        }
        rf_search_results.append(entry)
        print(
            f"  [RF B={cfg['n_estimators']}, depth={str(cfg['max_depth']):<4}, leaf={cfg['min_samples_leaf']}] "
            f"th_OOF={entry['best_th']:.2f} | Custo/Dobra=${entry['oof_cost_fold_mean']:,.1f} ± {entry['oof_cost_fold_std']:,.1f} | "
            f"ROC-AUC={roc_oof:.4f} | PR-AUC={pr_oof:.4f}"
        )
        if oof_summary["oof_cost_total_60k"] < best_rf_cost:
            best_rf_cost = oof_summary["oof_cost_total_60k"]
            best_rf_cfg = cfg.copy()
            best_rf_oof_probs = oof_probs.copy()

    best_lr_oof_summary = encontrar_melhor_limiar_oof(y_train, best_lr_oof_probs, folds)
    best_rf_oof_summary = encontrar_melhor_limiar_oof(y_train, best_rf_oof_probs, folds)

    print(f"\n-> Melhor LR selecionada no treino: C={best_lr_C} (th_OOF={best_lr_oof_summary['best_th']})")
    print(f"-> Melhor RF selecionada no treino: {best_rf_cfg} (th_OOF={best_rf_oof_summary['best_th']})")

    # Treina os melhores modelos no treino completo (E1) e avalia no teste
    t_lr = time.time()
    lr_final = LogisticRegression(
        C=best_lr_C, class_weight="balanced", max_iter=500, random_state=RANDOM_STATE, n_jobs=-1
    )
    lr_final.fit(X_tr_full_sc, y_train)
    lr_train_time = round(time.time() - t_lr, 3)
    t_inf = time.time()
    lr_test_probs = lr_final.predict_proba(X_te_full_sc)[:, 1]
    lr_infer_time_ms = round((time.time() - t_inf) * 1000.0, 2)

    t_rf = time.time()
    rf_final = RandomForestClassifier(
        n_estimators=best_rf_cfg["n_estimators"],
        max_depth=best_rf_cfg["max_depth"],
        min_samples_leaf=best_rf_cfg["min_samples_leaf"],
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_final.fit(X_tr_full_imp, y_train)
    rf_train_time = round(time.time() - t_rf, 3)
    t_inf = time.time()
    rf_test_probs = rf_final.predict_proba(X_te_full_imp)[:, 1]
    rf_infer_time_ms = round((time.time() - t_inf) * 1000.0, 2)

    lr_test_default = avaliar_no_teste(y_test, lr_test_probs, 0.50)
    lr_test_opt = avaliar_no_teste(y_test, lr_test_probs, best_lr_oof_summary["best_th"])
    rf_test_default = avaliar_no_teste(y_test, rf_test_probs, 0.50)
    rf_test_opt = avaliar_no_teste(y_test, rf_test_probs, best_rf_oof_summary["best_th"])

    print("\n--- RESULTADOS PRINCIPAIS NO CONJUNTO DE TESTE (16.000 INSTÂNCIAS) ---")
    for label, res in [
        ("LR (th=0.50 padrão)", lr_test_default),
        (f"LR (th={best_lr_oof_summary['best_th']:.2f} OOF)", lr_test_opt),
        ("RF (th=0.50 padrão)", rf_test_default),
        (f"RF (th={best_rf_oof_summary['best_th']:.2f} OOF)", rf_test_opt),
    ]:
        print(
            f"{label:<24} -> Custo=${res['custo']:,} | FP={res['fp']} FN={res['fn']} TP={res['tp']} | "
            f"Recall={res['recall']:.4f} Prec={res['prec']:.4f} F1={res['f1']:.4f} "
            f"ROC-AUC={res['roc_auc']:.4f} PR-AUC={res['pr_auc']:.4f}"
        )

    # =========================================================================
    # BLOCO 2: ESTABILIDADE SOB DIFERENTES ESTRATÉGIAS DE IMPUTAÇÃO (E1 a E5)
    # =========================================================================
    print("\n" + "=" * 72)
    print("4. ESTABILIDADE FRENTE AO TRATAMENTO DE ATRIBUTOS AUSENTES (E1 A E5)...")
    print("=" * 72)

    estrategias_info = [
        ("E1", "Corte >50% + Mediana (Ref.)"),
        ("E2", "Corte >50% + Média"),
        ("E3", "Corte >50% + Constante Zero"),
        ("E4", "Corte >50% + Mediana + Missing Ind."),
        ("E5", "Sem Descarte (170 col.) + Mediana"),
    ]
    imputacao_results = []

    for cod, desc in estrategias_info:
        t_e = time.time()
        if cod == "E1":
            oof_lr = best_lr_oof_probs
            oof_rf = best_rf_oof_probs
            te_probs_lr = lr_test_probs
            te_probs_rf = rf_test_probs
            n_cols = n_feats_e1
        else:
            oof_lr = np.zeros(len(y_train))
            oof_rf = np.zeros(len(y_train))
            for tr_idx, val_idx in folds:
                X_tr_i, X_val_i, X_tr_s, X_val_s, n_cols = transformar_estrategia(
                    X_train_df.iloc[tr_idx], X_train_df.iloc[val_idx], cod
                )
                m_lr = LogisticRegression(
                    C=best_lr_C, class_weight="balanced", max_iter=500, random_state=RANDOM_STATE, n_jobs=-1
                )
                m_lr.fit(X_tr_s, y_train[tr_idx])
                oof_lr[val_idx] = m_lr.predict_proba(X_val_s)[:, 1]

                m_rf = RandomForestClassifier(
                    n_estimators=best_rf_cfg["n_estimators"],
                    max_depth=best_rf_cfg["max_depth"],
                    min_samples_leaf=best_rf_cfg["min_samples_leaf"],
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
                m_rf.fit(X_tr_i, y_train[tr_idx])
                oof_rf[val_idx] = m_rf.predict_proba(X_val_i)[:, 1]

            X_tr_fi, X_te_fi, X_tr_fs, X_te_fs, n_cols = transformar_estrategia(X_train_df, X_test_df, cod)
            m_lr_full = LogisticRegression(
                C=best_lr_C, class_weight="balanced", max_iter=500, random_state=RANDOM_STATE, n_jobs=-1
            )
            m_lr_full.fit(X_tr_fs, y_train)
            te_probs_lr = m_lr_full.predict_proba(X_te_fs)[:, 1]

            m_rf_full = RandomForestClassifier(
                n_estimators=best_rf_cfg["n_estimators"],
                max_depth=best_rf_cfg["max_depth"],
                min_samples_leaf=best_rf_cfg["min_samples_leaf"],
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            m_rf_full.fit(X_tr_fi, y_train)
            te_probs_rf = m_rf_full.predict_proba(X_te_fi)[:, 1]

        oof_res_lr = encontrar_melhor_limiar_oof(y_train, oof_lr, folds)
        oof_res_rf = encontrar_melhor_limiar_oof(y_train, oof_rf, folds)
        te_res_lr = avaliar_no_teste(y_test, te_probs_lr, oof_res_lr["best_th"])
        te_res_rf = avaliar_no_teste(y_test, te_probs_rf, oof_res_rf["best_th"])

        row = {
            "estrategia": cod,
            "descricao": desc,
            "n_atributos": int(n_cols),
            "lr_th_oof": oof_res_lr["best_th"],
            "lr_oof_fold_mean": oof_res_lr["oof_cost_fold_mean"],
            "lr_oof_fold_std": oof_res_lr["oof_cost_fold_std"],
            "lr_test_custo": te_res_lr["custo"],
            "lr_test_fp": te_res_lr["fp"],
            "lr_test_fn": te_res_lr["fn"],
            "lr_test_recall": te_res_lr["recall"],
            "lr_test_roc_auc": te_res_lr["roc_auc"],
            "lr_test_pr_auc": te_res_lr["pr_auc"],
            "rf_th_oof": oof_res_rf["best_th"],
            "rf_oof_fold_mean": oof_res_rf["oof_cost_fold_mean"],
            "rf_oof_fold_std": oof_res_rf["oof_cost_fold_std"],
            "rf_test_custo": te_res_rf["custo"],
            "rf_test_fp": te_res_rf["fp"],
            "rf_test_fn": te_res_rf["fn"],
            "rf_test_recall": te_res_rf["recall"],
            "rf_test_roc_auc": te_res_rf["roc_auc"],
            "rf_test_pr_auc": te_res_rf["pr_auc"],
            "diferenca_custo_lr_menos_rf": te_res_lr["custo"] - te_res_rf["custo"],
            "razao_custo_lr_rf": round(te_res_lr["custo"] / te_res_rf["custo"], 2),
        }
        imputacao_results.append(row)
        print(
            f"[{cod}] {desc:<33} ({n_cols} cols) | "
            f"LR Teste=${te_res_lr['custo']:,} (FP={te_res_lr['fp']}, FN={te_res_lr['fn']}, th={oof_res_lr['best_th']:.2f}) | "
            f"RF Teste=${te_res_rf['custo']:,} (FP={te_res_rf['fp']}, FN={te_res_rf['fn']}, th={oof_res_rf['best_th']:.2f}) "
            f"[{time.time() - t_e:.1f}s]"
        )

    # =========================================================================
    # BLOCO 3: CALIBRAÇÃO DE PROBABILIDADES (PLATT SCALING E ISOTÔNICA)
    # =========================================================================
    print("\n" + "=" * 72)
    print("5. CALIBRAÇÃO DE PROBABILIDADES (PLATT E ISOTÔNICA) + LIMIAR DE BAYES...")
    print("=" * 72)

    # Ajusta calibradores sem vazamento: para cada dobra, ajusta nos 4/5 de treino (usando best_oof_probs)
    # e prediz no 1/5 de validação; para o teste, ajusta nos 60k OOF e aplica em test_probs.
    def calibrar_oof_e_teste(oof_raw, test_raw, y_tr, folds_list):
        eps = 1e-6
        oof_logit = np.log(np.clip(oof_raw, eps, 1 - eps) / (1 - np.clip(oof_raw, eps, 1 - eps))).reshape(-1, 1)
        test_logit = np.log(np.clip(test_raw, eps, 1 - eps) / (1 - np.clip(test_raw, eps, 1 - eps))).reshape(-1, 1)

        oof_platt = np.zeros_like(oof_raw)
        oof_iso = np.zeros_like(oof_raw)

        for tr_idx, val_idx in folds_list:
            # Platt Scaling sobre o escore/log-odds
            platt = LogisticRegression(C=1e5, solver="lbfgs")
            platt.fit(oof_logit[tr_idx], y_tr[tr_idx])
            oof_platt[val_idx] = platt.predict_proba(oof_logit[val_idx])[:, 1]

            # Regressão Isotônica (PAVA)
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(oof_raw[tr_idx], y_tr[tr_idx])
            oof_iso[val_idx] = iso.transform(oof_raw[val_idx])

        platt_full = LogisticRegression(C=1e5, solver="lbfgs")
        platt_full.fit(oof_logit, y_tr)
        test_platt = platt_full.predict_proba(test_logit)[:, 1]

        iso_full = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso_full.fit(oof_raw, y_tr)
        test_iso = iso_full.transform(test_raw)

        return oof_platt, test_platt, oof_iso, test_iso

    lr_oof_platt, lr_test_platt, lr_oof_iso, lr_test_iso = calibrar_oof_e_teste(
        best_lr_oof_probs, lr_test_probs, y_train, folds
    )
    rf_oof_platt, rf_test_platt, rf_oof_iso, rf_test_iso = calibrar_oof_e_teste(
        best_rf_oof_probs, rf_test_probs, y_train, folds
    )

    th_bayes_nominal = 10.0 / (10.0 + 500.0)  # 0.019608

    calibracao_results = []
    modelos_cal = [
        ("Regressão Logística", "Sem Calibração (Pond. Balanceada)", best_lr_oof_probs, lr_test_probs),
        ("Regressão Logística", "Escalonamento de Platt (Sigmoidal)", lr_oof_platt, lr_test_platt),
        ("Regressão Logística", "Regressão Isotônica (PAVA)", lr_oof_iso, lr_test_iso),
        ("Random Forest", "Sem Calibração (Pond. Subamostra)", best_rf_oof_probs, rf_test_probs),
        ("Random Forest", "Escalonamento de Platt (Sigmoidal)", rf_oof_platt, rf_test_platt),
        ("Random Forest", "Regressão Isotônica (PAVA)", rf_oof_iso, rf_test_iso),
    ]

    for mod_nome, cal_nome, oof_p, test_p in modelos_cal:
        oof_opt = encontrar_melhor_limiar_oof(y_train, oof_p, folds)
        res_bayes = avaliar_no_teste(y_test, test_p, th_bayes_nominal)
        res_oof_th = avaliar_no_teste(y_test, test_p, oof_opt["best_th"])
        brier_oof = round(float(brier_score_loss(y_train, np.clip(oof_p, 0, 1))), 5)
        brier_test = round(float(brier_score_loss(y_test, np.clip(test_p, 0, 1))), 5)

        item = {
            "modelo": mod_nome,
            "calibracao": cal_nome,
            "brier_oof": brier_oof,
            "brier_test": brier_test,
            "prob_media_teste": round(float(np.mean(test_p)), 4),
            "th_bayes": round(th_bayes_nominal, 4),
            "custo_teste_bayes": res_bayes["custo"],
            "fp_bayes": res_bayes["fp"],
            "fn_bayes": res_bayes["fn"],
            "recall_bayes": res_bayes["recall"],
            "th_oof": oof_opt["best_th"],
            "custo_teste_oof": res_oof_th["custo"],
            "fp_oof": res_oof_th["fp"],
            "fn_oof": res_oof_th["fn"],
            "recall_oof": res_oof_th["recall"],
            "roc_auc_test": res_oof_th["roc_auc"],
            "pr_auc_test": res_oof_th["pr_auc"],
        }
        calibracao_results.append(item)
        print(
            f"[{mod_nome} | {cal_nome:<34}] Brier={brier_test:.5f} | "
            f"Bayes(th=0.02): Custo=${res_bayes['custo']:,} (FP={res_bayes['fp']}, FN={res_bayes['fn']}) | "
            f"OOF(th={oof_opt['best_th']:.2f}): Custo=${res_oof_th['custo']:,} (FP={res_oof_th['fp']}, FN={res_oof_th['fn']})"
        )

    # =========================================================================
    # BLOCO 4: ANÁLISE DE SENSIBILIDADE AOS CUSTOS DE ERRO (c_FN / c_FP)
    # =========================================================================
    print("\n" + "=" * 72)
    print("6. ANÁLISE DE SENSIBILIDADE À RAZÃO DE CUSTOS (c_FN / c_FP)...")
    print("=" * 72)

    cost_ratios = [10, 25, 50, 100, 200]  # c_FP = 10 fixo => c_FN in {100, 250, 500, 1000, 2000}
    sensibilidade_results = []

    for rho in cost_ratios:
        c_fp = 10
        c_fn = 10 * rho
        th_b = c_fp / (c_fp + c_fn)

        # Custo do classificador trivial no teste: min(inspecionar todos = 10*15625, inspecionar nenhum = c_fn*375)
        custo_trivial_teste = int(min(c_fp * np.sum(y_test == 0), c_fn * np.sum(y_test == 1)))

        oof_lr_r = encontrar_melhor_limiar_oof(y_train, best_lr_oof_probs, folds, c_fp=c_fp, c_fn=c_fn)
        oof_rf_r = encontrar_melhor_limiar_oof(y_train, best_rf_oof_probs, folds, c_fp=c_fp, c_fn=c_fn)
        oof_lr_iso_r = encontrar_melhor_limiar_oof(y_train, lr_oof_iso, folds, c_fp=c_fp, c_fn=c_fn)
        oof_rf_iso_r = encontrar_melhor_limiar_oof(y_train, rf_oof_iso, folds, c_fp=c_fp, c_fn=c_fn)

        te_lr_r = avaliar_no_teste(y_test, lr_test_probs, oof_lr_r["best_th"], c_fp=c_fp, c_fn=c_fn)
        te_rf_r = avaliar_no_teste(y_test, rf_test_probs, oof_rf_r["best_th"], c_fp=c_fp, c_fn=c_fn)
        te_lr_iso_bayes = avaliar_no_teste(y_test, lr_test_iso, th_b, c_fp=c_fp, c_fn=c_fn)
        te_rf_iso_bayes = avaliar_no_teste(y_test, rf_test_iso, th_b, c_fp=c_fp, c_fn=c_fn)
        te_rf_iso_oof = avaliar_no_teste(y_test, rf_test_iso, oof_rf_iso_r["best_th"], c_fp=c_fp, c_fn=c_fn)

        s_row = {
            "rho": rho,
            "c_fp": c_fp,
            "c_fn": c_fn,
            "th_bayes": round(th_b, 4),
            "custo_trivial_teste": custo_trivial_teste,
            "lr_th_oof": oof_lr_r["best_th"],
            "lr_custo_teste": te_lr_r["custo"],
            "lr_custo_norm": round(te_lr_r["custo"] / custo_trivial_teste, 4),
            "lr_fp": te_lr_r["fp"],
            "lr_fn": te_lr_r["fn"],
            "lr_recall": te_lr_r["recall"],
            "lr_iso_bayes_custo": te_lr_iso_bayes["custo"],
            "rf_th_oof": oof_rf_r["best_th"],
            "rf_custo_teste": te_rf_r["custo"],
            "rf_custo_norm": round(te_rf_r["custo"] / custo_trivial_teste, 4),
            "rf_fp": te_rf_r["fp"],
            "rf_fn": te_rf_r["fn"],
            "rf_recall": te_rf_r["recall"],
            "rf_iso_bayes_custo": te_rf_iso_bayes["custo"],
            "rf_iso_oof_custo": te_rf_iso_oof["custo"],
            "sobrecusto_lr_abs": te_lr_r["custo"] - te_rf_r["custo"],
            "razao_lr_rf": round(te_lr_r["custo"] / te_rf_r["custo"], 2),
        }
        sensibilidade_results.append(s_row)
        print(
            f"[rho={rho:>3}:1 (c_FN=${c_fn:<4})] Trivial=${custo_trivial_teste:>7,} | "
            f"LR(th={oof_lr_r['best_th']:.2f})=${te_lr_r['custo']:>6,} (FP={te_lr_r['fp']}, FN={te_lr_r['fn']}) | "
            f"RF(th={oof_rf_r['best_th']:.2f})=${te_rf_r['custo']:>6,} (FP={te_rf_r['fp']}, FN={te_rf_r['fn']}) | "
            f"RF-Iso(Bayes th={th_b:.3f})=${te_rf_iso_bayes['custo']:>6,} | Razão LR/RF={s_row['razao_lr_rf']:.2f}x"
        )

    # =========================================================================
    # BLOCO 5: GERAÇÃO DAS FIGURAS DE ALTA RESOLUÇÃO
    # =========================================================================
    print("\n" + "=" * 72)
    print("7. GERANDO FIGURAS CIENTÍFICAS (PDF E PNG)...")
    print("=" * 72)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
    })

    # Figura 1: Curvas de Custo em função do Limiar theta (Validação OOF vs Teste)
    custos_te_lr, _, _ = calcular_custo_vetorizado(y_test, lr_test_probs, THRESHOLDS)
    custos_te_rf, _, _ = calcular_custo_vetorizado(y_test, rf_test_probs, THRESHOLDS)
    custos_oof_lr = np.array(best_lr_oof_summary["curva_custos_oof"]) * (16000.0 / 60000.0)
    custos_oof_rf = np.array(best_rf_oof_summary["curva_custos_oof"]) * (16000.0 / 60000.0)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    ax.plot(THRESHOLDS, custos_oof_lr / 1000.0, "--", color="#1f77b4", alpha=0.8, label="LR - Validação OOF (escalado 16k)")
    ax.plot(THRESHOLDS, custos_te_lr / 1000.0, "-", color="#1f77b4", lw=2, label="LR - Conjunto de Teste")
    ax.plot(THRESHOLDS, custos_oof_rf / 1000.0, "--", color="#2ca02c", alpha=0.8, label="RF - Validação OOF (escalado 16k)")
    ax.plot(THRESHOLDS, custos_te_rf / 1000.0, "-", color="#2ca02c", lw=2, label="RF - Conjunto de Teste")

    ax.axvline(best_lr_oof_summary["best_th"], color="#1f77b4", linestyle=":", lw=1.5)
    ax.scatter([best_lr_oof_summary["best_th"]], [lr_test_opt["custo"] / 1000.0], color="#1f77b4", zorder=5, s=55,
               label=f"Ótimo OOF LR ($\\theta={best_lr_oof_summary['best_th']:.2f}$, \\${lr_test_opt['custo']:,})")
    ax.axvline(best_rf_oof_summary["best_th"], color="#2ca02c", linestyle=":", lw=1.5)
    ax.scatter([best_rf_oof_summary["best_th"]], [rf_test_opt["custo"] / 1000.0], color="#2ca02c", zorder=5, s=55,
               label=f"Ótimo OOF RF ($\\theta={best_rf_oof_summary['best_th']:.2f}$, \\${rf_test_opt['custo']:,})")

    ax.set_xlabel("Limiar de Decisão ($\\theta$)")
    ax.set_ylabel("Custo Total Scania (Milhares de US$)")
    ax.set_title("(a) Otimização de Limiar Out-of-Fold vs. Teste")
    ax.set_ylim(0, 80)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=8.2)

    # Subplot (b): Decomposição do custo (FP vs FN) no teste
    ax2 = axes[1]
    labels_mod = [
        "LR\n($\\theta=0,50$)",
        f"LR Otim.\n($\\theta={best_lr_oof_summary['best_th']:.2f}$)",
        "RF\n($\\theta=0,50$)",
        f"RF Otim.\n($\\theta={best_rf_oof_summary['best_th']:.2f}$)",
    ]
    c_fps = [lr_test_default["fp"] * 10, lr_test_opt["fp"] * 10, rf_test_default["fp"] * 10, rf_test_opt["fp"] * 10]
    c_fns = [lr_test_default["fn"] * 500, lr_test_opt["fn"] * 500, rf_test_default["fn"] * 500, rf_test_opt["fn"] * 500]
    x_pos = np.arange(len(labels_mod))
    b1 = ax2.bar(x_pos, np.array(c_fps) / 1000.0, width=0.52, color="#ff7f0e", label="Custo por Falsos Positivos ($10 × FP)")
    b2 = ax2.bar(x_pos, np.array(c_fns) / 1000.0, width=0.52, bottom=np.array(c_fps) / 1000.0, color="#d62728",
                 label="Custo por Falsos Negativos ($500 × FN)")
    for i, tot in enumerate(np.array(c_fps) + np.array(c_fns)):
        ax2.text(i, tot / 1000.0 + 0.8, f"\\${tot:,}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels_mod)
    ax2.set_ylabel("Custo no Teste (Milhares de US$)")
    ax2.set_title("(b) Decomposição Financeira do Erro no Teste")
    ax2.set_ylim(0, max((np.array(c_fps) + np.array(c_fns)) / 1000.0) * 1.18)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_custo_limiar.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig_custo_limiar.png"), bbox_inches="tight")
    plt.close(fig)

    # Figura 2: Estabilidade sob Diferentes Estratégias de Imputação (E1 a E5)
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    cods = [r["estrategia"] + "\n(" + r["descricao"].split("+")[1].strip()[:12] + ("+Ind" if "Ind" in r["descricao"] else "") + ")" for r in imputacao_results]
    cods_clean = [
        "E1: >50% + Mediana\n(162 atrib.)",
        "E2: >50% + Média\n(162 atrib.)",
        "E3: >50% + Zero\n(162 atrib.)",
        "E4: Mediana + Ind. MNAR\n(323 atrib.)",
        "E5: Sem Corte + Mediana\n(170 atrib.)",
    ]
    lr_costs = [r["lr_test_custo"] for r in imputacao_results]
    rf_costs = [r["rf_test_custo"] for r in imputacao_results]
    x = np.arange(len(cods_clean))
    w = 0.35
    rects1 = ax.bar(x - w / 2, lr_costs, w, label="Regressão Logística (Limiar OOF)", color="#1f77b4")
    rects2 = ax.bar(x + w / 2, rf_costs, w, label="Random Forest (Limiar OOF)", color="#2ca02c")

    for r in rects1:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2.0, h + 250, f"\\${int(h):,}", ha="center", va="bottom", fontsize=8.5, color="#1f77b4", fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2.0, h + 250, f"\\${int(h):,}", ha="center", va="bottom", fontsize=8.5, color="#2ca02c", fontweight="bold")

    ax.set_ylabel("Custo Total Scania no Teste (US$)")
    ax.set_title("Estabilidade do Custo no Teste sob Diferentes Estratégias de Tratamento de Dados Ausentes")
    ax.set_xticks(x)
    ax.set_xticklabels(cods_clean, fontsize=8.8)
    ax.set_ylim(0, max(lr_costs) * 1.16)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_estabilidade_imputacao.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig_estabilidade_imputacao.png"), bbox_inches="tight")
    plt.close(fig)

    # Figura 3: Calibração de Probabilidades (Curvas de Confiabilidade + Impacto no Limiar de Bayes vs OOF)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k:", label="Perfeitamente Calibrado")

    for probs, label, col, ls in [
        (lr_test_probs, f"LR Sem Calib. (Brier={brier_score_loss(y_test, lr_test_probs):.4f})", "#1f77b4", "--"),
        (lr_test_iso, f"LR Isotônica (Brier={brier_score_loss(y_test, lr_test_iso):.4f})", "#1f77b4", "-"),
        (rf_test_probs, f"RF Sem Calib. (Brier={brier_score_loss(y_test, rf_test_probs):.4f})", "#2ca02c", "--"),
        (rf_test_iso, f"RF Isotônica (Brier={brier_score_loss(y_test, rf_test_iso):.4f})", "#2ca02c", "-"),
    ]:
        prob_true, prob_pred = calibration_curve(y_test, np.clip(probs, 0, 1), n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", markersize=4, color=col, linestyle=ls, lw=1.8, label=label)

    ax.set_xlabel("Probabilidade Média Predita")
    ax.set_ylabel("Fração Empírica de Positivos (Teste)")
    ax.set_title("(a) Diagrama de Confiabilidade (Calibração no Teste)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=8.0)

    ax2 = axes[1]
    categorias = [
        "LR\nNão Calib.",
        "LR\nPlatt",
        "LR\nIsotônica",
        "RF\nNão Calib.",
        "RF\nPlatt",
        "RF\nIsotônica",
    ]
    custos_bayes = np.array([c["custo_teste_bayes"] for c in calibracao_results]) / 1000.0
    custos_oof_c = np.array([c["custo_teste_oof"] for c in calibracao_results]) / 1000.0
    y_cap = 22.0
    custos_bayes_plot = np.minimum(custos_bayes, y_cap)
    x_c = np.arange(len(categorias))
    w_c = 0.38
    b_bayes = ax2.bar(x_c - w_c / 2, custos_bayes_plot, w_c, color="#9467bd",
                      label="Limiar Teórico de Bayes ($\\theta^* = 0,0196$)")
    b_oof = ax2.bar(x_c + w_c / 2, custos_oof_c, w_c, color="#2ca02c",
                    label="Limiar Empírico Otimizado ($\\theta_{OOF}$)")
    for idx in range(len(categorias)):
        bar_b = b_bayes[idx]
        bar_o = b_oof[idx]
        val_b = custos_bayes[idx]
        val_o = custos_oof_c[idx]
        hb = bar_b.get_height()
        ho = bar_o.get_height()
        if val_b > y_cap:
            ax2.text(
                bar_b.get_x() + bar_b.get_width() / 2.0 + 0.16,
                y_cap - 0.8,
                f"\\${val_b:.1f}k ↑\n(fora de escala)",
                ha="left",
                va="top",
                fontsize=7.6,
                fontweight="bold",
                color="#5b2c83",
                bbox=dict(boxstyle="round,pad=0.2", fc="#f3e8ff", ec="#9467bd", lw=0.8),
            )
            ax2.text(
                bar_o.get_x() + bar_o.get_width() / 2.0 + 0.03,
                ho + 0.35,
                f"\\${val_o:.1f}k",
                ha="center",
                va="bottom",
                fontsize=7.4,
                fontweight="bold",
                color="#145214",
            )
        else:
            y_off_b = 0.25 if abs(hb - ho) < 0.4 else 0.35
            y_off_o = 1.25 if abs(hb - ho) < 0.4 else 0.35
            ax2.text(bar_b.get_x() + bar_b.get_width() / 2.0 - 0.04, hb + y_off_b, f"\\${val_b:.1f}k",
                     ha="center", va="bottom", fontsize=7.3, color="#4a2373")
            ax2.text(bar_o.get_x() + bar_o.get_width() / 2.0 + 0.04, ho + y_off_o, f"\\${val_o:.1f}k",
                     ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#145214")
    ax2.set_xticks(x_c)
    ax2.set_xticklabels(categorias, fontsize=8.5)
    ax2.set_ylabel("Custo no Teste (Milhares de US$)")
    ax2.set_title("(b) Custo no Teste: Regra de Bayes ($\\theta^*$) vs. Limiar OOF ($\\theta_{OOF}$)")
    ax2.set_ylim(0, y_cap + 3.2)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.2)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_calibracao_probabilidades.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig_calibracao_probabilidades.png"), bbox_inches="tight")
    plt.close(fig)

    # Figura 4: Análise de Sensibilidade à Razão de Custos (rho = c_FN / c_FP)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    rhos = [r["rho"] for r in sensibilidade_results]
    lr_norm = [r["lr_custo_norm"] * 100.0 for r in sensibilidade_results]
    rf_norm = [r["rf_custo_norm"] * 100.0 for r in sensibilidade_results]
    rf_iso_b_norm = [r["rf_iso_bayes_custo"] / r["custo_trivial_teste"] * 100.0 for r in sensibilidade_results]

    ax = axes[0]
    ax.plot(rhos, lr_norm, "o-", color="#1f77b4", lw=2, label="Regressão Logística ($\\theta_{OOF}$)")
    ax.plot(rhos, rf_norm, "s-", color="#2ca02c", lw=2, label="Random Forest ($\\theta_{OOF}$)")
    ax.plot(rhos, rf_iso_b_norm, "^--", color="#9467bd", lw=1.8, label="Random Forest Calibrada ($\\theta^*$ Bayes)")
    ax.set_xscale("log")
    ax.set_xticks(rhos)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Razão de Custos de Erro ($\\rho = c_{FN} / c_{FP}$)")
    ax.set_ylabel("Custo Normalizado (% do Classificador Trivial)")
    ax.set_title("(a) Curva de Sensibilidade de Custo Normalizado")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.5)

    ax2 = axes[1]
    lr_ths = [r["lr_th_oof"] for r in sensibilidade_results]
    rf_ths = [r["rf_th_oof"] for r in sensibilidade_results]
    bayes_ths = [r["th_bayes"] for r in sensibilidade_results]
    ax2.plot(rhos, lr_ths, "o-", color="#1f77b4", lw=2, label="Limiar $\\theta_{OOF}$ - Regressão Logística")
    ax2.plot(rhos, rf_ths, "s-", color="#2ca02c", lw=2, label="Limiar $\\theta_{OOF}$ - Random Forest")
    ax2.plot(rhos, bayes_ths, "k:", lw=1.8, label="Limiar Teórico de Bayes $\\theta^* = 1/(1+\\rho)$")
    ax2.set_xscale("log")
    ax2.set_xticks(rhos)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_xlabel("Razão de Custos de Erro ($\\rho = c_{FN} / c_{FP}$)")
    ax2.set_ylabel("Limiar de Decisão Selecionado ($\\theta$)")
    ax2.set_title("(b) Deslocamento do Limiar Ótimo com a Penalidade $c_{FN}$")
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_sensibilidade_custos.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig_sensibilidade_custos.png"), bbox_inches="tight")
    plt.close(fig)

    # Salva todos os resultados em JSON para consulta exata
    all_output = {
        "versoes": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "sklearn": sklearn.__version__,
        },
        "tempos": {
            "lr_train_s": lr_train_time,
            "lr_infer_ms": lr_infer_time_ms,
            "rf_train_s": rf_train_time,
            "rf_infer_ms": rf_infer_time_ms,
            "total_script_s": round(time.time() - t_global, 2),
        },
        "hiperparametros": {
            "lr_grid": lr_search_results,
            "rf_grid": rf_search_results,
            "best_lr_C": best_lr_C,
            "best_rf_cfg": {k: str(v) for k, v in best_rf_cfg.items()},
        },
        "teste_principal_E1": {
            "lr_default": lr_test_default,
            "lr_opt": lr_test_opt,
            "rf_default": rf_test_default,
            "rf_opt": rf_test_opt,
            "lr_oof_summary": {k: v for k, v in best_lr_oof_summary.items() if k != "curva_custos_oof"},
            "rf_oof_summary": {k: v for k, v in best_rf_oof_summary.items() if k != "curva_custos_oof"},
        },
        "estabilidade_imputacao": imputacao_results,
        "calibracao": calibracao_results,
        "sensibilidade_custos": sensibilidade_results,
    }

    json_path = os.path.join(OUT_DIR, "resultados_etapa3.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_output, f, indent=2, ensure_ascii=False)

    print(f"\nExperimento concluído em {time.time() - t_global:.1f}s! Resultados salvos em {json_path}")


if __name__ == "__main__":
    main()
