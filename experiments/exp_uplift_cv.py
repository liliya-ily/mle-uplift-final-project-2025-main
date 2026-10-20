"""Сравнение uplift-подходов повторной стратифицированной кросс-валидацией.

Шаг 0: воспроизводит результаты ноутбука (разбиение 80/20, random_state=42) для проверки окружения.
Шаг 1: для каждой модели считает Uplift@30, Uplift AUC и Qini AUC на 15 отложенных частях
       (5 блоков x 3 повтора, стратификация по treatment и target) и сравнивает с S-learner.

Запуск из корня репозитория (подробнее в experiments/README.md):
    python experiments/exp_uplift_cv.py
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from causalml.inference.meta import BaseSClassifier, BaseTClassifier, BaseXClassifier
from causalml.inference.tree import UpliftRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklift.metrics import qini_auc_score, uplift_at_k, uplift_auc_score

warnings.filterwarnings("ignore")
FEATURES = ["recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel"]
RF = dict(n_estimators=100, max_depth=6, min_samples_leaf=100, random_state=42, n_jobs=-1)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def metrics(y, uplift, treatment):
    uplift = np.asarray(uplift).ravel()
    return {
        "uplift@30": float(uplift_at_k(y, uplift, treatment, strategy="overall", k=0.3)),
        "uplift_auc": float(uplift_auc_score(y, uplift, treatment)),
        "qini_auc": float(qini_auc_score(y, uplift, treatment)),
    }


# ---- модели: каждая принимает X_train, t_train, y_train, X_test и возвращает uplift для X_test
def fit_predict_uplift_rf(Xtr, ttr, ytr, Xte, **params):
    model = UpliftRandomForestClassifier(control_name="control", random_state=42, n_jobs=-1, **params)
    model.fit(Xtr.copy(), treatment=np.where(ttr == 1, "treatment", "control"), y=ytr)
    return model.predict(Xte.copy())


def fit_predict_s(Xtr, ttr, ytr, Xte, learner):
    model = BaseSClassifier(learner=learner, control_name=0)
    model.fit(Xtr.copy(), treatment=ttr, y=ytr)
    return model.predict(Xte.copy())


def fit_predict_t(Xtr, ttr, ytr, Xte, learner):
    model = BaseTClassifier(learner=learner, control_name=0)
    model.fit(Xtr.copy(), treatment=ttr, y=ytr)
    return model.predict(Xte.copy())


def fit_predict_x(Xtr, ttr, ytr, Xte):
    # эксперимент рандомизирован, поэтому propensity score одинаков для всех: доля воздействия в train
    p_train = np.full(len(Xtr), ttr.mean())
    p_test = np.full(len(Xte), ttr.mean())
    model = BaseXClassifier(
        outcome_learner=RandomForestClassifier(**RF),
        effect_learner=RandomForestRegressor(**RF),
        control_name=0,
    )
    model.fit(Xtr.copy(), treatment=ttr, y=ytr, p=p_train)
    return model.predict(Xte.copy(), p=p_test)


def catboost():
    return CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, random_seed=42, verbose=0, thread_count=-1)


MODELS = {
    "UpliftRF (как в ноутбуке)": lambda a, b, c, d: fit_predict_uplift_rf(a, b, c, d, n_estimators=100, max_depth=6, min_samples_leaf=100),
    "UpliftRF ED, depth 6": lambda a, b, c, d: fit_predict_uplift_rf(a, b, c, d, evaluationFunction="ED", n_estimators=50, max_depth=6, min_samples_leaf=100),
    "UpliftRF KL, depth 7": lambda a, b, c, d: fit_predict_uplift_rf(a, b, c, d, evaluationFunction="KL", n_estimators=50, max_depth=7, min_samples_leaf=100),
    "S-learner RF": lambda a, b, c, d: fit_predict_s(a, b, c, d, RandomForestClassifier(**RF)),
    "T-learner RF": lambda a, b, c, d: fit_predict_t(a, b, c, d, RandomForestClassifier(**RF)),
    "X-learner RF": fit_predict_x,
    "S-learner CatBoost": lambda a, b, c, d: fit_predict_s(a, b, c, d, catboost()),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="uplift_fp_data.csv")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", default="", help="через запятую: подстроки названий моделей")
    ap.add_argument("--skip-repro", action="store_true")
    ap.add_argument("--out", default="experiments/results/uplift_cv.json")
    args = ap.parse_args()

    # те же преобразования, что в ноутбуке: удаление полностью совпадающих строк
    data = pd.read_csv(args.data).drop_duplicates().reset_index(drop=True)
    log(f"данные: {data.shape}")
    X = data[FEATURES].to_numpy()
    t = data["treatment"].to_numpy()
    y = data["target"].to_numpy()

    if not args.skip_repro:
        # шаг 0: то же разбиение и те же настройки, что в ноутбуке (в X остаётся и колонка treatment, как там)
        Xtr, Xte, ytr, yte = train_test_split(
            data.drop(columns=["target"]), data["target"], test_size=0.2,
            stratify=data[["target", "treatment"]], random_state=42)
        ttr, tte = Xtr["treatment"].to_numpy(), Xte["treatment"].to_numpy()
        up = fit_predict_uplift_rf(Xtr[FEATURES].to_numpy(), ttr, ytr.to_numpy(), Xte[FEATURES].to_numpy(),
                                   n_estimators=100, max_depth=6, min_samples_leaf=100)
        log(f"воспроизведение UpliftRF: {metrics(yte.to_numpy(), up, tte)['uplift@30']:.4f} (в ноутбуке 0,0494)")
        up = fit_predict_s(Xtr.to_numpy(), ttr, ytr.to_numpy(), Xte.to_numpy(), RandomForestClassifier(**RF))
        log(f"воспроизведение S-learner: {metrics(yte.to_numpy(), up, tte)['uplift@30']:.4f} (в ноутбуке 0,0518)")

    names = [n for n in MODELS if not args.only or any(s.lower() in n.lower() for s in args.only.split(","))]
    strat = t * 2 + y
    cv = RepeatedStratifiedKFold(n_splits=args.splits, n_repeats=args.repeats, random_state=42)
    folds = list(cv.split(X, strat))
    log(f"моделей: {len(names)}, разбиений: {len(folds)}")

    scores = {n: [] for n in names}
    for fold_id, (tr, te) in enumerate(folds, 1):
        for n in names:
            t0 = time.time()
            up = MODELS[n](X[tr], t[tr], y[tr], X[te])
            m = metrics(y[te], up, t[te])
            scores[n].append(m)
            log(f"разбиение {fold_id}/{len(folds)} | {n:28} U@30={m['uplift@30']:.4f} ({time.time() - t0:.0f} с)")

    base = "S-learner RF" if "S-learner RF" in names else names[0]
    base_vals = np.array([m["uplift@30"] for m in scores[base]])
    rows = []
    for n in names:
        v = np.array([m["uplift@30"] for m in scores[n]])
        d = v - base_vals
        rows.append({
            "model": n,
            "uplift@30 mean": v.mean(), "std": v.std(ddof=1),
            "uplift_auc": np.mean([m["uplift_auc"] for m in scores[n]]),
            "qini_auc": np.mean([m["qini_auc"] for m in scores[n]]),
            f"diff vs {base}": d.mean(), "diff se": d.std(ddof=1) / np.sqrt(len(d)),
            "better in folds": float((d > 0).mean()),
        })
    table = pd.DataFrame(rows).sort_values("uplift@30 mean", ascending=False)
    print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"args": vars(args), "rows": rows, "scores": scores}, open(args.out, "w"), indent=1, ensure_ascii=False)
    log(f"сохранено: {args.out}")


if __name__ == "__main__":
    main()
