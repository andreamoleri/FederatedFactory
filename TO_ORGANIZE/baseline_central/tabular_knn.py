import torch
from sklearn.datasets import load_iris, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. Data Loading (Cached)
# ==========================================
_DATA_CACHE = {}

def load_data_source(dataset_name):
    if dataset_name in _DATA_CACHE:
        return _DATA_CACHE[dataset_name]

    if dataset_name == 'iris':
        d = load_iris()
        X, y = StandardScaler().fit_transform(d.data), d.target
        meta = [X.shape[1]]
    
    elif dataset_name == 'breast_cancer':
        d = load_breast_cancer()
        X, y = StandardScaler().fit_transform(d.data), d.target
        meta = [X.shape[1]]
    
    elif dataset_name == 'titanic':
        X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
        features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
        X = X[features].copy()
        X['age'] = X['age'].fillna(X['age'].median())
        X['fare'] = X['fare'].fillna(X['fare'].median())
        le = LabelEncoder()
        X['sex'] = le.fit_transform(X['sex'].astype(str))
        X = RobustScaler().fit_transform(X)
        y = y.astype(int).values
        meta = [X.shape[1]]
    
    elif dataset_name == 'adult':
        X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
        cat_cols = X.select_dtypes(include=['category', 'object']).columns
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        for col in cat_cols:
            X[col] = X[col].fillna(X[col].mode()[0])
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        for col in num_cols:
            X[col] = X[col].fillna(X[col].median())
        
        X = RobustScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
        meta = [X.shape[1]]
    
    elif dataset_name == 'covertype':
        X, y = fetch_openml("covertype", version=4, as_frame=True, return_X_y=True)
        X = StandardScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
        meta = [X.shape[1]]
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    _DATA_CACHE[dataset_name] = (X, y, meta)
    return X, y, meta

# ==========================================
# 2. Heuristic: Tune k on Training Split
# ==========================================
def tune_k_via_likelihood(X_train_full, dim_features):
    """
    Selects the best k by maximizing the Pseudo-Log-Likelihood on a 
    held-out validation subset of the NORMAL training data.
    
    Likelihood propto: k / (Volume_of_ball_at_dist_k)
    Log-Likelihood propto: log(k) - d * log(dist_k)
    """
    n_samples = len(X_train_full)
    
    # 1. Create a "Internal Validation Set" strictly from Training Data
    #    (Standard 80/20 split)
    split_idx = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    X_train_sub = X_train_full[train_idx]
    X_val_sub = X_train_full[val_idx]
    
    # If split is too small, fallback to sqrt(N) rule immediately
    if len(X_train_sub) < 5 or len(X_val_sub) < 1:
        return int(np.sqrt(n_samples))
    
    # 2. Define Candidate k range
    #    From 3 up to sqrt(N). We step to avoid exhaustive search.
    max_k = int(np.sqrt(n_samples))
    candidates = list(range(3, max_k + 1, max(1, max_k // 10)))
    
    # Ensure a minimum set of candidates
    if 5 not in candidates and 5 < max_k:
        candidates.append(5)
    candidates = sorted(list(set(candidates)))
    
    if not candidates:
        return 5

    best_k = candidates[0]
    best_score = -np.inf

    # 3. Evaluate Candidates
    #    We use a larger pool of neighbors (max_k) once to avoid re-fitting
    nbrs = NearestNeighbors(
        n_neighbors=candidates[-1], algorithm='auto', metric='minkowski', p=2
    )
    nbrs.fit(X_train_sub)
    
    # Get distances to all neighbors up to max_k
    dists_full, _ = nbrs.kneighbors(X_val_sub)
    
    for k in candidates:
        # Distance to the k-th neighbor (column index k-1)
        k_dist = dists_full[:, k-1]
        
        # Avoid log(0)
        k_dist = np.maximum(k_dist, 1e-10)
        
        # Pseudo-Log-Likelihood score
        # We balance the neighbor count (k) vs the volume expansion (dist**d)
        score = np.log(k) - (dim_features * np.mean(np.log(k_dist)))
        
        if score > best_score:
            best_score = score
            best_k = k
            
    return best_k

# ==========================================
# 3. Centralized K-NN Execution
# ==========================================
def run_single_seed_knn(seed, dataset_name, normal_class):
    np.random.seed(seed)
    
    # 1. Load Data
    X_full, y_full, meta = load_data_source(dataset_name)
    n_features = meta[0]
    
    # 2. Split (Identical to FedAvg Baseline)
    X_train_full, X_rest, y_train_full, y_rest = train_test_split(
        X_full, y_full, test_size=0.4, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed
    )
    
    # 3. Filter for One-Class Training (Only Normal data)
    train_mask = (y_train_full == normal_class)
    X_train_normal = X_train_full[train_mask]
    
    if len(X_train_normal) < 5:
        return 0.5, 0.0, 0.0, 0.0, 0.0, 5

    # ==============================================================
    # 4. TUNING k (Using Validation Set Derived from Training Data)
    # ==============================================================
    # We strictly use X_train_normal to pick k. 
    # X_val (from X_rest) is NOT used for finding k.
    
    best_k = tune_k_via_likelihood(X_train_normal, n_features)

    # 5. Final Fit on Full Normal Training Data
    nbrs = NearestNeighbors(
        n_neighbors=best_k, algorithm='auto', metric='minkowski', p=2
    )
    nbrs.fit(X_train_normal)
    
    # 6. Inference Function
    def get_anomaly_scores(X_eval):
        distances, _ = nbrs.kneighbors(X_eval)
        # Score is distance to the k-th nearest neighbor
        return distances[:, -1]

    val_scores = get_anomaly_scores(X_val)
    test_scores = get_anomaly_scores(X_test)
    
    # 7. Thresholding
    # We use X_val (containing anomalies) ONLY to set the decision threshold
    # (AUC/F1 optimization). This is standard transductive evaluation.
    y_val_binary = np.where(y_val == normal_class, 0, 1)
    y_test_binary = np.where(y_test == normal_class, 0, 1)
    
    if len(np.unique(y_val_binary)) > 1:
        fpr, tpr, thresholds = roc_curve(y_val_binary, val_scores)
        best_thresh = thresholds[np.argmax(tpr - fpr)]
    else:
        best_thresh = np.percentile(val_scores, 95)
        
    preds = (test_scores > best_thresh).astype(int)
    
    # 8. Metrics
    try:
        auc = roc_auc_score(y_test_binary, test_scores)
    except:
        auc = 0.5
        
    return (
        auc, 
        accuracy_score(y_test_binary, preds),
        precision_score(y_test_binary, preds, zero_division=0),
        recall_score(y_test_binary, preds, zero_division=0),
        f1_score(y_test_binary, preds, zero_division=0),
        best_k
    )

# ==========================================
# 4. Main Execution
# ==========================================
def evaluate_knn_baseline():
    seeds = [1, 6, 31, 42, 53]
    
    datasets = [
        ('iris', range(3)),
        ('breast_cancer', range(2)),
        ('titanic', range(2)),
        ('adult', range(2)),
        ('covertype', range(7))
    ]
    
    print(f"\n\n{'='*50}")
    print(f"STARTING EXPERIMENT: CENTRALIZED K-NN (Adaptive k)")
    print(f"Heuristic: Maximum Likelihood on Internal Train Split")
    print(f"{'='*50}")
    
    # Pre-load data
    for d_name, _ in datasets:
        load_data_source(d_name)

    all_results = {name: [] for name, _ in datasets}
    
    total_tasks = sum(len(c) for _, c in datasets) * len(seeds)
    pbar = tqdm(total=total_tasks, desc="Running K-NN")

    for d_name, d_classes in datasets:
        
        class_metrics = {'auc': [], 'acc': [], 'pre': [], 'rec': [], 'f1': [], 'k': []}

        for cls in d_classes:
            metrics_per_seed = []
            for seed in seeds:
                res = run_single_seed_knn(seed, d_name, cls)
                metrics_per_seed.append(res)
                pbar.update(1)
            
            aucs = [r[0] for r in metrics_per_seed]
            accs = [r[1] for r in metrics_per_seed]
            pres = [r[2] for r in metrics_per_seed]
            recs = [r[3] for r in metrics_per_seed]
            f1s = [r[4] for r in metrics_per_seed]
            ks   = [r[5] for r in metrics_per_seed]

            class_metrics['auc'].append(np.mean(aucs))
            class_metrics['acc'].append(np.mean(accs))
            class_metrics['pre'].append(np.mean(pres))
            class_metrics['rec'].append(np.mean(recs))
            class_metrics['f1'].append(np.mean(f1s))
            class_metrics['k'].append(np.mean(ks))
        
        def fmt(arr):
            return f"{np.mean(arr):.4f} ± {np.std(arr):.4f}"
        
        all_results[d_name].append({
            "Method": "Adaptive K-NN",
            "Avg k": f"{np.mean(class_metrics['k']):.1f}",
            "AUROC": fmt(class_metrics['auc']),
            "Accuracy": fmt(class_metrics['acc']),
            "Precision": fmt(class_metrics['pre']),
            "Recall": fmt(class_metrics['rec']),
            "F1-Score": fmt(class_metrics['f1'])
        })

    pbar.close()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    for d_name, _ in datasets:
        print(f"\n--- {d_name.upper()} RESULTS (Adaptive K-NN) ---")
        if all_results[d_name]:
            df = pd.DataFrame(all_results[d_name])
            print(df.to_string(index=False))
            df.to_csv(f"{d_name}_knn_adaptive.csv", index=False)

if __name__ == "__main__":
    evaluate_knn_baseline()
