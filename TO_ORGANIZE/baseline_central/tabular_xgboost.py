import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_iris, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

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
    elif dataset_name == 'breast_cancer':
        d = load_breast_cancer()
        X, y = StandardScaler().fit_transform(d.data), d.target
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
    elif dataset_name == 'covertype':
        X, y = fetch_openml("covertype", version=4, as_frame=True, return_X_y=True)
        X = StandardScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    meta = [X.shape[1]]
    _DATA_CACHE[dataset_name] = (X, y, meta)
    return X, y, meta

# ==========================================
# 2. Federated Splitting Logic
# ==========================================
def get_client_splits_indices(total_samples, num_clients, method, alpha):
    actual_clients = min(num_clients, total_samples)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    if method == 'iid' or total_samples < actual_clients:
        splits = np.array_split(indices, actual_clients)
    else:
        alpha = float(alpha)
        eps = 1e-12
        props = np.random.dirichlet(np.full(actual_clients, max(alpha, eps)))
        props = np.clip(props, eps, None)
        props = props / props.sum()

        if total_samples >= actual_clients:
            counts = np.ones(actual_clients, dtype=np.int64)
            remaining = total_samples - actual_clients
            extra = np.random.multinomial(remaining, props)
            counts += extra
        else:
            counts = np.zeros(actual_clients, dtype=np.int64)
            counts[:total_samples] = 1

        split_pts = np.cumsum(counts)[:-1]
        splits = np.split(indices, split_pts)
    
    return splits

# ==========================================
# 3. Model Definitions
# ==========================================

class LocalXGBoostAgent:
    def __init__(self, X_local):
        self.X_local = X_local
        self.model = None

    def fit(self):
        n_samples, n_features = self.X_local.shape
        if n_samples < 5: return

        # 1. Positive Class: Raw Data
        X_pos = self.X_local
        n_pos = len(X_pos)

        # 2. Negative Class: Synthetic Noise
        X_marginal = self.X_local.copy()
        for i in range(n_features):
            np.random.shuffle(X_marginal[:, i])
            
        mins = self.X_local.min(axis=0)
        maxs = self.X_local.max(axis=0)
        ranges = maxs - mins
        X_uniform = np.random.uniform(
            low=mins - (ranges * 0.1), 
            high=maxs + (ranges * 0.1), 
            size=(n_pos, n_features)
        )

        n_marg = n_pos // 2
        n_uni = n_pos - n_marg
        X_neg = np.vstack([X_marginal[:n_marg], X_uniform[:n_uni]])

        X_train = np.vstack([X_pos, X_neg])
        y_train = np.hstack([np.zeros(n_pos), np.ones(n_pos)]) 

        self.model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            objective='binary:logistic', n_jobs=1,
            tree_method='hist', eval_metric='logloss',
            subsample=0.8, colsample_bytree=0.8,
            random_state=42
        )
        self.model.fit(X_train, y_train)

    def predict_anomaly_score(self, X):
        if self.model is None: return np.full(X.shape[0], 0.5)
        return self.model.predict_proba(X)[:, 1]

class LocalIsoForestAgent:
    def __init__(self, X_local):
        self.X_local = X_local
        self.model = None

    def fit(self):
        if len(self.X_local) < 5: return
        self.model = IsolationForest(
            n_estimators=100, contamination='auto', 
            n_jobs=1, random_state=42
        )
        self.model.fit(self.X_local)

    def predict_anomaly_score(self, X):
        if self.model is None: return np.full(X.shape[0], 0.0)
        # Decision function: lower is more anomalous. 
        # We negate it so higher = more anomalous.
        return -self.model.decision_function(X)

# ==========================================
# 4. Unified Runner (Enhanced)
# ==========================================
# ==========================================
# 4. Unified Runner (Enhanced)
# ==========================================
def run_single_seed(seed, dataset_name, normal_class, split_method, alpha, num_clients, model_type):
    np.random.seed(seed)
    
    # 1. Load & Split
    X_full, y_full, _ = load_data_source(dataset_name)
    X_train, X_rest, y_train, y_rest = train_test_split(X_full, y_full, test_size=0.4, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=seed)

    # 2. Filter Normal Class
    train_mask = (y_train == normal_class)
    X_train_normal = X_train[train_mask]
    
    if len(X_train_normal) == 0:
        return {} # Empty result

    # 3. Create Clients
    splits = get_client_splits_indices(len(X_train_normal), num_clients, split_method, alpha)
    
    agents = []
    for idxs in splits:
        if len(idxs) > 0:
            X_local = X_train_normal[idxs]
            if model_type == 'xgb':
                agent = LocalXGBoostAgent(X_local)
            elif model_type == 'iforest':
                agent = LocalIsoForestAgent(X_local)
            agents.append(agent)
    
    # 4. Train
    valid_agents = []
    for agent in agents:
        agent.fit()
        if agent.model is not None:
            valid_agents.append(agent)
            
    if not valid_agents:
        return {}

    # 5. Helper: Calculate Metrics
    def calc_metrics(test_scores, val_scores, y_test_bin, y_val_bin):
        if len(np.unique(y_val_bin)) > 1:
            fpr, tpr, thresholds = roc_curve(y_val_bin, val_scores)
            best_thresh = thresholds[np.argmax(tpr - fpr)]
        else:
            best_thresh = np.percentile(val_scores, 95)

        preds = (test_scores > best_thresh).astype(int)
        
        try:
            auc = roc_auc_score(y_test_bin, test_scores)
        except:
            auc = 0.5
            
        return {
            'auc': auc,
            'acc': accuracy_score(y_test_bin, preds),
            'pre': precision_score(y_test_bin, preds, zero_division=0),
            'rec': recall_score(y_test_bin, preds, zero_division=0),
            'f1': f1_score(y_test_bin, preds, zero_division=0)
        }

    # 6. Generate Raw Scores from all Agents (THE FIX)
    val_preds_raw = []
    test_preds_raw = []
    
    for agent in valid_agents:
        s_val = agent.predict_anomaly_score(X_val)
        s_test = agent.predict_anomaly_score(X_test)
        
        if model_type == 'iforest':
            # --- CORRECTION START ---
            # 1. Fit scaler ONLY on Validation Data (No Leakage)
            min_val = s_val.min()
            max_val = s_val.max()
            denom = max_val - min_val + 1e-10  # Safety epsilon
            
            # 2. Transform Validation Data
            s_val = (s_val - min_val) / denom
            
            # 3. Transform Test Data using VALIDATION Params
            s_test = (s_test - min_val) / denom
            
            # 4. CLIP Test Data to [0, 1]
            # This handles cases where test outliers are more extreme than validation outliers.
            # Without this, you get values > 1.0 or < 0.0, which breaks PoE.
            s_test = np.clip(s_test, 0.0, 1.0)
            # --- CORRECTION END ---
        
        val_preds_raw.append(s_val)
        test_preds_raw.append(s_test)

    val_stack = np.stack(val_preds_raw)
    test_stack = np.stack(test_preds_raw)
    
    # Clip for PoE stability (log(0) avoidance)
    val_stack = np.clip(val_stack, 1e-9, 1.0 - 1e-9)
    test_stack = np.clip(test_stack, 1e-9, 1.0 - 1e-9)
    
    y_val_bin = (y_val != normal_class).astype(int)
    y_test_bin = (y_test != normal_class).astype(int)

    # 7. Aggregate
    results_per_agg = {}
    
    for method in ['min', 'mean', 'max', 'poe']:
        if method == 'min':
            v_agg = np.min(val_stack, axis=0)
            t_agg = np.min(test_stack, axis=0)
        elif method == 'mean':
            v_agg = np.mean(val_stack, axis=0)
            t_agg = np.mean(test_stack, axis=0)
        elif method == 'max':
            v_agg = np.max(val_stack, axis=0)
            t_agg = np.max(test_stack, axis=0)
        elif method == 'poe':
            v_agg = np.sum(np.log(val_stack), axis=0)
            t_agg = np.sum(np.log(test_stack), axis=0)
            
        results_per_agg[method] = calc_metrics(t_agg, v_agg, y_test_bin, y_val_bin)

    return results_per_agg

# ==========================================
# 5. Main Execution Loop
# ==========================================
def evaluate_scenarios():
    # --- CONFIG ---
    NUM_CLIENTS = 3 
    models_to_run = ['xgb', 'iforest']
    # --------------

    seeds = [1, 6, 31, 42, 53]
    scenarios = [
        ('IID', 'iid', 0.0),
        ('Dirichlet (0.5)', 'dirichlet', 0.5),
        ('Dirichlet (0.1)', 'dirichlet', 0.1),
        ('Dirichlet (0.01)', 'dirichlet', 0.01),
    ]
    
    datasets = [
        ('iris', range(3)),
        ('breast_cancer', range(2)),
        ('titanic', range(2)),
        ('adult', range(2)),
        ('covertype', range(7))
    ]

    print(f"Running Federated Baselines (Requested Clients={NUM_CLIENTS})")
    
    for d_name, _ in datasets:
        load_data_source(d_name)

    final_rows = []
    
    total_steps = len(models_to_run) * len(scenarios) * sum(len(c) for _, c in datasets) * len(seeds)
    pbar = tqdm(total=total_steps, desc="Evaluation")

    for model_type in models_to_run:
        for s_name, s_method, s_alpha in scenarios:
            for d_name, d_classes in datasets:
                
                # Temporary storage for this Dataset/Scenario/Model combination
                # --- ADDED 'poe' HERE ---
                agg_storage = {
                    'min': {'auc':[], 'acc':[], 'pre':[], 'rec':[], 'f1':[]},
                    'mean': {'auc':[], 'acc':[], 'pre':[], 'rec':[], 'f1':[]},
                    'max': {'auc':[], 'acc':[], 'pre':[], 'rec':[], 'f1':[]},
                    'poe': {'auc':[], 'acc':[], 'pre':[], 'rec':[], 'f1':[]}
                }
                
                for cls in d_classes:
                    for seed in seeds:
                        res_dict = run_single_seed(seed, d_name, cls, s_method, s_alpha, NUM_CLIENTS, model_type)
                        pbar.update(1)
                        
                        if not res_dict: continue

                        # --- ADDED 'poe' HERE ---
                        for agg in ['min', 'mean', 'max', 'poe']:
                            for metric in ['auc', 'acc', 'pre', 'rec', 'f1']:
                                agg_storage[agg][metric].append(res_dict[agg][metric])
                
                # Aggregate and create rows
                # --- ADDED 'poe' HERE ---
                for agg in ['min', 'mean', 'max', 'poe']:
                    row = {
                        "Model": "XGB-Synthetic" if model_type == 'xgb' else "IsoForest",
                        "Dataset": d_name,
                        "Scenario": s_name,
                        "Agg_Method": agg.upper(),
                    }
                    
                    # Compute Mean +/- Std for all metrics
                    for metric in ['auc', 'acc', 'pre', 'rec', 'f1']:
                        vals = agg_storage[agg][metric]
                        if len(vals) > 0:
                            row[metric.upper()] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
                        else:
                            row[metric.upper()] = "N/A"
                            
                    final_rows.append(row)

    pbar.close()
    
    df = pd.DataFrame(final_rows)
    print("\n\n" + "="*80)
    print("FINAL RESULTS TABLE")
    print("="*80)
    
    # Sort for easier reading
    df_sorted = df.sort_values(by=["Dataset", "Model", "Scenario", "Agg_Method"])
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_sorted.to_string(index=False))
    
    df_sorted.to_csv("federated_baselines_full_results.csv", index=False)

if __name__ == "__main__":
    evaluate_scenarios()
