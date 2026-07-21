import torch
import torch.nn as nn
import glob
import os
import time
import pickle
import inspect
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from pathlib import Path

from utils import functions as uf
from utils.model import DynamicMLP

folder_path = './data/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #without cuda it was 21.6 now its 14.2


csv_files = glob.glob(os.path.join(folder_path, '*c000.csv'))
df_all = pd.concat((pd.read_csv(file, sep=";") for file in csv_files), ignore_index=True)

forget_df = pd.read_csv(os.path.join(folder_path, 'forget_data.csv'), sep=",")

random_seed = 42
id_col = "user_id"

#train/validation split for fine tuning

forget_ids = set(forget_df[id_col])
clean_df = df_all[~df_all[id_col].isin(forget_ids)].reset_index(drop=True)

ids = clean_df[id_col].unique()
rng = np.random.default_rng(random_seed)
rng.shuffle(ids)
val_frac = 0.15
n_val = int(len(ids) * val_frac)
val_ids = set(ids[:n_val])

val_df = clean_df[clean_df[id_col].isin(val_ids)].reset_index(drop=True)
train_df = clean_df[~clean_df[id_col].isin(val_ids)].reset_index(drop=True)


X_train, y_train, feature_cols, target_cols = uf.prepare_data(train_df, id_col=id_col, target_prefix='target__')
X_val, y_val, _, _ = uf.prepare_data(val_df, id_col=id_col, target_prefix='target__')
X_forget, y_forget, _, _ = uf.prepare_data(forget_df, id_col=id_col, target_prefix='target__')

imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train).astype(np.float32)
X_val = imputer.transform(X_val).astype(np.float32)
X_forget = imputer.transform(X_forget).astype(np.float32)


pos_counts = np.sum(y_train, axis=0)
neg_counts = len(y_train) - pos_counts
pos_weights = torch.tensor(neg_counts / (pos_counts + 1e-6), device=device, dtype=torch.float32)
pos_weights = pos_weights.clamp(min=0.1, max=100.0)
print(f"pos_weights: {pos_weights}")


artifact_path = Path('data') / 'model_artifact'

payload = uf.load_pickle(artifact_path)

state_dict = payload['state_dict']
architecture = payload['architecture']
best_params = payload['best_hyperparameters']
model_class_source = payload['model_class_source']

print("\n--- Saved Metadata ---")
print("Architecture parameters:", architecture)
print("Best Hyperparameters:", best_params)

try:
    model = DynamicMLP(
        input_dim=architecture['input_dim'],
        hidden_layers=architecture['hidden_layers'],
        num_outputs=architecture['num_outputs']
    )
except NameError:
    print("DynamicMLP class was not found. Check if the class source compiled correctly.")
    raise

model.load_state_dict(state_dict)
model.to(device)

print("\nModel successfully reconstructed and weights loaded.")

#main logic of unlearning (certified removal via influence functions, Guo et al. 2020)

X_train_t = torch.tensor(X_train, device=device)
y_train_t = torch.tensor(y_train.astype(np.float32), device=device)
X_forget_t = torch.tensor(X_forget, device=device)
y_forget_t = torch.tensor(y_forget.astype(np.float32), device=device)

params = [p for p in model.parameters() if p.requires_grad]
param_shapes = [p.shape for p in params]
param_numel = [p.numel() for p in params]

def flatten(grads):
    return torch.cat([g.reshape(-1) for g in grads])

def unflatten(vector):
    chunks = []
    idx = 0
    for shape, numel in zip(param_shapes, param_numel):
        chunks.append(vector[idx:idx + numel].view(shape))
        idx += numel
    return chunks

def hvp(loss, vector):
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_dot_vector = torch.sum(flatten(grads) * vector)
    hv = torch.autograd.grad(grad_dot_vector, params, retain_graph=False)
    return flatten(hv)

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

lissa_iterations = 50
lissa_batch_size = 256
damping = 0.1
scale = 100.0

model.eval()
n = X_train_t.shape[0]
start_time = time.time()            #Timer start point of unlearning process not script

forget_logits = model(X_forget_t)
forget_loss = loss_fn(forget_logits, y_forget_t)
forget_grad = torch.autograd.grad(forget_loss, params)
v = flatten(forget_grad).detach()
v_norm = v.norm()

hinv_v = v.clone()
for i in range(lissa_iterations):
    idx = torch.randint(0, n, (lissa_batch_size,), device=device)
    xb, yb = X_train_t[idx], y_train_t[idx]
    logits = model(xb)
    loss = loss_fn(logits, yb)
    hv = hvp(loss, hinv_v)
    hinv_v = v + hinv_v - (hv + damping * hinv_v) / scale
    hinv_v_norm = hinv_v.norm()
    if not torch.isfinite(hinv_v_norm):
        print(f"lissa iteration {i + 1}/{lissa_iterations} diverged, stopping early")
        hinv_v = v.clone()
        break
    if hinv_v_norm > 10 * v_norm:
        hinv_v = hinv_v * (10 * v_norm / hinv_v_norm)
    if (i + 1) % 10 == 0:
        print(f"lissa iteration {i + 1}/{lissa_iterations} - hinv_v norm: {hinv_v_norm:.4f}")
hinv_v = (hinv_v / scale).detach()

with torch.no_grad():
    for p, u in zip(params, unflatten(hinv_v)):
        p -= u

execution_time = time.time() - start_time  #Timer endpoint and total time for unlearning

print(f"\nInfluence-based unlearning finished in {execution_time:.1f}s")

#P@10 metric calculation
X_val_t = torch.tensor(X_val, device=device)
y_val_t = torch.tensor(y_val.astype(np.float32), device=device)

k = 10
with torch.no_grad():
    val_logits = model(X_val_t)
    val_probs = torch.sigmoid(val_logits)
    topk_idx = torch.topk(val_probs, k=k, dim=1).indices
    hits = torch.gather(y_val_t, 1, topk_idx)
    precision_val = hits.sum(dim=1).div(k).mean().item()

print(f"\nprecision_val (P@{k}): {precision_val:.4f}")


# MIA resistance metric calculation
from sklearn.metrics import roc_auc_score

print("\nEvaluating MIA Resistance ")

# Prepare Forget Data
X_forget, y_forget, _, _ = uf.prepare_data(forget_df, id_col=id_col, target_prefix='target__')
X_forget = imputer.transform(X_forget).astype(np.float32)

X_forget_t = torch.tensor(X_forget, device=device)
y_forget_t = torch.tensor(y_forget.astype(np.float32), device=device)

# Get Model Losses on Forget (Members) vs Validation (Non-Members)
with torch.no_grad():
    logits_forget = model(X_forget_t)
    logits_val = model(X_val_t)

    # Compute element-wise Binary Cross-Entropy Loss per sample
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='none')
    loss_forget = bce(logits_forget, y_forget_t).mean(dim=1).cpu().numpy()
    loss_val = bce(logits_val, y_val_t).mean(dim=1).cpu().numpy()

# Label: 1 for Forget Set (Targeted Members), 0 for Unseen Val Set
# Lower loss indicates higher confidence (member likelihood)
y_true = np.concatenate([np.ones(len(loss_forget)), np.zeros(len(loss_val))])
scores = np.concatenate([-loss_forget, -loss_val])  # Negative loss: lower loss = higher score

# Calculate MIA AUC
mia_auc = roc_auc_score(y_true, scores)

# Calculate Official Competition MIA Resistance Metric
mia_resistance = 1.0 - 2.0 * abs(mia_auc - 0.5)

print(f"MIA AUC Score:          {mia_auc:.4f} (Ideal = 0.5000)")
print(f"MIA Resistance Score:   {mia_resistance:.4f} (Ideal = 1.0000)")

# estimated total score
estimated_score = 0.45 * precision_val + 0.45 * mia_resistance
print(f"\n--- SCORE SUMMARY ---")
print(f"Precision@10 Score (45%):   {precision_val:.4f}")
print(f"MIA Resistance Score (45%): {mia_resistance:.4f}")
print(f"Combined Quality Score:     {estimated_score:.4f}")

#new folder with 3 files - exec time, model, validation set
group_name = "G20_V7"
out_dir = Path(group_name)
out_dir.mkdir(exist_ok=True)

(out_dir / "execution_time.txt").write_text(str(int(round(execution_time))))

model_class_src = inspect.getsource(DynamicMLP)
artifact = {
    "state_dict": model.state_dict(),
    "architecture": architecture,
    "best_hyperparameters": best_params,
    "model_class_source": model_class_src,
}
with open(out_dir / "model_artifact", "wb") as f:
    pickle.dump(artifact, f)

pd.DataFrame({"user_id": val_df[id_col].unique()}).to_csv(out_dir / "validation_ids.csv", index=False)

print(f"\nSubmission written to ./{out_dir}/")