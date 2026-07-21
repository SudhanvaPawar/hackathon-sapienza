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

imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train).astype(np.float32)
X_val = imputer.transform(X_val).astype(np.float32)


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

#main logic of unlearning

X_train_t = torch.tensor(X_train, device=device)
y_train_t = torch.tensor(y_train.astype(np.float32), device=device)

finetune_epochs = 5
finetune_lr = 1e-4
batch_size = 256

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

n = X_train_t.shape[0]
start_time = time.time()            #Timer start point of unlearning process not script
for epoch in range(finetune_epochs):
    perm = torch.randperm(n, device=device)
    epoch_loss = 0.0
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        if len(idx) < 2:
            continue
        xb, yb = X_train_t[idx], y_train_t[idx]
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(idx)
    print(f"epoch {epoch + 1}/{finetune_epochs} - loss: {epoch_loss / n:.4f}")
execution_time = time.time() - start_time  #Timer endpoint and total time for unlearning
model.eval()

print(f"\nFine-tuning finished in {execution_time:.1f}s")

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

#change this for making new folder with 3 files - exec time, model, validation set
group_name = "G20_V1_submission_test"
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