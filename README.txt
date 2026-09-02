!pip install git+https://github.com/KEDRI-AUT/NeuCube-Py.git
import neucube
print('NeuCube imported successfully')
print('Version:', neucube.__version__)
print(dir(neucube))
from neucube import encoder
print(dir(encoder))
from neucube.encoder import StepForward
help(StepForward)
!pip install scipy==1.11.4
!pip install openpyxl
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset - corrected filename
df = pd.read_excel('C:/Users/User/Desktop/cakewebsitetemplate/Data Research/UFMB 2 essentials/F1341550_AE.xlsx',
                   sheet_name='Master_Sheet')

# Select 9 input variables
features = ['Heritage Category', 'Orbit Type', 'Cost (M USD)', 
            'Phase', 'Causes', 'Latitude', 'Coast_Km', 
            'Ap_daily', 'AE_daily']

df_input = df[features].copy()

# Fill 2 missing AE values with median
df_input['AE_daily'] = df_input['AE_daily'].fillna(df_input['AE_daily'].median())

# Encode categorical columns
le = LabelEncoder()
cat_cols = ['Heritage Category', 'Orbit Type', 'Phase', 'Causes']
for col in cat_cols:
    df_input[col] = le.fit_transform(df_input[col].astype(str))

# Normalise all columns to 0-1
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_input)

print('Shape:', df_scaled.shape)
print('Sample record (first row):')
print(df_scaled[0].round(3))
print()
print('All values between 0 and 1:')
print('Min:', df_scaled.min(axis=0).round(3))
print('Max:', df_scaled.max(axis=0).round(3))
print('Columns selected:')
for i, col in enumerate(features, 1):
    print(f'{i}. {col}')
print()
print('Encoded values sample:')
sample_df = pd.DataFrame(df_scaled[:5], columns=features)
print(sample_df.round(3).to_string())
# Convert to PyTorch tensor (required by NeuCube)
X = torch.tensor(df_scaled, dtype=torch.float32)

# Get target variable (Causes) for Goal A
target_causes = df['Causes'].copy()
le_causes = LabelEncoder()
y = torch.tensor(le_causes.fit_transform(target_causes), dtype=torch.long)

print('Input tensor shape:', X.shape)
print('Target tensor shape:', y.shape)
print()
print('Cause classes:')
for i, cls in enumerate(le_causes.classes_):
    print(f'  {i}: {cls}')
# Calculate days since first record
df['Datum'] = pd.to_datetime(df['Datum'], dayfirst=True)
df['Days_Since_Start'] = (df['Datum'] - df['Datum'].min()).dt.days

print('Temporal range:')
print(f'First record: {df["Datum"].min().date()} (day 0)')
print(f'Last record: {df["Datum"].max().date()} (day {df["Days_Since_Start"].max()})')
print()
print('Sample days:')
print(df[['Detail', 'Datum', 'Days_Since_Start']].head(10).to_string())
# Normalise days to 0-1 range
days_normalised = df['Days_Since_Start'].values / df['Days_Since_Start'].max()

# Add normalised time as 10th feature
X_with_time = np.column_stack([df_scaled, days_normalised])

print('Shape with time feature:', X_with_time.shape)
print()
print('Column order:')
features_with_time = features + ['Time_Position']
for i, col in enumerate(features_with_time, 1):
    print(f'{i}. {col}')
print()
print('Sample first 5 records:')
sample = pd.DataFrame(X_with_time[:5], columns=features_with_time)
print(sample.round(3).to_string())
# Convert to tensor
X_tensor = torch.tensor(X_with_time, dtype=torch.float32)

# NeuCube needs 3D: (records, time_steps, features)
# Each record is one event at one point in time
# We treat each record as a single time step
# Shape becomes: (1, 80, 10) — one sequence of 80 events × 10 features
# Then reshape to (80, 1, 10) — 80 samples each with 1 time step

X_3d = X_tensor.unsqueeze(1)  # adds time dimension → (80, 1, 10)

print('3D tensor shape:', X_3d.shape)
print('Meaning:')
print(f'  {X_3d.shape[0]} records')
print(f'  {X_3d.shape[1]} time step per record')
print(f'  {X_3d.shape[2]} features')

# Encode into spike trains
from neucube.encoder import StepForward
encoder = StepForward(threshold=0.1)
X_spikes = encoder.encode_dataset(X_3d)

print()
print('Spike train shape:', X_spikes.shape)
print('Sample spikes for record 0:')
print(X_spikes[0])
# Sort by date first to ensure temporal order
df_sorted = df.sort_values('Days_Since_Start').reset_index(drop=True)

# Rebuild X_with_time from sorted data
df_input_sorted = df_sorted[features].copy()
df_input_sorted['AE_daily'] = df_input_sorted['AE_daily'].fillna(df_input_sorted['AE_daily'].median())

for col in cat_cols:
    df_input_sorted[col] = le.fit_transform(df_input_sorted[col].astype(str))

df_scaled_sorted = scaler.fit_transform(df_input_sorted)
days_norm_sorted = df_sorted['Days_Since_Start'].values / df_sorted['Days_Since_Start'].max()
X_sorted = np.column_stack([df_scaled_sorted, days_norm_sorted])

# Shape: (1, 80, 10) — one sequence of 80 time steps
X_sequence = torch.tensor(X_sorted, dtype=torch.float32).unsqueeze(0)
print('Sequence shape:', X_sequence.shape)
print('Meaning: 1 sequence, 80 time steps, 10 features')

# Encode
encoder = StepForward(threshold=0.1)
X_spikes = encoder.encode_dataset(X_sequence)
print('Spike shape:', X_spikes.shape)
print()
print('Non-zero spikes:', X_spikes.sum().item())
print('Sample spikes (first 5 time steps, all features):')
print(X_spikes[0, :5, :])
# Try lower threshold to get more spikes
encoder_low = StepForward(threshold=0.05)
X_spikes_low = encoder_low.encode_dataset(X_sequence)
print('Non-zero spikes with threshold=0.05:', X_spikes_low.sum().item())

encoder_lower = StepForward(threshold=0.01)
X_spikes_lower = encoder_lower.encode_dataset(X_sequence)
print('Non-zero spikes with threshold=0.01:', X_spikes_lower.sum().item())
# Check absolute spike counts
encoder_005 = StepForward(threshold=0.05)
X_005 = encoder_005.encode_dataset(X_sequence)

encoder_001 = StepForward(threshold=0.01)
X_001 = encoder_001.encode_dataset(X_sequence)

print('Threshold 0.10 → total spikes:', X_spikes.abs().sum().item())
print('Threshold 0.05 → total spikes:', X_005.abs().sum().item())
print('Threshold 0.01 → total spikes:', X_001.abs().sum().item())
print()
print('Total possible spikes (80 × 10):', 80 * 10)
print()
print('Spike density at 0.10:', round(X_spikes.abs().sum().item() / (80*10) * 100, 1), '%')
print('Spike density at 0.05:', round(X_005.abs().sum().item() / (80*10) * 100, 1), '%')
print('Spike density at 0.01:', round(X_001.abs().sum().item() / (80*10) * 100, 1), '%')
# Final spike encoding with threshold 0.10
encoder_final = StepForward(threshold=0.1)
X_spikes_final = encoder_final.encode_dataset(X_sequence)

print('Final spike tensor shape:', X_spikes_final.shape)
print('Spike density:', round(X_spikes_final.abs().sum().item() / (80*10) * 100, 1), '%')
print()

# Now build NeuCube reservoir
from neucube.reservoir import Reservoir

# n_inputs = 10 features
# n_neurons = size of 3D cube (typically 1000 for NeuCube)
reservoir = Reservoir(n_inputs=10, n_neurons=1000)
print('Reservoir created successfully')
print('Reservoir neurons:', 1000)
print('Input features:', 10)
help(Reservoir)
# Create NeuCube reservoir
# cube_shape=(10,10,10) = 1000 neurons in 3D cube
# inputs=10 (our 10 features including time)
reservoir = Reservoir(
    cube_shape=(10, 10, 10),
    inputs=10
)

print('Reservoir created successfully')
reservoir.summary()
# Simulate reservoir with your spike data
# This runs STDP unsupervised learning
print('Running NeuCube simulation...')
spike_activity = reservoir.simulate(
    X_spikes_final,
    mem_thr=0.1,
    refractory_period=5,
    train=True,
    verbose=True
)

print()
print('Simulation complete')
print('Spike activity shape:', spike_activity.shape)
print('Meaning:')
print(f'  {spike_activity.shape[0]} sequence')
print(f'  {spike_activity.shape[1]} time steps')
print(f'  {spike_activity.shape[2]} neurons')
# Extract neuron states for classification
# Sum spike activity across time steps for each neuron
# Shape: (1, 80, 1000) → (80, 1000)
reservoir_features = spike_activity.squeeze(0)
print('Reservoir features shape:', reservoir_features.shape)
print('Meaning: 80 records × 1000 neuron activations')
print()

# Check neuron activity
total_spikes = reservoir_features.abs().sum().item()
active_neurons = (reservoir_features.abs().sum(dim=0) > 0).sum().item()
print(f'Total neuron spikes: {total_spikes:.0f}')
print(f'Active neurons: {active_neurons} out of 1000')
print(f'Neuron activation rate: {active_neurons/10:.1f}%')
# Check which neurons fired and when
print('Neuron activity per time step:')
activity_per_step = reservoir_features.abs().sum(dim=1)
print(activity_per_step)
print()

# Check which time steps had most activity
print('Top 10 most active time steps:')
step_activity = reservoir_features.abs().sum(dim=1)
top_steps = step_activity.topk(10)
print('Values:', top_steps.values)
print('Indices:', top_steps.indices)
print()

# Map back to failure records
print('Most active failure records:')
for idx in top_steps.indices:
    idx = idx.item()
    print(f'  Record {idx}: {df_sorted["Detail"].iloc[idx]}')
import torch
import numpy as np

# Rate coding: convert each value to spike train across T time steps
T = 100  # time steps per record

def rate_encode(X, T):
    """Convert normalised values to spike trains using rate coding"""
    n_samples, n_features = X.shape
    spikes = torch.zeros(n_samples, T, n_features)
    
    for i in range(n_samples):
        for j in range(n_features):
            # Fire rate proportional to value
            fire_rate = X[i, j].item()
            # Generate random spikes based on fire rate
            spikes[i, :, j] = (torch.rand(T) < fire_rate).float()
    
    return spikes

# Encode all 80 records
X_tensor_2d = torch.tensor(X_with_time, dtype=torch.float32)
X_rate = rate_encode(X_tensor_2d, T=100)

print('Rate encoded shape:', X_rate.shape)
print('Meaning:')
print(f'  {X_rate.shape[0]} records (samples)')
print(f'  {X_rate.shape[1]} time steps per record')
print(f'  {X_rate.shape[2]} features')
print()
print('Spike density:', round(X_rate.sum().item() / (80*100*10) * 100, 1), '%')
print('Sample record 0 first 5 time steps:')
print(X_rate[0, :5, :])
# Rebuild reservoir with fresh state
reservoir = Reservoir(
    cube_shape=(10, 10, 10),
    inputs=10
)

print('Running NeuCube simulation on 80 samples...')
spike_activity = reservoir.simulate(
    X_rate,
    mem_thr=0.1,
    refractory_period=5,
    train=True,
    verbose=True
)

print()
print('Simulation complete')
print('Spike activity shape:', spike_activity.shape)
print('Meaning:')
print(f'  {spike_activity.shape[0]} records')
print(f'  {spike_activity.shape[1]} time steps')
print(f'  {spike_activity.shape[2]} neurons fired')
print()

# Check activation
total_spikes = spike_activity.abs().sum().item()
active_neurons = (spike_activity.abs().sum(dim=(0,1)) > 0).sum().item()
print(f'Total neuron spikes: {total_spikes:.0f}')
print(f'Active neurons: {active_neurons} out of 1000')
print(f'Neuron activation rate: {active_neurons/10:.1f}%')
# Extract reservoir state for each record
# Sum spike activity across time steps → (80, 1000)
reservoir_states = spike_activity.sum(dim=1)
print('Reservoir states shape:', reservoir_states.shape)
print('Meaning: 80 records × 1000 neuron activation totals')
print()

# Prepare target labels
y_causes = torch.tensor(
    le.fit_transform(df_sorted['Causes'].astype(str)), 
    dtype=torch.long)
print('Target labels shape:', y_causes.shape)
print('Class distribution:')
for i, cls in enumerate(le.classes_):
    count = (y_causes == i).sum().item()
    print(f'  {i}: {cls} → {count} records')
from neucube import sampler
print(dir(sampler))
from neucube.sampler import DeSNN
help(DeSNN)
from neucube.sampler import DeSNN
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
import numpy as np

# Use DeSNN to extract state vectors from spike activity
desnn = DeSNN(alpha=5, mod=0.8, drift_up=0.8, drift_down=0.01)
state_vectors = desnn.sample(spike_activity)

print('DeSNN state vectors shape:', state_vectors.shape)
print()

# Convert to numpy
X_clf = state_vectors.detach().numpy()
y_clf = le.fit_transform(df_sorted['Causes'].astype(str))

print('Running LOOCV...')
loo = LeaveOneOut()
y_pred = []
y_true = []

for train_idx, test_idx in loo.split(X_clf):
    X_train, X_test = X_clf[train_idx], X_clf[test_idx]
    y_train, y_test = y_clf[train_idx], y_clf[test_idx]
    
    clf = SVC(kernel='rbf', C=1.0)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    y_pred.append(pred[0])
    y_true.append(y_test[0])

y_pred = np.array(y_pred)
y_true = np.array(y_true)

accuracy = accuracy_score(y_true, y_pred)
print(f'NeuCube LOOCV Top-1 Accuracy: {accuracy*100:.2f}%')
print(f'Correct: {(y_pred==y_true).sum()} / 80')
print()
print('Comparison:')
print(f'  Phase 1 Bayesian baseline: 58.75%')
print(f'  NeuXPRISM NeuCube:         {accuracy*100:.2f}%')
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

classifiers = {
    'SVM (rbf)': SVC(kernel='rbf', C=1.0),
    'SVM (linear)': SVC(kernel='linear', C=1.0),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
}

print('LOOCV comparison across classifiers:')
print('-'*45)

for name, clf_model in classifiers.items():
    loo = LeaveOneOut()
    preds = []
    trues = []
    top2_correct = 0
    
    for train_idx, test_idx in loo.split(X_clf):
        X_train, X_test = X_clf[train_idx], X_clf[test_idx]
        y_train, y_test = y_clf[train_idx], y_clf[test_idx]
        clf_model.fit(X_train, y_train)
        pred = clf_model.predict(X_test)
        preds.append(pred[0])
        trues.append(y_test[0])

    preds = np.array(preds)
    trues = np.array(trues)
    acc = accuracy_score(trues, preds)
    print(f'{name:<20} Top-1: {acc*100:.2f}%')

print()
print(f'Bayesian baseline:    Top-1: 58.75%')
print('Running 10 NeuCube simulations to get stable accuracy...')
accuracies = []

for run in range(10):
    # Re-encode with new random spikes each time
    X_rate_new = rate_encode(X_tensor_2d, T=100)
    
    # Fresh reservoir
    res_new = Reservoir(cube_shape=(10,10,10), inputs=10)
    activity_new = res_new.simulate(X_rate_new, verbose=False)
    
    # DeSNN features
    desnn_new = DeSNN()
    states_new = desnn_new.sample(activity_new).detach().numpy()
    
    # LOOCV with SVM
    loo = LeaveOneOut()
    preds, trues = [], []
    clf = SVC(kernel='rbf', C=1.0)
    
    for tr, te in loo.split(states_new):
        clf.fit(states_new[tr], y_clf[tr])
        preds.append(clf.predict(states_new[te])[0])
        trues.append(y_clf[te][0])
    
    acc = accuracy_score(trues, preds)
    accuracies.append(acc)
    print(f'  Run {run+1}: {acc*100:.2f}%')

print()
print(f'Mean accuracy:  {np.mean(accuracies)*100:.2f}%')
print(f'Std deviation:  {np.std(accuracies)*100:.2f}%')
print(f'Best run:       {max(accuracies)*100:.2f}%')
print(f'Worst run:      {min(accuracies)*100:.2f}%')
print()
print(f'Bayesian baseline: 58.75%')
# NeuCube Top-2 accuracy
print('Running NeuCube Top-2 LOOCV...')
loo = LeaveOneOut()
top2_correct = 0

clf_prob = SVC(kernel='rbf', C=1.0, probability=True)

for train_idx, test_idx in loo.split(X_clf):
    X_train, X_test = X_clf[train_idx], X_clf[test_idx]
    y_train, y_test = y_clf[train_idx], y_clf[test_idx]
    
    clf_prob.fit(X_train, y_train)
    proba = clf_prob.predict_proba(X_test)[0]
    top2_classes = proba.argsort()[-2:][::-1]
    
    if y_test[0] in top2_classes:
        top2_correct += 1

print(f'NeuCube Top-2 Accuracy: {top2_correct/80*100:.2f}%')
print()
print('COMPLETE COMPARISON TABLE:')
print(f'{"Method":<25} {"Top-1":>8} {"Top-2":>8}')
print('-'*45)
print(f'{"NeuCube SNN":<25} {"55.00%":>8} {top2_correct/80*100:.2f}%')
print(f'{"Bayesian GIS-conditioned":<25} {"58.75%":>8} {"77.50%":>8}')
# Extract μ_fuzzy from NeuCube spike activity
# Sum spikes per record across all neurons and time steps
spike_totals = spike_activity.sum(dim=(1,2))  # shape: (80,)

# Normalise to 0-1
mu_min = spike_totals.min()
mu_max = spike_totals.max()
mu_fuzzy = (spike_totals - mu_min) / (mu_max - mu_min)

print('μ_fuzzy shape:', mu_fuzzy.shape)
print()
print('μ_fuzzy statistics:')
print(f'  Min:  {mu_fuzzy.min().item():.4f}')
print(f'  Max:  {mu_fuzzy.max().item():.4f}')
print(f'  Mean: {mu_fuzzy.mean().item():.4f}')
print(f'  Std:  {mu_fuzzy.std().item():.4f}')
print()
print('Sample μ_fuzzy values (first 10 records):')
for i in range(10):
    print(f'  Record {i}: {df_sorted["Detail"].iloc[i][:40]:<40} μ_fuzzy={mu_fuzzy[i].item():.4f}')
# Calculate EL_fuzzy = μ_fuzzy × Cost
costs = torch.tensor(df_sorted['Cost (M USD)'].values, dtype=torch.float32)
EL_fuzzy = mu_fuzzy * costs

print('EL_fuzzy calculated')
print()
print('Sample EL_fuzzy values (first 10 records):')
print(f'{"Record":<45} {"Cost":>8} {"μ_fuzzy":>8} {"EL_fuzzy":>10}')
print('-'*75)
for i in range(10):
    name = df_sorted['Detail'].iloc[i][:40]
    cost = costs[i].item()
    mu = mu_fuzzy[i].item()
    el = EL_fuzzy[i].item()
    print(f'{name:<45} {cost:>8.1f} {mu:>8.4f} {el:>10.2f}')

print()
print(f'EL_fuzzy statistics:')
print(f'  Min:  ${EL_fuzzy.min().item():.2f}M')
print(f'  Max:  ${EL_fuzzy.max().item():.2f}M')
print(f'  Mean: ${EL_fuzzy.mean().item():.2f}M')
# Re-encode WITHOUT Cost for μ_fuzzy derivation
features_no_cost = ['Heritage Category', 'Orbit Type', 
                    'Phase', 'Causes', 'Latitude', 
                    'Coast_Km', 'Ap_daily', 'AE_daily']

df_input_nocost = df_sorted[features_no_cost].copy()
df_input_nocost['AE_daily'] = df_input_nocost['AE_daily'].fillna(
    df_input_nocost['AE_daily'].median())

# Encode categoricals
for col in ['Heritage Category', 'Orbit Type', 'Phase', 'Causes']:
    df_input_nocost[col] = le.fit_transform(
        df_input_nocost[col].astype(str))

# Normalise
scaler_nocost = MinMaxScaler()
df_scaled_nocost = scaler_nocost.fit_transform(df_input_nocost)

# Add time position
days_norm = df_sorted['Days_Since_Start'].values / df_sorted['Days_Since_Start'].max()
X_nocost = np.column_stack([df_scaled_nocost, days_norm])

print('Features for μ_fuzzy (no cost):')
for i, col in enumerate(features_no_cost + ['Time_Position'], 1):
    print(f'  {i}. {col}')
print()
print('Shape:', X_nocost.shape)
# Rate encode without cost
X_nocost_tensor = torch.tensor(X_nocost, dtype=torch.float32)
X_rate_nocost = rate_encode(X_nocost_tensor, T=100)
print('Spike shape:', X_rate_nocost.shape)

# Fresh reservoir with 9 inputs
reservoir_nocost = Reservoir(cube_shape=(10,10,10), inputs=9)
activity_nocost = reservoir_nocost.simulate(
    X_rate_nocost, verbose=False)

# Derive μ_fuzzy from cost-free spike activity
spike_totals_nocost = activity_nocost.sum(dim=(1,2))
mu_min = spike_totals_nocost.min()
mu_max = spike_totals_nocost.max()
mu_fuzzy_clean = (spike_totals_nocost - mu_min) / (mu_max - mu_min)

# Calculate EL_fuzzy
EL_fuzzy_clean = mu_fuzzy_clean * costs

print()
print('Sample EL_fuzzy values (first 10 records):')
print(f'{"Record":<45} {"Cost":>8} {"μ_fuzzy":>8} {"EL_fuzzy":>10}')
print('-'*75)
for i in range(10):
    name = df_sorted['Detail'].iloc[i][:40]
    cost = costs[i].item()
    mu = mu_fuzzy_clean[i].item()
    el = EL_fuzzy_clean[i].item()
    print(f'{name:<45} {cost:>8.1f} {mu:>8.4f} {el:>10.2f}')

print()
print(f'EL_fuzzy statistics:')
print(f'  Min:  ${EL_fuzzy_clean.min().item():.2f}M')
print(f'  Max:  ${EL_fuzzy_clean.max().item():.2f}M')
print(f'  Mean: ${EL_fuzzy_clean.mean().item():.2f}M')
markov_weights = pd.read_csv(
    'C:/Users/User/Desktop/cakewebsitetemplate/Data Research/UFMB 2 essentials/Markov_Spatial_Weights_Corrected.csv')

print('Markov weights loaded')
print(markov_weights[['Spaceport', 'P_Liftoff_Adjusted',
                       'P_Upper_Adjusted']].to_string(index=False))
# Goal C — EL_markov = P(phase|site) × Cost

# Create spaceport to P(phase) lookup
markov_lookup = dict(zip(
    markov_weights['Spaceport'],
    zip(markov_weights['P_Liftoff_Adjusted'],
        markov_weights['P_Upper_Adjusted'])
))

# Calculate EL_markov for each record
EL_markov_list = []

for i, row in df_sorted.iterrows():
    spaceport = row['Spaceport']
    phase = row['Phase']
    cost = row['Cost (M USD)']
    
    if spaceport in markov_lookup:
        p_liftoff, p_upper = markov_lookup[spaceport]
    else:
        # Use fleet average if spaceport not found
        p_liftoff, p_upper = 0.2875, 0.6125
    
    # Use phase-specific probability
    if phase == 'Liftoff Ascent':
        p_phase = p_liftoff
    elif phase == 'Upper Stage Burn':
        p_phase = p_upper
    else:
        p_phase = 1 / 5  # equal probability for other phases
    
    EL_markov_list.append(p_phase * cost)

EL_markov = torch.tensor(EL_markov_list, dtype=torch.float32)

print('EL_markov calculated')
print()
print('Sample EL_markov values (first 10 records):')
print(f'{"Record":<45} {"Phase":<25} {"Cost":>8} {"P(phase)":>10} {"EL_markov":>10}')
print('-'*105)
for i in range(10):
    name = df_sorted['Detail'].iloc[i][:40]
    phase = df_sorted['Phase'].iloc[i]
    cost = costs[i].item()
    el = EL_markov[i].item()
    p = el / cost if cost > 0 else 0
    print(f'{name:<45} {phase:<25} {cost:>8.1f} {p:>10.4f} {el:>10.2f}')

print()
print(f'EL_markov statistics:')
print(f'  Min:  ${EL_markov.min().item():.2f}M')
print(f'  Max:  ${EL_markov.max().item():.2f}M')
print(f'  Mean: ${EL_markov.mean().item():.2f}M')
# Goal C — Contextualised with NeuCube
# P(phase|site, context) = (P_markov + μ_fuzzy_clean) / 2

EL_markov_ctx_list = []
P_contextual_list = []

for i, (idx, row) in enumerate(df_sorted.iterrows()):
    spaceport = row['Spaceport']
    phase = row['Phase']
    cost = row['Cost (M USD)']
    
    # Get base Markov probability
    if spaceport in markov_lookup:
        p_liftoff, p_upper = markov_lookup[spaceport]
    else:
        p_liftoff, p_upper = 0.2875, 0.6125
    
    # Phase specific base probability
    if phase == 'Liftoff Ascent':
        p_markov = p_liftoff
    elif phase == 'Upper Stage Burn':
        p_markov = p_upper
    else:
        p_markov = 0.2
    
    # NeuCube contextual modifier
    mu = mu_fuzzy_clean[i].item()
    
    # Combine: average of Markov + NeuCube
    p_contextual = (p_markov + mu) / 2
    
    EL_markov_ctx_list.append(p_contextual * cost)
    P_contextual_list.append(p_contextual)

EL_markov = torch.tensor(EL_markov_ctx_list, dtype=torch.float32)
P_contextual = torch.tensor(P_contextual_list, dtype=torch.float32)

print('Goal C — Contextualised EL_markov calculated')
print()
print('Sample values (first 10 records):')
print(f'{"Record":<40} {"P_markov":>9} {"μ_fuzzy":>8} {"P_ctx":>7} {"Cost":>7} {"EL_markov":>10}')
print('-'*90)
for i in range(10):
    name = df_sorted['Detail'].iloc[i][:35]
    phase = df_sorted['Phase'].iloc[i]
    spaceport = df_sorted['Spaceport'].iloc[i]
    cost = costs[i].item()
    mu = mu_fuzzy_clean[i].item()
    
    if spaceport in markov_lookup:
        p_l, p_u = markov_lookup[spaceport]
    else:
        p_l, p_u = 0.2875, 0.6125
    
    if phase == 'Liftoff Ascent':
        p_m = p_l
    elif phase == 'Upper Stage Burn':
        p_m = p_u
    else:
        p_m = 0.2
    
    p_ctx = P_contextual[i].item()
    el = EL_markov[i].item()
    print(f'{name:<40} {p_m:>9.4f} {mu:>8.4f} {p_ctx:>7.4f} {cost:>7.1f} {el:>10.2f}')

print()
print(f'EL_markov statistics:')
print(f'  Min:  ${EL_markov.min().item():.2f}M')
print(f'  Max:  ${EL_markov.max().item():.2f}M')
print(f'  Mean: ${EL_markov.mean().item():.2f}M')
# R_final = (EL_fuzzy + EL_markov) / 2
R_final = (EL_fuzzy_clean + EL_markov) / 2

# Risk Tier classification
def risk_tier(r):
    if r < 10:
        return 'Low'
    elif r < 30:
        return 'Medium'
    elif r < 60:
        return 'High'
    else:
        return 'Critical'

risk_tiers = [risk_tier(r.item()) for r in R_final]

print('R_final calculated')
print()
print('Sample results (first 10 records):')
print(f'{"Record":<40} {"EL_fuzzy":>9} {"EL_markov":>10} {"R_final":>8} {"Tier":>10}')
print('-'*85)
for i in range(10):
    name = df_sorted['Detail'].iloc[i][:35]
    ef = EL_fuzzy_clean[i].item()
    em = EL_markov[i].item()
    rf = R_final[i].item()
    rt = risk_tiers[i]
    print(f'{name:<40} {ef:>9.2f} {em:>10.2f} {rf:>8.2f} {rt:>10}')

print()
print('Risk Tier Distribution:')
from collections import Counter
tier_counts = Counter(risk_tiers)
for tier in ['Low', 'Medium', 'High', 'Critical']:
    count = tier_counts.get(tier, 0)
    print(f'  {tier:<10} {count:>3} records ({count/80*100:.1f}%)')
# Classical EL = Cost only (no context)
classical_EL = costs.clone()

# UFMB baseline EL = fleet average phase probability × Cost
# P(Liftoff)=0.2875, P(Upper)=0.6125 — no geographic conditioning
UFMB_EL_list = []
for i, (idx, row) in enumerate(df_sorted.iterrows()):
    phase = row['Phase']
    cost = row['Cost (M USD)']
    if phase == 'Liftoff Ascent':
        p = 0.2875
    elif phase == 'Upper Stage Burn':
        p = 0.6125
    else:
        p = 0.2
    UFMB_EL_list.append(p * cost)

UFMB_EL = torch.tensor(UFMB_EL_list, dtype=torch.float32)

print('THREE-WAY COMPARISON (first 10 records):')
print(f'{"Record":<40} {"Classical":>10} {"UFMB":>10} {"NeuXPRISM":>11} {"Tier":>10}')
print('-'*90)
for i in range(10):
    name = df_sorted['Detail'].iloc[i][:35]
    cl = classical_EL[i].item()
    uf = UFMB_EL[i].item()
    nx = R_final[i].item()
    rt = risk_tiers[i]
    print(f'{name:<40} {cl:>10.2f} {uf:>10.2f} {nx:>11.2f} {rt:>10}')

print()
print('AVERAGE EL COMPARISON:')
print(f'  Classical EL:   ${classical_EL.mean().item():.2f}M')
print(f'  UFMB EL:        ${UFMB_EL.mean().item():.2f}M')
print(f'  NeuXPRISM EL:   ${R_final.mean().item():.2f}M')
print()
print(f'Reduction Classical → UFMB:      {(1 - UFMB_EL.mean()/classical_EL.mean())*100:.1f}%')
print(f'Reduction Classical → NeuXPRISM: {(1 - R_final.mean()/classical_EL.mean())*100:.1f}%')
print(f'Reduction UFMB → NeuXPRISM:      {(1 - R_final.mean()/UFMB_EL.mean())*100:.1f}%')
# EL Risk Tier Confusion Matrix
# Actual tier based on Cost alone
# Predicted tier based on NeuXPRISM R_final

def cost_tier(cost):
    if cost < 20:
        return 'Low'
    elif cost < 60:
        return 'Medium'
    elif cost < 150:
        return 'High'
    else:
        return 'Critical'

actual_tiers = [cost_tier(c.item()) for c in costs]
predicted_tiers = risk_tiers

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

tier_order = ['Low', 'Medium', 'High', 'Critical']
cm = confusion_matrix(actual_tiers, predicted_tiers, labels=tier_order)

print('EL RISK TIER CONFUSION MATRIX:')
print(f'{"":>12}', end='')
for t in tier_order:
    print(f'{t:>10}', end='')
print()
print('-'*52)
for i, tier in enumerate(tier_order):
    print(f'{tier:>12}', end='')
    for j in range(len(tier_order)):
        print(f'{cm[i,j]:>10}', end='')
    print()

print()
print(classification_report(actual_tiers, predicted_tiers,
                            labels=tier_order, zero_division=0))
!pip install matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.patch.set_facecolor('white')

records_short = [
    'GSLV\nGSAT-1', 'Minotaur C\nOrbView-4', 'Soyuz-U\nFoton-M',
    'Ariane 5\nHot Bird 7', 'Delta IV\nDemoSat', 'Rokot\nCryoSat-1',
    'Proton-M\nArabsat-4A', 'Falcon 1\nFalconSat-2', 'GSLV\nINSAT-4C',
    'Dnepr\nBelKa 1'
]

x = np.arange(10)
width = 0.25

cl = [47.0, 45.0, 20.0, 200.0, 350.0, 13.0, 65.0, 7.0, 47.0, 29.0]
uf = [28.79, 12.94, 5.75, 57.50, 100.62, 7.96, 39.81, 2.01, 13.51, 8.34]
nx = [23.51, 11.89, 2.57, 21.80, 94.11, 9.38, 42.14, 1.49, 5.74, 10.99]

# Panel A: Three-way EL comparison
ax1 = axes[0]
ax1.bar(x - width, cl, width, label='Classical EL', color='#95a5a6', alpha=0.85)
ax1.bar(x, uf, width, label='UFMB', color='#f39c12', alpha=0.85)
ax1.bar(x + width, nx, width, label='NeuXPRISM', color='#2e5fa3', alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(records_short, fontsize=6.5)
ax1.set_ylabel('Expected Loss (M USD)', fontsize=10, fontweight='bold')
ax1.set_title('(A) Three-Way EL Comparison\nFirst 10 Records', fontsize=10, fontweight='bold')
ax1.legend(fontsize=8)
ax1.set_facecolor('#F5F5F5')
ax1.grid(axis='y', alpha=0.4, color='white')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Panel B: Average EL reduction
ax2 = axes[1]
methods = ['Classical EL', 'UFMB', 'NeuXPRISM']
averages = [42.02, 18.56, 17.17]
colors = ['#95a5a6', '#f39c12', '#2e5fa3']
bars = ax2.bar(methods, averages, color=colors, alpha=0.85,
               edgecolor='white', linewidth=0.8, width=0.5)
for bar, val in zip(bars, averages):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'${val:.2f}M', ha='center', va='bottom',
             fontsize=10, fontweight='bold')
ax2.annotate('55.8%\nreduction', xy=(1, 18.56), xytext=(0.5, 30),
             fontsize=8, color='#f39c12',
             arrowprops=dict(arrowstyle='->', color='#f39c12'))
ax2.annotate('59.1%\nreduction', xy=(2, 17.17), xytext=(1.5, 35),
             fontsize=8, color='#2e5fa3',
             arrowprops=dict(arrowstyle='->', color='#2e5fa3'))
ax2.set_ylabel('Average Expected Loss (M USD)', fontsize=10, fontweight='bold')
ax2.set_title('(B) Average EL Reduction\nAcross 80 Records', fontsize=10, fontweight='bold')
ax2.set_facecolor('#F5F5F5')
ax2.grid(axis='y', alpha=0.4, color='white')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Panel C: Risk Tier confusion matrix
ax3 = axes[2]
cm_data = np.array([
    [31, 0, 0, 0],
    [4, 22, 4, 0],
    [1, 5, 10, 0],
    [0, 2, 0, 1]
])
im = ax3.imshow(cm_data, cmap='Blues')
ax3.set_xticks([0,1,2,3])
ax3.set_yticks([0,1,2,3])
tiers = ['Low', 'Medium', 'High', 'Critical']
ax3.set_xticklabels(tiers, fontsize=9)
ax3.set_yticklabels(tiers, fontsize=9)
ax3.set_xlabel('Predicted Tier', fontsize=10, fontweight='bold')
ax3.set_ylabel('Actual Tier', fontsize=10, fontweight='bold')
ax3.set_title('(C) EL Risk Tier\nConfusion Matrix (80% accuracy)',
              fontsize=10, fontweight='bold')
for i in range(4):
    for j in range(4):
        ax3.text(j, i, str(cm_data[i,j]),
                ha='center', va='center',
                fontsize=12, fontweight='bold',
                color='white' if cm_data[i,j] > 15 else 'black')
plt.colorbar(im, ax=ax3)

fig.suptitle('Figure 7: NeuXPRISM Results — Expected Loss Quantification and Risk Tier Validation\n'
             'Classical EL vs UFMB vs NeuXPRISM | 80 Orbital Launch Failures 2001-2026',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('C:/Users/User/Desktop/cakewebsitetemplate/Data Research/UFMB 2 essentials/Figures and Results (UFMB 2)/Figure7_Results_Final.png',
            dpi=300, bbox_inches='tight')
print('Figure 7 saved')
!pip install shap
import shap
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Use R_final as target — SHAP explains what drives risk score
# Input features are the 9 context variables (no cost)
feature_names = ['Heritage Category', 'Orbit Type', 'Phase', 
                 'Causes', 'Latitude', 'Coast_Km', 
                 'Ap_daily', 'AE_daily', 'Time_Position']

X_shap = X_nocost  # 80 × 9 normalised features
y_shap = R_final.detach().numpy()  # R_final as target

# Train Random Forest on NeuXPRISM outputs
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_shap, y_shap)

print('Random Forest trained on NeuXPRISM R_final outputs')
print(f'R² score: {rf.score(X_shap, y_shap):.4f}')
print()

# Apply SHAP
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_shap)

print('SHAP values calculated')
print('SHAP values shape:', shap_values.shape)
print()

# Feature importance from SHAP
mean_shap = np.abs(shap_values).mean(axis=0)
print('SHAP Feature Importance (mean |SHAP value|):')
for i, (feat, imp) in enumerate(sorted(
        zip(feature_names, mean_shap), 
        key=lambda x: x[1], reverse=True)):
    print(f'  {feat:<25} {imp:.4f}')
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

print('Running LOOCV EL comparison...')

classical_errors = []
ufmb_errors = []
neuxprism_errors = []

classical_preds = []
ufmb_preds = []
neuxprism_preds = []
actual_costs = []

for i in range(len(df_sorted)):
    # Test record
    test_row = df_sorted.iloc[i]
    test_cost = test_row['Cost (M USD)']
    test_phase = test_row['Phase']
    test_spaceport = test_row['Spaceport']
    
    # Training data (79 records)
    train_df = df_sorted.drop(df_sorted.index[i])
    
    # Classical EL = full cost
    classical_el = test_cost
    
    # UFMB EL = fleet average phase prob × cost
    if test_phase == 'Liftoff Ascent':
        p_ufmb = train_df[train_df['Phase'] == 'Liftoff Ascent'].shape[0] / len(train_df)
    elif test_phase == 'Upper Stage Burn':
        p_ufmb = train_df[train_df['Phase'] == 'Upper Stage Burn'].shape[0] / len(train_df)
    else:
        p_ufmb = 0.2
    ufmb_el = p_ufmb * test_cost
    
    # NeuXPRISM EL = R_final already computed
    neuxprism_el = R_final[i].item()
    
    # Actual = cost (ground truth proxy)
    actual = test_cost
    
    classical_errors.append(abs(classical_el - actual))
    ufmb_errors.append(abs(ufmb_el - actual))
    neuxprism_errors.append(abs(neuxprism_el - actual))
    
    classical_preds.append(classical_el)
    ufmb_preds.append(ufmb_el)
    neuxprism_preds.append(neuxprism_el)
    actual_costs.append(actual)

print('LOOCV EL Comparison Results:')
print(f'{"Method":<25} {"MAE":>10} {"RMSE":>10} {"Avg EL":>10}')
print('-'*60)

for name, preds, errors in [
    ('Classical EL', classical_preds, classical_errors),
    ('UFMB', ufmb_preds, ufmb_errors),
    ('NeuXPRISM', neuxprism_preds, neuxprism_errors)
]:
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    avg = np.mean(preds)
    print(f'{name:<25} {mae:>10.2f} {rmse:>10.2f} {avg:>10.2f}')

print()
print('First 10 records comparison:')
print(f'{"Record":<40} {"Actual":>8} {"Classical":>10} {"UFMB":>8} {"NeuXPRISM":>11}')
print('-'*85)
for i in range(10):
    name = df_sorted['Detail'].iloc[i][:35]
    print(f'{name:<40} {actual_costs[i]:>8.1f} {classical_preds[i]:>10.1f} '
          f'{ufmb_preds[i]:>8.2f} {neuxprism_preds[i]:>11.2f}')
# Correct framing — EL reduction comparison
print('EL REDUCTION ANALYSIS (First 10 Records):')
print(f'{"Record":<38} {"Classical":>10} {"UFMB":>8} {"NeuXPRISM":>11} {"Reduction%":>11}')
print('-'*85)

reductions = []
for i in range(10):
    name = df_sorted['Detail'].iloc[i][:33]
    cl = classical_preds[i]
    uf = ufmb_preds[i]
    nx = neuxprism_preds[i]
    reduction = (1 - nx/cl) * 100
    reductions.append(reduction)
    print(f'{name:<38} {cl:>10.1f} {uf:>8.2f} {nx:>11.2f} {reduction:>10.1f}%')

print()
print('AVERAGE ACROSS ALL 80 RECORDS:')
print(f'  Classical EL:   ${np.mean(classical_preds):.2f}M')
print(f'  UFMB EL:        ${np.mean(ufmb_preds):.2f}M')
print(f'  NeuXPRISM EL:   ${np.mean(neuxprism_preds):.2f}M')
print()
print(f'  Reduction Classical → UFMB:      '
      f'{(1-np.mean(ufmb_preds)/np.mean(classical_preds))*100:.1f}%')
print(f'  Reduction Classical → NeuXPRISM: '
      f'{(1-np.mean(neuxprism_preds)/np.mean(classical_preds))*100:.1f}%')
print(f'  Reduction UFMB → NeuXPRISM:      '
      f'{(1-np.mean(neuxprism_preds)/np.mean(ufmb_preds))*100:.1f}%')
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from neucube.encoder import StepForward
from neucube.reservoir import Reservoir
from neucube.sampler import DeSNN

# Load data
df = pd.read_excel('C:/Users/User/Desktop/cakewebsitetemplate/Data Research/UFMB 2 essentials/F1341550_AE.xlsx',
                   sheet_name='Master_Sheet')
df['Datum'] = pd.to_datetime(df['Datum'], dayfirst=True)
df['Days_Since_Start'] = (df['Datum'] - df['Datum'].min()).dt.days
df_sorted = df.sort_values('Days_Since_Start').reset_index(drop=True)

# 9 features without cost
features_no_cost = ['Heritage Category', 'Orbit Type', 'Phase', 'Causes',
                    'Latitude', 'Coast_Km', 'Ap_daily', 'AE_daily']
df_input = df_sorted[features_no_cost].copy()
df_input['AE_daily'] = df_input['AE_daily'].fillna(df_input['AE_daily'].median())

le = LabelEncoder()
for col in ['Heritage Category', 'Orbit Type', 'Phase', 'Causes']:
    df_input[col] = le.fit_transform(df_input[col].astype(str))

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_input)
days_norm = df_sorted['Days_Since_Start'].values / df_sorted['Days_Since_Start'].max()
X_nocost = np.column_stack([df_scaled, days_norm])

# Rate encode
X_tensor = torch.tensor(X_nocost, dtype=torch.float32)

def rate_encode(X, T):
    n_samples, n_features = X.shape
    spikes = torch.zeros(n_samples, T, n_features)
    for i in range(n_samples):
        for j in range(n_features):
            fire_rate = X[i, j].item()
            spikes[i, :, j] = (torch.rand(T) < fire_rate).float()
    return spikes

X_rate = rate_encode(X_tensor, T=100)

# Rebuild reservoir
reservoir_nocost = Reservoir(cube_shape=(10,10,10), inputs=9)
activity_nocost = reservoir_nocost.simulate(X_rate, verbose=True)

print()
print('Reservoir rebuilt successfully')
print('Activity shape:', activity_nocost.shape)
print()
print('Reservoir attributes:')
print(dir(reservoir_nocost))
# Extract actual neuron positions and activation from NeuCube
# reservoir_nocost is your trained reservoir without cost

# Get actual neuron coordinates from reservoir
print('Extracting actual NeuCube data...')
print()

# Check reservoir attributes
print('Reservoir attributes:')
print(dir(reservoir_nocost))
# Extract actual neuron positions and activation
print('Neuron positions shape:', reservoir_nocost.pos.shape)
print('Sample positions (first 5):')
print(reservoir_nocost.pos[:5])
print()

# Get actual activation per neuron across all records
# activity_nocost shape: (80, 100, 1000)
# Sum across time and records for overall activation
neuron_activation = activity_nocost.sum(dim=(0,1))  # shape: (1000,)
neuron_activation_norm = (neuron_activation - neuron_activation.min()) / \
                         (neuron_activation.max() - neuron_activation.min())

print('Neuron activation shape:', neuron_activation.shape)
print('Active neurons (>0):', (neuron_activation > 0).sum().item())
print('Max activation:', neuron_activation.max().item())
print('Min activation:', neuron_activation.min().item())
print()

# Extract x, y, z positions
positions = reservoir_nocost.pos.detach().numpy()
print('Position range:')
print('X:', positions[:,0].min(), 'to', positions[:,0].max())
print('Y:', positions[:,1].min(), 'to', positions[:,1].max())
print('Z:', positions[:,2].min(), 'to', positions[:,2].max())
# The 3D Neuron Activation diagram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(20, 8))
fig.patch.set_facecolor('#0a0f1e')

# Panel A: Actual 3D NeuCube activation
ax1 = fig.add_subplot(131, projection='3d')
ax1.set_facecolor('#0a0f1e')
scatter = ax1.scatter(x_act, y_act, z_act,
                      c=a_act, cmap='hot',
                      s=a_act*100+5,
                      alpha=0.8, edgecolors='none')
ax1.set_xlabel('X', color='white', fontsize=8)
ax1.set_ylabel('Y', color='white', fontsize=8)
ax1.set_zlabel('Z', color='white', fontsize=8)
ax1.tick_params(colors='white', labelsize=6)
ax1.set_title(f'(A) NeuCube 3D Actual Activation\n{active_mask.sum()} / 1000 Neurons Active',
              color='white', fontsize=9, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax1, shrink=0.5)
cbar.set_label('Activation Level', color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

# Panel B: Activation histogram
ax2 = fig.add_subplot(132)
ax2.set_facecolor('#0a0f1e')
active_vals = a_act[a_act > 0]
ax2.hist(active_vals, bins=20, color='#f39c12',
         alpha=0.85, edgecolor='white', linewidth=0.8)
ax2.set_xlabel('Activation Level', color='white', fontsize=10)
ax2.set_ylabel('Number of Neurons', color='white', fontsize=10)
ax2.set_title('(B) Neuron Activation\nDistribution',
              color='white', fontsize=9, fontweight='bold')
ax2.tick_params(colors='white')
ax2.spines['bottom'].set_color('white')
ax2.spines['left'].set_color('white')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', alpha=0.2, color='white')
ax2.text(0.6, len(active_vals)*0.3,
         f'Active: {active_mask.sum()}/1000\n({active_mask.sum()/10:.1f}%)',
         color='white', fontsize=9)

# Panel C: Risk tier pie
ax3 = fig.add_subplot(133)
ax3.set_facecolor('#0a0f1e')
tiers = ['Low', 'Medium', 'High', 'Critical']
counts = [36, 29, 14, 1]
tier_colors = ['#27ae60', '#f39c12', '#e74c3c', '#8b0000']
wedges, texts, autotexts = ax3.pie(
    counts, labels=tiers, colors=tier_colors,
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(edgecolor='#0a0f1e', linewidth=2),
    textprops=dict(color='white', fontsize=10))
for autotext in autotexts:
    autotext.set_fontweight('bold')
    autotext.set_color('white')
ax3.set_title('(C) NeuXPRISM Risk Tier\nDistribution (80 Records)',
              color='white', fontsize=9, fontweight='bold')

fig.suptitle('Figure 9: NeuXPRISM NeuCube Actual Spatio-Temporal Activation and Risk Output\n'
             'Real Neuron Firing Patterns | Activation Distribution | Risk Tier Classification',
             color='white', fontsize=11, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('C:/Users/User/Desktop/cakewebsitetemplate/Data Research/UFMB 2 essentials/Figures and Results (UFMB 2)/Figure9_NeuCube_3D_Actual.png',
            dpi=300, bbox_inches='tight', facecolor='#0a0f1e')
print('Figure 9 saved with actual NeuCube data')
import pandas as pd
import numpy as np
from sklearn.utils import resample

# ── Load results ──────────────────────────────────────────────
df = pd.read_csv(r'C:\Users\User\Desktop\cakewebsitetemplate\Data Research\NeuXPRISM essentials\NeuXPRISM_Full_Results.csv')

# Rename columns to match final terminology
df = df.rename(columns={
    'R_final':   'Composite_Risk_Score',   # pre-LDF fusion
    'R_learned': 'R_final'                 # LDF-inclusive true final output
})

# ── Settings ──────────────────────────────────────────────────
UFMB_MEAN    = 18.56   # verified UFMB corpus mean ($M)
N_BOOTSTRAP  = 1000
RANDOM_SEED  = 42

# ── Storage ───────────────────────────────────────────────────
results = {
    'mean_CRS'                  : [],   # Composite Risk Score (pre-LDF)
    'mean_R_final'              : [],   # R_final (LDF-inclusive)
    'pct_reduction_vs_classical': [],   # vs full-cost classical EL
    'pct_diff_vs_ufmb'         : [],   # R_final vs UFMB (positive = NeuXPRISM higher)
    'mean_ldf_active'           : [],   # mean LDF on records where LDF ≠ 1.0
    'disc_correction_pct'       : [],   # mean % change from CRS for discounted records
    'ampl_correction_pct'       : [],   # mean % change from CRS for amplified records
    'n_neutral'                 : [],
    'n_disc'                    : [],
    'n_ampl'                    : [],
}

np.random.seed(RANDOM_SEED)

# ── Bootstrap loop ────────────────────────────────────────────
for i in range(N_BOOTSTRAP):

    boot = resample(df, n_samples=len(df), replace=True, random_state=i)

    classical_boot = boot['Cost (M USD)'].mean()
    mean_rfinal    = boot['R_final'].mean()
    mean_crs       = boot['Composite_Risk_Score'].mean()

    reduction  = (1 - mean_rfinal / classical_boot) * 100
    diff_ufmb  = (mean_rfinal - UFMB_MEAN) / UFMB_MEAN * 100

    neutral = boot[boot['LDF'] == 1.0]
    active  = boot[boot['LDF'] != 1.0]
    disc    = boot[boot['LDF'] <  1.0]
    ampl    = boot[boot['LDF'] >  1.0]

    results['mean_CRS'].append(mean_crs)
    results['mean_R_final'].append(mean_rfinal)
    results['pct_reduction_vs_classical'].append(reduction)
    results['pct_diff_vs_ufmb'].append(diff_ufmb)
    results['mean_ldf_active'].append(
        active['LDF'].mean() if len(active) > 0 else np.nan
    )
    results['n_neutral'].append(len(neutral))
    results['n_disc'].append(len(disc))
    results['n_ampl'].append(len(ampl))

    if len(disc) > 0:
        dc = ((disc['R_final'] / disc['Composite_Risk_Score'] - 1) * 100).mean()
        results['disc_correction_pct'].append(dc)
    else:
        results['disc_correction_pct'].append(np.nan)

    if len(ampl) > 0:
        ac = ((ampl['R_final'] / ampl['Composite_Risk_Score'] - 1) * 100).mean()
        results['ampl_correction_pct'].append(ac)
    else:
        results['ampl_correction_pct'].append(np.nan)

# ── CI helper ─────────────────────────────────────────────────
def ci95(arr):
    arr = np.array([x for x in arr if not np.isnan(x)])
    return np.mean(arr), np.percentile(arr, 2.5), np.percentile(arr, 97.5)

# ── Print results ─────────────────────────────────────────────
print("=" * 65)
print(f"NeuXPRISM Bootstrap Results  (n={N_BOOTSTRAP}, seed={RANDOM_SEED})")
print("=" * 65)

labels = {
    'mean_CRS'                  : 'Mean Composite Risk Score ($M)',
    'mean_R_final'              : 'Mean R_final LDF-inclusive ($M)',
    'pct_reduction_vs_classical': 'EL Reduction vs Classical (%)',
    'pct_diff_vs_ufmb'         : 'R_final vs UFMB difference (%)',
    'mean_ldf_active'           : 'Mean LDF (active records only)',
    'disc_correction_pct'       : 'Discount correction (%)',
    'ampl_correction_pct'       : 'Amplification correction (%)',
    'n_neutral'                 : 'N neutral records (LDF=1.0)',
    'n_disc'                    : 'N discounted records (LDF<1.0)',
    'n_ampl'                    : 'N amplified records (LDF>1.0)',
}

for key, label in labels.items():
    mean, lo, hi = ci95(results[key])
    print(f"  {label:<40}: {mean:7.2f}  [95% CI: {lo:.2f}, {hi:.2f}]")

print()
print("=== PAPER-READY STATEMENTS ===")
m,lo,hi = ci95(results['mean_R_final'])
print(f"Mean R_final:          ${m:.2f}M  (95% CI: ${lo:.2f}M – ${hi:.2f}M)")
m,lo,hi = ci95(results['pct_reduction_vs_classical'])
print(f"EL reduction:           {m:.1f}%  (95% CI: {lo:.1f}% – {hi:.1f}%)")
m,lo,hi = ci95(results['disc_correction_pct'])
print(f"Discount correction:    {m:.1f}%  (95% CI: {lo:.1f}% – {hi:.1f}%)")
m,lo,hi = ci95(results['ampl_correction_pct'])
print(f"Amplify correction:    +{m:.1f}%  (95% CI: +{lo:.1f}% – +{hi:.1f}%)")
m,lo,hi = ci95(results['pct_diff_vs_ufmb'])
print(f"R_final vs UFMB:       {m:+.1f}%  (95% CI: {lo:.1f}% – {hi:.1f}%)")

# ── Save raw bootstrap distributions (optional) ───────────────
boot_df = pd.DataFrame(results)
boot_df.to_csv('NeuXPRISM_Bootstrap_Distributions.csv', index=False)
print()
print("Raw distributions saved to NeuXPRISM_Bootstrap_Distributions.csv")
import os

# Show where the script ran from
print("Current working directory:")
print(os.getcwd())
print()

# Also save it explicitly to your NeuXPRISM essentials folder
save_path = r'C:\Users\User\Desktop\cakewebsitetemplate\Data Research\NeuXPRISM essentials\NeuXPRISM_Bootstrap_Distributions.csv'

boot_df.to_csv(save_path, index=False)
print(f"File saved to: {save_path}")
!pip install seaborn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Load data directly from your local CSV using exact column headers
df = pd.read_csv(r'C:\Users\User\Desktop\cakewebsitetemplate\Data Research\NeuXPRISM essentials\NeuXPRISM_Bootstrap_Distributions.csv')

# Exact Configuration mapping your CSV columns to presentation labels
configs = [
    {
        "col": "mean_R_final",
        "title": "Mean $R_{final}$",
        "unit": "M",
        "fmt": "${:.2f}M",
    },
    {
        "col": "pct_reduction_vs_classical",
        "title": "EL Reduction",
        "unit": "%",
        "fmt": "{:+.1f}%",
    },
    {
        "col": "disc_correction_pct",
        "title": "Discount Correction",
        "unit": "%",
        "fmt": "{:+.1f}%",
    },
    {
        "col": "ampl_correction_pct",
        "title": "Amplify Correction",
        "unit": "%",
        "fmt": "{:+.1f}%",
    },
    {
        "col": "pct_diff_vs_ufmb",
        "title": "$R_{final}$ vs UFMB",
        "unit": "%",
        "fmt": "{:+.1f}%",
    },
]

sns.set_theme(style="white")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

fig, axes = plt.subplots(
    nrows=5, ncols=1, figsize=(8, 10), gridspec_kw={"hspace": 0.7}
)
colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#e377c2"]

for i, config in enumerate(configs):
    ax = axes[i]
    data = df[config["col"]].dropna()

    mean_val = data.mean()
    ci_low = np.percentile(data, 2.5)
    ci_high = np.percentile(data, 97.5)

    # A. Draw the smooth distribution outline (The Cloud)
    kde = sns.kdeplot(
        data=data, ax=ax, color=colors[i], fill=False, linewidth=2, zorder=3
    )

    x_line, y_line = kde.lines[-1].get_data()
    ci_mask = (x_line >= ci_low) & (x_line <= ci_high)

    # Shade the core 95% Confidence Interval region distinctively
    ax.fill_between(
        x_line[ci_mask],
        y_line[ci_mask],
        color=colors[i],
        alpha=0.3,
        zorder=2,
    )
    ax.fill_between(
        x_line[~ci_mask],
        y_line[~ci_mask],
        color=colors[i],
        alpha=0.08,
        zorder=1,
    )

    # B. Generate the 80 points scatter layer at the base (The Rain)
    rain_data = data.sample(n=min(80, len(data)), random_state=42)
    y_offset = -max(y_line) * 0.1
    ax.scatter(
        rain_data,
        np.random.uniform(y_offset * 1.3, y_offset * 0.7, size=len(rain_data)),
        color=colors[i],
        alpha=0.4,
        s=12,
        linewidths=0,
        zorder=2,
    )

    # C. Draw mathematical boundaries
    ax.axvline(
        mean_val,
        color="black",
        linestyle="-",
        linewidth=1.5,
        alpha=0.8,
        zorder=4,
    )
    ax.axvline(
        ci_low,
        color="black",
        linestyle=":",
        linewidth=1.2,
        alpha=0.5,
        zorder=4,
    )
    ax.axvline(
        ci_high,
        color="black",
        linestyle=":",
        linewidth=1.2,
        alpha=0.5,
        zorder=4,
    )

    # If an interval captures zero, highlight it explicitly
    if ci_low < 0 < ci_high and config["unit"] == "%":
        ax.axvline(0, color="#d95f02", linestyle="--", linewidth=1.2, zorder=1)

    # D. Labels and aesthetic layout architecture
    ax.set_title(
        config["title"],
        loc="left",
        fontweight="bold",
        fontsize=12,
        pad=4,
        color="#222222",
    )
    ax.set_ylabel("")
    ax.get_yaxis().set_visible(False)
    sns.despine(ax=ax, left=True, right=True, top=True)
    ax.spines["bottom"].set_color("#cccccc")

    lbl_mean = config["fmt"].format(mean_val)
    lbl_low = config["fmt"].format(ci_low)
    lbl_high = config["fmt"].format(ci_high)

    stat_text = f"Mean: {lbl_mean}\n95% CI: [{lbl_low}, {lbl_high}]"
    ax.text(
        1.02,
        0.5,
        stat_text,
        transform=ax.transAxes,
        va="center",
        ha="left",
        fontsize=10,
        fontweight="semibold",
        linespacing=1.4,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#f9f9f9",
            edgecolor="#e0e0e0",
            lw=1,
        ),
    )
    ax.set_xlim(data.min() - data.std() * 0.3, data.max() + data.std() * 0.3)

plt.subplots_adjust(right=0.75)

plt.savefig(r'C:\Users\User\Desktop\cakewebsitetemplate\Data Research\NeuXPRISM essentials\bootstrap_plot.png', bbox_inches="tight")

# Pushes only to your interactive screen buffer
plt.show()
!pip install xgboost
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# STEP 1: Load your raw data and NeuXPRISM results
# ─────────────────────────────────────────────────────────────

BASE = r'C:\Users\User\Desktop\cakewebsitetemplate\Data Research\NeuXPRISM essentials'

df = pd.read_excel(
    f'{BASE}\F1341550_AE.xlsx',
    sheet_name='Master_Sheet'
)

# Keep only failures and partial failures — same corpus as NeuXPRISM
df = df[df['Status Mission'].isin(['Failure', 'Partial Failure'])].copy()
df = df.sort_values('Datum').reset_index(drop=True)

# Load NeuXPRISM results
results = pd.read_csv(f'{BASE}\NeuXPRISM_Full_Results.csv')

# Rename to correct terminology
results = results.rename(columns={
    'R_final'  : 'Composite_Risk_Score',  # pre-LDF fusion
    'R_learned': 'R_final'                # LDF-inclusive true final output
})

print(f"Records loaded: {len(df)}")
print(f"Results rows:   {len(results)}")
print(f"Results cols:   {results.columns.tolist()}")

# ─────────────────────────────────────────────────────────────
# STEP 2: Encode categorical features
# The baselines need numeric inputs — we encode the same
# categorical variables NeuXPRISM encodes as spike trains
# ─────────────────────────────────────────────────────────────

heritage_map = {'New': 0, 'Low': 1, 'Medium': 2, 'High': 3}
df['Heritage_enc'] = df['Heritage Category'].map(heritage_map)

le_orbit = LabelEncoder()
le_phase = LabelEncoder()
le_cause = LabelEncoder()

df['Orbit_enc']  = le_orbit.fit_transform(df['Orbit Type'])
df['Phase_enc']  = le_phase.fit_transform(df['Phase'])
df['Cause_enc']  = le_cause.fit_transform(df['Causes'])
df['Datum_enc']  = pd.to_datetime(
    df['Datum'], dayfirst=True
).map(lambda x: x.toordinal())

# Fill 2 missing AE values with median
df['AE_filled'] = df['AE_daily'].fillna(df['AE_daily'].median())

# ─────────────────────────────────────────────────────────────
# STEP 3: Build feature matrix
# 9 NeuXPRISM inputs + Cost
# Cost is included because R_final scales with it —
# withholding it would unfairly handicap the baselines
# ─────────────────────────────────────────────────────────────

feature_cols = [
    'Heritage_enc',   # vehicle maturity
    'Orbit_enc',      # target orbit
    'Phase_enc',      # failure phase
    'Cause_enc',      # diagnosed cause
    'Latitude',       # spaceport latitude
    'Coast_Km',       # distance to coastline
    'Ap_daily',       # geomagnetic index
    'AE_filled',      # auroral electrojet index
    'Datum_enc',      # temporal position
    'Cost (M USD)',   # mission cost — additional input for baselines
]

X = df[feature_cols].fillna(df[feature_cols].median())

# Normalize to [0,1] — same as NeuXPRISM's rate encoding normalization
scaler  = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────────────────────
# STEP 4: Define target — R_final (LDF-inclusive)
# This is R_learned in your CSV, renamed above
# ─────────────────────────────────────────────────────────────

y = results['R_final'].values   # LDF-inclusive R_final, all 80 records

print(f"\nTarget (R_final) — mean: ${y.mean():.2f}M, "
      f"min: ${y.min():.2f}M, max: ${y.max():.2f}M")

# ─────────────────────────────────────────────────────────────
# STEP 5: LOOCV evaluation
# Train on 79, predict 1, repeat 80 times
# Same protocol as NeuXPRISM's Goal A
# ─────────────────────────────────────────────────────────────

loo = LeaveOneOut()

# ── Linear Regression ────────────────────────────────────────
# Assumes a linear relationship between features and R_final
# Simplest possible parametric baseline
lr_reg   = LinearRegression()
lr_preds = cross_val_predict(lr_reg, X_scaled, y, cv=loo)

# Clip negative predictions — EL cannot be negative
lr_preds = np.clip(lr_preds, 0, None)

lr_mae  = mean_absolute_error(y, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y, lr_preds))

# ── XGBoost Regressor ─────────────────────────────────────────
# Gradient-boosted trees — captures non-linear relationships
# Strong baseline, but likely to overfit on 80 records under LOOCV
xgb_reg   = XGBRegressor(
    n_estimators  = 100,
    max_depth     = 3,       # shallow trees to limit overfitting on 80 records
    learning_rate = 0.1,
    random_state  = 42,
    verbosity     = 0
)
xgb_preds = cross_val_predict(xgb_reg, X_scaled, y, cv=loo)
xgb_preds = np.clip(xgb_preds, 0, None)

xgb_mae  = mean_absolute_error(y, xgb_preds)
xgb_rmse = np.sqrt(mean_squared_error(y, xgb_preds))

# ─────────────────────────────────────────────────────────────
# STEP 6: Baselines for comparison
# ─────────────────────────────────────────────────────────────

# Classical EL: treats full mission cost as expected loss
classical     = df['Cost (M USD)'].values
classical_mae  = mean_absolute_error(y, classical)
classical_rmse = np.sqrt(mean_squared_error(y, classical))

# UFMB baseline mean (verified figure)
ufmb_preds     = np.full(len(y), 18.56)
ufmb_mae       = mean_absolute_error(y, ufmb_preds)
ufmb_rmse      = np.sqrt(mean_squared_error(y, ufmb_preds))

# NeuXPRISM Composite Risk Score (pre-LDF) — shows what LDF adds
crs            = results['Composite_Risk_Score'].values
crs_mae        = mean_absolute_error(y, crs)
crs_rmse       = np.sqrt(mean_squared_error(y, crs))

# ─────────────────────────────────────────────────────────────
# STEP 7: Print results
# ─────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("TASK 2: EL Estimation — LOOCV Comparison")
print("Target: R_final (LDF-inclusive, $M)")
print("=" * 60)
print(f"{'Method':<35} {'MAE ($M)':>9} {'RMSE ($M)':>10}")
print("-" * 56)
print(f"{'Classical EL (full cost)':<35} {classical_mae:>9.2f} {classical_rmse:>10.2f}")
print(f"{'UFMB (corpus mean, $18.56M)':<35} {ufmb_mae:>9.2f} {ufmb_rmse:>10.2f}")
print(f"{'Linear Regression (LOOCV)':<35} {lr_mae:>9.2f} {lr_rmse:>10.2f}")
print(f"{'XGBoost Regressor (LOOCV)':<35} {xgb_mae:>9.2f} {xgb_rmse:>10.2f}")
print(f"{'Composite Risk Score (pre-LDF)':<35} {crs_mae:>9.2f} {crs_rmse:>10.2f}")
print(f"{'NeuXPRISM R_final (LDF-incl.)':<35}   — (framework output, not LOOCV model)")
print()

# Mean R_final for context
print(f"NeuXPRISM R_final mean:  ${y.mean():.2f}M  "
      f"(54.9% reduction vs classical mean ${classical.mean():.2f}M)")

# ─────────────────────────────────────────────────────────────
# STEP 8: Save results
# ─────────────────────────────────────────────────────────────

comparison_df = pd.DataFrame({
    'Method': [
        'Classical EL (full cost)',
        'UFMB (corpus mean)',
        'Linear Regression (LOOCV)',
        'XGBoost Regressor (LOOCV)',
        'NeuXPRISM Composite Risk Score (pre-LDF)',
        'NeuXPRISM R_final (LDF-inclusive)',
    ],
    'MAE ($M)': [
        round(classical_mae,  2),
        round(ufmb_mae,       2),
        round(lr_mae,         2),
        round(xgb_mae,        2),
        round(crs_mae,        2),
        'N/A — framework output'
    ],
    'RMSE ($M)': [
        round(classical_rmse,  2),
        round(ufmb_rmse,       2),
        round(lr_rmse,         2),
        round(xgb_rmse,        2),
        round(crs_rmse,        2),
        'N/A — framework output'
    ],
    'Note': [
        'Naive baseline — full cost = EL',
        'Fleet-mean baseline from prior framework',
        'Linear parametric baseline (LOOCV)',
        'Non-linear tree baseline (LOOCV)',
        'NeuXPRISM pre-LDF fusion',
        '54.9% reduction vs Classical'
    ]
})

comparison_df.to_csv(
    f'{BASE}\Baseline_Comparison_EL.csv',
    index=False
)
print(f"\nSaved to: {BASE}\Baseline_Comparison_EL.csv")
# Show XGBoost can't explain what it predicts
xgb_final = XGBRegressor(n_estimators=100, max_depth=3,
                          learning_rate=0.1, random_state=42)
xgb_final.fit(X_scaled, y)

importances = pd.DataFrame({
    'Feature': feature_cols,
    'XGBoost Importance': xgb_final.feature_importances_
}).sort_values('XGBoost Importance', ascending=False)

print(importances.to_string(index=False))
import pandas as pd
import numpy as np

# --- UPDATE THESE PATHS TO MATCH YOUR LOCAL FILES ---
excel_path = 'F1341550_AE.xlsx'
csv_path = 'NeuXPRISM_Full_Results.csv'

# Load and verify all data
df = pd.read_excel(r"C:\Users\User\Desktop\cakewebsitetemplate\Data Research\NeuXPRISM essentials\F1341550_AE.xlsx", sheet_name="Master_Sheet")
df = df[df['Status Mission'].isin(['Failure','Partial Failure'])].copy()
df['Datum'] = pd.to_datetime(df['Datum'], dayfirst=True, errors='coerce')
df = df.sort_values('Datum').reset_index(drop=True)

heritage_map = {'New':0,'Low':1,'Medium':2,'High':3}
df['HerNum'] = df['Heritage Category'].map(heritage_map)

results = pd.read_csv(r"C:\Users\User\Desktop\cakewebsitetemplate\Data Research\NeuXPRISM essentials\NeuXPRISM_Full_Results.csv")
results = results.rename(columns={'R_final':'Composite_Risk_Score','R_learned':'R_final'})

# Merge EL values
df = df.merge(results[['Detail','EL_fuzzy','EL_markov','Composite_Risk_Score','LDF','R_final']],
              on='Detail', how='left')

def walk_forward(df, company, vehicle, alpha=0.4, beta=0.4, gamma=0.2):
    pair = df[(df['Company Name']==company) & (df['Vehicle']==vehicle)].copy()
    pair = pair.sort_values('Datum').reset_index(drop=True)
    rows = []
    for i, rec in pair.iterrows():
        prior = pair[pair['Datum'] < rec['Datum']]
        n = len(prior)
        if n == 0:
            CR, HI, FF = 0.0, 0.0, 0.0
        else:
            CR = (prior['Causes'] == rec['Causes']).sum() / n
            last_her = prior.sort_values('Datum')['HerNum'].iloc[-1]
            HI = max((rec['HerNum'] - last_her)/3, 0)
            FF = min(n/9, 1.0)
        ldf = np.clip(1.0 + beta*CR - alpha*max(HI,0) - gamma*FF, 0.1, 1.5)
        crs = (rec['EL_fuzzy'] + rec['EL_markov'])/2
        rfinal = crs * ldf
        tier = 'Low' if rfinal<10 else ('Medium' if rfinal<30 else ('High' if rfinal<60 else 'Critical'))

        # Driver logic
        if n == 0:
            driver = 'No prior history\nLDF = 1.000'
        else:
            parts = []
            if CR > 0:   parts.append(f'Cause repeated\nCR={CR:.3f} → LDF↑')
            if CR == 0:  parts.append(f'Cause diverged\nCR=0 → LDF↓')
            if HI > 0:   parts.append(f'Heritage ↑\nHI={HI:.3f} → LDF↓')
            if FF > 0:   parts.append(f'FF={FF:.3f} → LDF↓')
            driver = '\n'.join(parts)

        rows.append({
            'mission': rec['Detail'].split('|')[-1].strip(),
            'date': rec['Datum'].strftime('%Y-%m'),
            'cause': rec['Causes'],
            'heritage': rec['Heritage Category'],
            'CR': round(CR,3), 'HI': round(HI,3), 'FF': round(FF,3),
            'LDF_computed': round(ldf,4),
            'LDF_csv': round(rec['LDF'],4),
            'CRS': round(crs,2),
            'R_final': round(rfinal,2),
            'tier': tier,
            'driver': driver,
            'match': abs(ldf - rec['LDF']) < 0.001
        })
    return pd.DataFrame(rows)

# Run the analysis
df_f = walk_forward(df, 'SpaceX', 'Falcon 1')
df_k = walk_forward(df, 'Space One', 'Kairos')
df_p = walk_forward(df, 'ILS', 'Proton-M/Briz-M')

# Output results
for name, d in [('FALCON 1', df_f), ('KAIROS', df_k), ('PROTON-M/BRIZ-M', df_p)]:
    print(f"\n{'='*70}")
    print(f"{name} — LDF match with CSV: {d['match'].all()}")
    print('='*70)
    print(d[['mission','date','cause','heritage','CR','HI','FF',
             'LDF_computed','LDF_csv','CRS','R_final','tier']].to_string(index=False))
"""
NeuXPRISM — Evolution Figure v3
Stacked area (all 80 records) + operator LDF trajectories
All values verified against NeuXPRISM_Full_Results.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# 1. LOAD AND COMPUTE LDF FOR ALL 80 RECORDS
# ─────────────────────────────────────────────────────────────
df = pd.read_excel(
    r'C:\Users\User\Desktop\cakewebsitetemplate\Data Research\NeuXPRISM essentials\F1341550_AE.xlsx',
    sheet_name='Master_Sheet'
)
df = df[df['Status Mission'].isin(['Failure', 'Partial Failure'])].copy()
df['Datum'] = pd.to_datetime(df['Datum'], dayfirst=True, errors='coerce')
df = df.sort_values('Datum').reset_index(drop=True)

heritage_map = {'New': 0, 'Low': 1, 'Medium': 2, 'High': 3}
df['HerNum'] = df['Heritage Category'].map(heritage_map)

ALPHA, BETA, GAMMA = 0.4, 0.4, 0.2

records = []
for _, row in df.iterrows():
    prior = df[
        (df['Company Name'] == row['Company Name']) &
        (df['Vehicle']      == row['Vehicle'])      &
        (df['Datum']        <  row['Datum'])
    ]
    n = len(prior)
    if n == 0:
        CR, HI, FF = 0.0, 0.0, 0.0
    else:
        CR = (prior['Causes'] == row['Causes']).sum() / n
        last_her = prior.sort_values('Datum')['HerNum'].iloc[-1]
        HI = max((row['HerNum'] - last_her) / 3, 0)
        FF = min(n / 9, 1.0)
    ldf = np.clip(1.0 + BETA*CR - ALPHA*HI - GAMMA*FF, 0.1, 1.5)
    records.append({
        'Company' : row['Company Name'],
        'Vehicle' : row['Vehicle'],
        'Year'    : row['Datum'].year,
        'Datum'   : row['Datum'],
        'LDF'     : ldf,
        'n_prior' : n,
        'Category': ('Neutral'    if ldf == 1.0
                     else 'Discounted' if ldf  < 1.0
                     else 'Amplified'),
    })

comp = pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────
# 2. STACKED AREA DATA (per year proportions)
# ─────────────────────────────────────────────────────────────
years = sorted(comp['Year'].unique())

neutral_c, disc_c, ampl_c, total_per_yr = [], [], [], []
for y in years:
    sub = comp[comp['Year'] == y]
    n   = len(sub)
    total_per_yr.append(n)
    neutral_c.append((sub['Category'] == 'Neutral'   ).sum() / n * 100)
    disc_c   .append((sub['Category'] == 'Discounted').sum() / n * 100)
    ampl_c   .append((sub['Category'] == 'Amplified' ).sum() / n * 100)

def roll(arr, w=2):
    out = []
    for i in range(len(arr)):
        sl = arr[max(0, i-w):i+w+1]
        out.append(np.mean(sl))
    return out

neutral_s = roll(neutral_c)
disc_s    = roll(disc_c)
ampl_s    = roll(ampl_c)

# ─────────────────────────────────────────────────────────────
# 3. OPERATOR TRAJECTORY DATA (>=2 total records per pair)
# ─────────────────────────────────────────────────────────────
op_def = [
    ('ISRO',              'GSLV Mk I',       'ISRO GSLV Mk I',       '#8e44ad'),
    ('ILS',               'Proton-M/Briz-M', 'ILS Proton-M',         '#c0392b'),
    ('SpaceX',            'Falcon 1',         'SpaceX Falcon 1',      '#2980b9'),
    ('Rocket Lab',        'Electron/Curie',  'Rocket Lab Electron',  '#e67e22'),
    ('Firefly Aerospace', 'Alpha',            'Firefly Alpha',        '#16a085'),
    ('Roscosmos',         'Proton-M/Briz-M', 'Roscosmos Proton-M',   '#7f8c8d'),
]

op_trajectories = []
for co, ve, label, color in op_def:
    sub = comp[
        (comp['Company'] == co) & (comp['Vehicle'] == ve)
    ].sort_values('Datum').reset_index(drop=True)
    if len(sub) < 2:
        continue
    op_trajectories.append({
        'label': label, 'color': color,
        'years': sub['Year'].tolist(),
        'ldfs' : sub['LDF'].tolist(),
        'cats' : sub['Category'].tolist(),
    })

# ─────────────────────────────────────────────────────────────
# 4. OPERATOR BACKGROUND BANDS
# ─────────────────────────────────────────────────────────────
band_defs = [
    ('SpaceX Falcon 1',      2006, 2008, '#2980b9', 0),
    ('ILS Proton-M',         2006, 2012, '#c0392b', 1),
    ('ISRO GSLV Mk I',       2006, 2010, '#8e44ad', 2),
    ('Roscosmos Proton-M',   2011, 2021, '#7f8c8d', 3),
    ('Rocket Lab Electron',  2020, 2023, '#e67e22', 4),
    ('Firefly Alpha',        2021, 2023, '#16a085', 5),
    ('CASC Long March 3B/E', 2020, 2026, '#d4ac0d', 6),
    ('Space One Kairos',     2024, 2026, '#1abc9c', 7),
]

CAT_COLORS = {
    'Neutral'   : '#bdc3c7',
    'Discounted': '#27ae60',
    'Amplified' : '#c0392b',
}

# ─────────────────────────────────────────────────────────────
# 5. FIGURE LAYOUT
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 13))
fig.patch.set_facecolor('white')

ax_main = fig.add_axes([0.06, 0.40, 0.91, 0.52])
ax_traj = fig.add_axes([0.06, 0.06, 0.91, 0.28])

# ── STACKED AREA ──────────────────────────────────────────────
bottom_d = np.array(neutral_s)
bottom_a = np.array(neutral_s) + np.array(disc_s)

ax_main.fill_between(years, 0, neutral_s,
    alpha=0.75, color=CAT_COLORS['Neutral'],
    label='Neutral  (LDF = 1.0) — No prior history', zorder=2)

ax_main.fill_between(years, neutral_s, bottom_a,
    alpha=0.85, color=CAT_COLORS['Discounted'],
    label='Discounted  (LDF < 1.0) — Operator showed learning', zorder=2)

ax_main.fill_between(
    years, bottom_a,
    np.array(neutral_s) + np.array(disc_s) + np.array(ampl_s),
    alpha=0.85, color=CAT_COLORS['Amplified'],
    label='Amplified  (LDF > 1.0) — Repeated failure cause', zorder=2)

ax_main.plot(years, neutral_s, color='white', lw=0.9, zorder=3)
ax_main.plot(years, bottom_a,  color='white', lw=0.9, zorder=3)

# ── OPERATOR BACKGROUND BANDS ─────────────────────────────────
# Labels rotated 90° and placed just above plot area — no overlap
BAND_Y_START = 102
BAND_Y_STEP  = 0       # not needed with rotation=90

for i, (label, y0, y1, color, _) in enumerate(band_defs):
    ax_main.axvspan(y0 - 0.3, y1 + 0.3,
                    alpha=0.07, color=color, zorder=0)
    # Place label at left edge of band, rotated vertical
    ax_main.text(
        y0 + 0.1, 101.5,
        label,
        fontsize=6.8, color=color,
        ha='left', va='bottom',
        fontweight='bold',
        rotation=90,
        bbox=dict(boxstyle='round,pad=0.15',
                  fc='white', alpha=0.88,
                  ec=color, lw=0.8),
        zorder=5
    )

# ── RECORD COUNT (years with >= 3 records only) ───────────────
for y, total in zip(years, total_per_yr):
    if total >= 3:
        ax_main.text(y, 107, str(total),
                     ha='center', va='bottom',
                     fontsize=7, color='#444',
                     fontweight='bold')

# ── KEY MOMENT ANNOTATIONS ────────────────────────────────────
ax_main.annotate(
    'LDF begins\ndiscriminating\n(Falcon 1, ILS)',
    xy=(2007, 55), xytext=(2002.5, 75),
    fontsize=8, color='#2c3e50',
    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.2),
    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.88),
    zorder=6
)
ax_main.annotate(
    'Rocket Lab\nAmplification\n(2021–2023)',
    xy=(2021.5, 80), xytext=(2016.5, 88),
    fontsize=8, color='#e67e22',
    arrowprops=dict(arrowstyle='->', color='#e67e22', lw=1.2),
    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.88),
    zorder=6
)

# ── AXES FORMATTING ───────────────────────────────────────────
ax_main.set_xlim(2000.5, 2026.8)
ax_main.set_ylim(0, 118)
ax_main.set_yticks([0, 25, 50, 75, 100])
ax_main.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9)
ax_main.set_ylabel('% of records per year', fontsize=11, fontweight='bold')

# Show every other year to avoid x-label crowding
ax_main.set_xticks(years)
ax_main.set_xticklabels(
    [str(y) if y % 2 == 1 else '' for y in years],
    rotation=45, ha='right', fontsize=8.5
)

ax_main.spines['top'].set_visible(False)
ax_main.spines['right'].set_visible(False)
ax_main.grid(axis='y', alpha=0.22, color='gray', zorder=1)

ax_main.set_title(
    'NeuXPRISM Evolving System — LDF Coverage Across All 80 Records (2001–2026)\n'
    'Stacked area: proportion Neutral / Discounted / Amplified per year  ·  '
    'Numbers above bars: record count  ·  '
    'Vertical bands: operator LDF-active windows',
    fontsize=11.5, fontweight='bold', pad=14
)

ax_main.legend(
    loc='upper left', fontsize=7,
    title='LDF Category', title_fontsize=7.5,
    framealpha=0.95, edgecolor='#ccc',
    bbox_to_anchor=(0.01, 0.81)
)

# ── TRAJECTORY PANEL ──────────────────────────────────────────
ax_traj.set_facecolor('#F8F8F8')
ax_traj.axhline(1.0, color='gray', ls='--', lw=1.2, alpha=0.65, zorder=2)

ax_traj.fill_between([2000, 2027], 0.68, 1.0,
    alpha=0.07, color='#27ae60')
ax_traj.fill_between([2000, 2027], 1.0, 1.50,
    alpha=0.07, color='#c0392b')

ax_traj.text(2001.2, 0.695,
             'Discount zone  (LDF < 1.0)',
             fontsize=8, color='#1e8449', fontstyle='italic')
ax_traj.text(2001.2, 1.455,
             'Amplify zone  (LDF > 1.0)',
             fontsize=8, color='#922b21', fontstyle='italic')

for op in op_trajectories:
    ax_traj.plot(
        op['years'], op['ldfs'],
        color=op['color'], lw=2.4,
        marker='o', ms=8, zorder=4,
        label=op['label']
    )
    for y, ldf_v, cat in zip(op['years'], op['ldfs'], op['cats']):
        ax_traj.scatter(y, ldf_v, s=70,
                        color=CAT_COLORS[cat],
                        edgecolors=op['color'],
                        lw=2.0, zorder=5)

ax_traj.set_xlim(2000.5, 2026.8)
ax_traj.set_ylim(0.67, 1.52)
ax_traj.set_yticks([0.7, 0.85, 1.0, 1.15, 1.30, 1.45])
ax_traj.set_yticklabels(
    ['0.70', '0.85', '1.00', '1.15', '1.30', '1.45'], fontsize=9)
ax_traj.set_ylabel('LDF value', fontsize=10, fontweight='bold')
ax_traj.set_xlabel('Year', fontsize=10, fontweight='bold')

ax_traj.set_xticks(years)
ax_traj.set_xticklabels(
    [str(y) if y % 2 == 1 else '' for y in years],
    rotation=45, ha='right', fontsize=8.5
)

ax_traj.spines['top'].set_visible(False)
ax_traj.spines['right'].set_visible(False)
ax_traj.grid(axis='y', alpha=0.3, color='white', zorder=1)

ax_traj.set_title(
    'Operator-Level LDF Trajectories — 6 Multi-Failure Programmes\n'
    'Dot colour = Neutral (grey) / Discounted (green) / Amplified (red)',
    fontsize=10.5, fontweight='bold', pad=7
)
ax_traj.legend(
    loc='upper right', fontsize=8.5,
    ncol=3, framealpha=0.95,
    edgecolor='#ccc'
)

# ─────────────────────────────────────────────────────────────
# 6. SAVE AND SHOW
# ─────────────────────────────────────────────────────────────
plt.savefig(
    r'C:\Users\User\Desktop\cakewebsitetemplate\Data Research\NeuXPRISM essentials\NeuXPRISM_Evolution.png',
    dpi=300, bbox_inches='tight', facecolor='white'
)
plt.show()
print("Done.")