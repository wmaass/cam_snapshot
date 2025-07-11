import pandas as pd
import numpy as np
import xgboost
print(f"XGBoost version: {xgboost.__version__}")
print(f"Using class: {xgboost.XGBRegressor}")
from xgboost import XGBRegressor


import shap
import matplotlib.pyplot as plt
import os
import io
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from optuna.integration import XGBoostPruningCallback
from optuna.samplers import TPESampler
from optuna.integration import XGBoostPruningCallback


# Suppress warnings
warnings.filterwarnings('ignore')

# Datenpfade und Konfiguration
app_name = "HIS17-two-classes"
path_data = '../data/'
d_name = "hdma_1M.csv"
target = "Total Costs"
leaky_features = 'Total Charges'
del_feature_names = ['Discharge Year', 'Birth Weight', 'Abortion Edit Indicator']

path_ranked = f'./results/{app_name}/ranked/'
os.makedirs(path_ranked, exist_ok=True)

path_df_original = f'{path_ranked}/df_original.csv'
path_df_train = f'{path_ranked}/train.csv'
path_df_test = f'{path_ranked}/test.csv'
path_rank_encodings = f'{path_ranked}/rank_encoded.csv'
path_shap_values = f'{path_ranked}/shap_values_df.csv'
path_best_model = f'{path_ranked}/best_model.json'
path_summary_plot = f'{path_ranked}/shap_summary_plot.png'

# Daten laden und vorbereiten
print("\U0001F4E5 Lade Daten...")
df = pd.read_csv(os.path.join(path_data, d_name), delimiter=',', quotechar='"', encoding='latin1')
#df = df.sample(n=10000, random_state=42)
df = df.drop([leaky_features], axis=1)
df['Length of Stay'] = pd.to_numeric(df['Length of Stay'], errors='coerce')
df.to_csv(path_df_original, index=False)

# Kategorische und numerische Features definieren
cat_features = ['Hospital Service Area', 'Hospital County', 'Facility Name', 'Age Group',
                'Zip Code', 'Gender', 'Race', 'Ethnicity', 'Patient Disposition',
                'CCS Diagnosis Description', 'CCS Procedure Description', 'APR DRG Code',
                'APR DRG Description', 'APR MDC Code', 'APR MDC Description',
                'APR Severity of Illness Code', 'APR Severity of Illness Description',
                'APR Risk of Mortality', 'APR Medical Surgical Description',
                'Payment Typology 1', 'Payment Typology 2', 'Payment Typology 3',
                'Emergency Department Indicator', 'Operating Certificate Number',
                'Permanent Facility Id', 'CCS Diagnosis Code', 'CCS Procedure Code', 'Type of Admission']

num_features = [col for col in df.columns if col not in cat_features and col != target and col not in del_feature_names]

# Kategorien nach Mediankosten ranken und ersetzen
print("\U0001F522 Ranke Kategorien nach Mediankosten...")
def rank_categories_inplace(df, cat_features, target_name=target):
    for feature in cat_features:
        if feature in df.columns and df[feature].nunique() > 1:
            try:
                median_costs = df.groupby(feature)[target_name].median().sort_values()
                rankings = {cat: rank for rank, cat in enumerate(median_costs.index, 1)}
                df[feature] = df[feature].map(rankings).fillna(0).astype(np.int32)
            except Exception as e:
                print(f"⚠️ Fehler beim Ranken von {feature}: {e}")
        else:
            df[feature] = 0

rank_categories_inplace(df, cat_features)

# Finales Feature-Set
feature_columns = num_features + cat_features

# Splitten und speichern
print("✂️ Splitte Daten in Train/Test und speichere...")
train, test = train_test_split(df[feature_columns + [target]], test_size=0.2, random_state=42)
train.to_csv(path_df_train, index=False)
test.to_csv(path_df_test, index=False)

X_train = train.drop(columns=[target]).astype(np.float32)
y_train = train[target]
X_test = test.drop(columns=[target]).astype(np.float32)
y_test = test[target]

# Optuna Optimierung
print("\U0001F9EA Starte Hyperparameter-Optimierung mit Optuna...")

def objective(trial):
    params = {
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'n_estimators': trial.suggest_int('n_estimators', 80, 150),
        'max_depth': trial.suggest_int('max_depth', 4, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'random_state': 42,
        'eval_metric': 'rmse',
    }

    model = XGBRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False
    )

    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds, squared=False)


study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=30)

print(f"\n✅ Beste Parameter: {study.best_params}")

# Modell mit besten Parametern trainieren
print("\U0001F680 Trainiere bestes Modell mit Optuna-Parametern...")

best_model = XGBRegressor(**study.best_params)

best_model.fit(X_train, y_train)

# Modell speichern
print(f"\U0001F4BE Speichere Modell unter: {path_best_model}")
best_model.save_model(path_best_model)

# Modellbewertung
print("\U0001F4CA Evaluiere Modell...")
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\n📈 Modellleistung:")
print(f"RMSE: {rmse:,.2f}")
print(f"R²: {r2:.4f}")

print("\n✅ Analyse abgeschlossen.")
