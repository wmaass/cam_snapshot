import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
import io
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from IPython.display import display, Image

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



# # Kategorische und numerische Features definieren
# cat_features = ['Hospital Service Area', 'Hospital County', 'Facility Name', 'Age Group',
#                 'Zip Code', 'Gender', 'Race', 'Ethnicity', 'Patient Disposition',
#                 'CCS Diagnosis Description', 'CCS Procedure Description', 'APR DRG Code',
#                 'APR DRG Description', 'APR MDC Code', 'APR MDC Description',
#                 'APR Severity of Illness Code', 'APR Severity of Illness Description',
#                 'APR Risk of Mortality', 'APR Medical Surgical Description',
#                 'Payment Typology 1', 'Payment Typology 2', 'Payment Typology 3',
#                 'Emergency Department Indicator', 'Operating Certificate Number',
#                 'Permanent Facility Id', 'CCS Diagnosis Code', 'CCS Procedure Code', 'Type of Admission']

# num_features = [col for col in df.columns if col not in cat_features and col != target and col not in del_feature_names]

# # Kategorien nach Mediankosten ranken und ersetzen
# print("\U0001F522 Ranke Kategorien nach Mediankosten...")
# def rank_categories_inplace(df, cat_features, target_name=target):
#     for feature in cat_features:
#         if feature in df.columns and df[feature].nunique() > 1:
#             try:
#                 median_costs = df.groupby(feature)[target_name].median().sort_values()
#                 rankings = {cat: rank for rank, cat in enumerate(median_costs.index, 1)}
#                 df[feature] = df[feature].map(rankings).fillna(0).astype(np.int32)
#             except Exception as e:
#                 print(f"⚠️ Fehler beim Ranken von {feature}: {e}")
#         else:
#             df[feature] = 0

# rank_categories_inplace(df, cat_features)

# # Finales Feature-Set
# feature_columns = num_features + cat_features

# Train/Test aus CSV-Dateien laden
print("📂 Lade vorbereitete Trainings- und Testdaten...")
train = pd.read_csv(path_df_train)
test = pd.read_csv(path_df_test)

X_train = train.drop(columns=[target]).astype(np.float32)
y_train = train[target]
X_test = test.drop(columns=[target]).astype(np.float32)
y_test = test[target]


# Modell laden
print(f"\U0001F4BE Lade gespeichertes Modell aus: {path_best_model}")
model = xgb.XGBRegressor()
model.load_model(path_best_model)

# Modellbewertung
print("\U0001F4CA Evaluiere Modell...")
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\n📈 Modellleistung:")
print(f"RMSE: {rmse:,.2f}")
print(f"R²: {r2:.4f}")

# SHAP TreeExplainer verwenden
print("\U0001F50D Berechne SHAP-Werte mit TreeExplainer...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP-Werte DataFrame
print("\U0001F4BE Speichere SHAP-Werte...")
shap_values_df = pd.DataFrame(shap_values, columns=X_test.columns, index=X_test.index)
shap_values_df.to_csv(path_shap_values)

# SHAP Summary Plot speichern und anzeigen
print("\U0001F4CA Erzeuge SHAP Summary Plot...")
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(path_summary_plot, dpi=300)
plt.show()
print(f"\n✅ SHAP Summary Plot gespeichert unter: {path_summary_plot}")

# Analyse der wichtigsten Merkmale
importances = pd.Series(np.abs(shap_values).mean(axis=0), index=X_test.columns)
print("\n🔝 Top 10 Merkmale nach mittlerer SHAP-Bedeutung:")
print(importances.sort_values(ascending=False).head(10))

# Statistische Zusammenfassung
print("\n📊 Summary Statistics:")
combined_rankings = shap_values_df.abs()
combined_rankings = combined_rankings.melt(var_name='Feature', value_name='SHAP')
combined_rankings['Count'] = 1
combined_rankings = combined_rankings.groupby(['Feature', 'SHAP']).agg({'Count': 'sum'}).reset_index()
print(f"Total number of features: {combined_rankings['Feature'].nunique()}")
print(f"Total number of categories: {len(combined_rankings)}")
print(f"Average number of categories per feature: {len(combined_rankings) / combined_rankings['Feature'].nunique():.2f}")
print(f"Total rows in dataset: {combined_rankings['Count'].sum()}")
print(f"Average count per category: {combined_rankings['Count'].mean():.2f}")
print(f"Median count per category: {combined_rankings['Count'].median():.2f}")

print("\n✅ Analyse abgeschlossen.")