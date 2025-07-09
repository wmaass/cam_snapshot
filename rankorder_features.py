import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import warnings
import GPUtil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    try:
        app_names = ["HIS17-random"]  # ["HIS17", "HIS17-2", "HIS17-two-classes"]

        # Test GPU
        GPUs = GPUtil.getGPUs()
        print(GPUs)

        if GPUs:
            path_data = '../../data/'
        else:
            path_data = '~/Dropbox/CAData/'

        d_name = "hdma_1M.csv"
        d_name_short = "HIS17"
        df = pd.read_csv(path_data + d_name, delimiter=',', quotechar='"', encoding='latin1')

        # Sample 100,000 rows
        df = df.sample(n=100000, random_state=42)

        # Remove only Total Charges to prevent data leakage
        df = df.drop(['Total Charges'], axis=1)

        # Convert 'Length of Stay' to numeric if it's not already
        df['Length of Stay'] = pd.to_numeric(df['Length of Stay'], errors='coerce')

        cat_features = ['Hospital Service Area', 'Hospital County', 'Facility Name', 'Age Group',
                        'Zip Code', 'Gender', 'Race', 'Ethnicity', 'Type of Admission',
                        'Patient Disposition', 'CCS Diagnosis Description', 'CCS Procedure Description',
                        'APR DRG Description', 'APR MDC Code', 'APR MDC Description',
                        'APR Severity of Illness Code', 'APR Severity of Illness Description',
                        'APR Risk of Mortality', 'APR Medical Surgical Description', 'Payment Typology 1',
                        'Payment Typology 2', 'Payment Typology 3', 'Abortion Edit Indicator',
                        'Emergency Department Indicator', 'Operating Certificate Number',
                        'Permanent Facility Id', 'CCS Diagnosis Code', 'CCS Procedure Code', 'APR DRG Code']
        

        # drop 'Discarge Year' and 'Birth Weight' column
        df = df.drop(['Discharge Year', 'Birth Weight'], axis=1)

        # Identify numerical columns
        num_features = [col for col in df.columns if col not in cat_features and col != 'Total Costs']

        print("Categorical features:", cat_features)
        print("Numerical features:", num_features)

        # Encoding categorical features
        label_encoders = {}
        for feature in cat_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[feature] = df[feature].astype(str)
                df[feature] = le.fit_transform(df[feature])
                label_encoders[feature] = le

        # Prepare the dataset
        X = df[num_features + cat_features]
        y = df['Total Costs']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

        # print first 10 rows of X_train
        print("\nFirst 10 rows of X_train:")
        print(X_train.head(10))

        # Display network traffic

        # Train GBM model
        gbm = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=1234)
        gbm.fit(X_train, y_train)

        # Model performance
        y_pred = gbm.predict(X_test)
        print("Model Performance:")
        print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
        print(f"R^2 Score: {r2_score(y_test, y_pred)}")

        # Variable Importance
        feature_importance = gbm.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importance})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        print("\nVariable Importance (Top 20):")
        print(feature_importance_df.head(20))

        # Function to display plot
        def show_plot(plot_func, title):
            plt.figure(figsize=(12, 8))
            try:
                plot_func()
                plt.title(title)
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_data = buf.getvalue()
                with open(f"{title}.png", "wb") as f:
                    f.write(img_data)
            except Exception as e:
                print(f"Error generating plot {title}: {str(e)}")
            finally:
                plt.close()

        # Function to create ranking dataframe
        def create_ranking_df(encoded_feature, encoding_dict, df):
            original_feature = encoded_feature.replace('encoded_', '')
            rankings = encoding_dict[original_feature]

            ranking_df = pd.DataFrame(list(rankings.items()), columns=[original_feature, encoded_feature])
            ranking_df['Median_Total_Costs'] = ranking_df[original_feature].map(df.groupby(original_feature)['Total Costs'].median())

            # Handle any NaN values
            ranking_df['Median_Total_Costs'] = ranking_df['Median_Total_Costs'].fillna(df['Total Costs'].median())

            ranking_df = ranking_df.sort_values(encoded_feature)
            return ranking_df

    except Exception as e:
        print(f"An error occurred during the analysis: {str(e)}")

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
