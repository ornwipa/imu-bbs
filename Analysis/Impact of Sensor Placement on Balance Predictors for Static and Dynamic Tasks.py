

import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats
import numpy as np

# file_path = 'C:/Users/youss/Downloads/result/result2/result3.csv'
file_path = r'C:\Users\othamsuwan\OneDrive - ETS\etsmtl\papers\imu_balance\imu-bbs\result2.csv'

def load_and_preprocess_data(filepath):
    return pd.read_csv(filepath)

def refine_outlier_removal(data):
    # Remove outliers based on Z-scores
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    z_scores = np.abs(stats.zscore(data[numeric_columns], nan_policy='omit'))
    data['z_score_max'] = z_scores.max(axis=1)

    refined_data = pd.DataFrame()
    for task in data['Task'].unique():
        task_data = data[data['Task'] == task]
        z_score_threshold = 4

        while z_score_threshold < 10:
            retained_data = task_data[task_data['z_score_max'] < z_score_threshold]
            if len(retained_data) >= max(2, 0.1 * len(task_data)) or z_score_threshold >= 10:
                refined_data = pd.concat([refined_data, retained_data], ignore_index=True)
                break
            z_score_threshold += 0.5

    refined_data = refined_data.drop('z_score_max', axis=1)
    return refined_data

# Preprocess and refine outliers
data = load_and_preprocess_data(file_path)
data_clean = refine_outlier_removal(data)
# run up to this point to remove outliers


# Data cleaning and categorization
data_clean['Sensor'] = data_clean['Sensor'].str.lower().replace({'loweback': 'lowerback'}) # run this too
data_clean['BBS'] = data_clean['BBS'].fillna(0)

static_tasks = ['task2', 'task3', 'task6', 'task7', 'task13', 'task14']
dynamic_tasks = ['task1', 'task4', 'task5', 'task8', 'task9', 'task10', 'task11', 'task12']
data_clean['Task Type'] = data_clean['Task'].apply(lambda x: 'static' if x in static_tasks else 'dynamic')

def analyze_sensor_task_data_ols(sensor_type, task_type, data):
    # Analyze data using OLS regression
    sensor_task_data = data[(data['Sensor'] == sensor_type) & (data['Task Type'] == task_type)]
    features = sensor_task_data.drop(['Participant', 'Sensor', 'Task', 'BBS', 'Mean Acceleration X', 'RMS Acceleration', 'Ellipse Area', 'Box Area', 'Task Type'], axis=1)
    target = sensor_task_data['BBS']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
    features_scaled_df = sm.add_constant(features_scaled_df)

    # Align indices
    features_scaled_df = features_scaled_df.reset_index(drop=True)
    target = target.reset_index(drop=True)

    model = sm.OLS(target, features_scaled_df)
    result = model.fit()

    return result

results_task_sensor = {}
sensor_types = data_clean['Sensor'].unique()

# Perform OLS analysis for each sensor and task type
for sensor in sensor_types:
    for task_type in ['static', 'dynamic']:
        key = f"{sensor}-{task_type}"
        results_task_sensor[key] = analyze_sensor_task_data_ols(sensor, task_type, data_clean)

def print_summary_and_interpretation(results):
    for key, result in results.items():
        print(f"\nSensor-Task Combination: {key}")
        summary = result.summary2().tables[1]
        significant_features = summary[summary['P>|t|'] < 0.05]

        if significant_features.empty:
            print("No significant features found.")
        else:
            print(significant_features[['Coef.', 'Std.Err.', 'P>|t|', '[0.025', '0.975]']])
            for index, row in significant_features.iterrows():
                if index != 'const':
                    effect = "increase" if row['Coef.'] > 0 else "decrease"
                    implication = "better" if effect == "increase" else "worse"
                    print(f"- {index}: An {effect} in this feature is associated with an {implication} balance.")
                else:
                    print("- The constant term's significance indicates a baseline level for the BBS score across observations.")

print_summary_and_interpretation(results_task_sensor)



def save_summaries_to_excel(results, file_path='C:/Users/youss/Downloads/result/result2/coefficient_summary_of_the_6_data.xlsx'):
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        for key, result in results.items():
            summary_df = pd.DataFrame(result.summary2().tables[1])
            significant_features = summary_df[summary_df['P>|t|'] < 0.05]

            # Interpret significant features
            interpretations = []
            for index, row in significant_features.iterrows():
                if index != 'const':
                    effect = "increase" if row['Coef.'] > 0 else "decrease"
                    balance_implication = "better" if effect == "increase" else "worse"
                    interpretations.append(f"{index}: An {effect} in this feature is associated with a {balance_implication} balance.")
                else:
                    interpretations.append("The constant term's significance indicates a baseline level for the BBS score across observations.")
            interpretations_df = pd.DataFrame({'Interpretation': interpretations})

            # Save summary and interpretations to Excel
            summary_df.to_excel(writer, sheet_name=key[:31], startrow=0, startcol=0, index=True)
            interpretations_df.to_excel(writer, sheet_name=key[:31], startrow=len(summary_df)+2, startcol=0, index=False)

save_summaries_to_excel(results_task_sensor)
