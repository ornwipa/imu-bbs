import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import stats

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath) 
    return data

def refine_outlier_removal(data):
    # Remove outliers based on Z-scores
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    z_scores = np.abs(stats.zscore(data[numeric_columns], nan_policy='omit'))
    data['z_score_max'] = z_scores.max(axis=1) 

    refined_data = pd.DataFrame()
    initial_count = len(data)
    print(f"Initial data count: {initial_count}")

    for task in data['Task'].unique():
        task_data = data[data['Task'] == task]
        initial_task_count = len(task_data)
        z_score_threshold = 4

        while z_score_threshold < 10:
            retained_data = task_data[task_data['z_score_max'] < z_score_threshold]
            if len(retained_data) >= max(2, 0.1 * len(task_data)) or z_score_threshold >= 10:
                refined_data = pd.concat([refined_data, retained_data], ignore_index=True)
                break
            z_score_threshold += 0.5

        removed_count = initial_task_count - len(retained_data)
        removed_percentage = (removed_count / initial_task_count) * 100
        print(f"Task '{task}': {removed_percentage:.2f}% data removed.")

    refined_data.drop('z_score_max', axis=1, inplace=True)
    total_removed = initial_count - len(refined_data)
    total_removed_percentage = (total_removed / initial_count) * 100
    print(f"Total data removed: {total_removed_percentage:.2f}%")

    return refined_data

def standardize_features(data, feature_columns):
    scaler = StandardScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    return data

def build_and_evaluate_model(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    cv_scores = cross_val_score(logreg, X_train, y_train, cv=5)
    y_pred = logreg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print("\nModel Evaluation (Logistic Regression):")
    print("Cross-validation scores:", cv_scores)
    print("Accuracy:", accuracy)
    print("Mean Squared Error:", mse)
    return logreg, X_test, X_train, y_test, cv_scores, accuracy, mse

def perform_inference_glm(X_train, y_train, feature_names):
    X_train_sm = sm.add_constant(X_train)
    glm_binom = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())
    result = glm_binom.fit()
    print("\nGLM Model Coefficients (with Binomial family):")
    coeff_summary = result.summary2().tables[1]
    coeff_summary.index = ['const'] + feature_names
    print(coeff_summary)
    return coeff_summary

def compare_and_discuss(coeff_summary_logit, coeff_summary_glm):
    # Compare significant features between Logit and GLM models
    print("\nComparison and Discussion:")
    
    significant_features_logit = coeff_summary_logit[coeff_summary_logit['P>|z|'] < 0.05]
    print("\nSignificant Features (Logit):")
    print(significant_features_logit.index.tolist())
    
    significant_features_glm = coeff_summary_glm[coeff_summary_glm['P>|z|'] < 0.05]
    print("\nSignificant Features (GLM):")
    print(significant_features_glm.index.tolist())
    
    for feature in set(significant_features_logit.index) | set(significant_features_glm.index):
        logit_coef = coeff_summary_logit.loc[feature, 'Coef.'] if feature in significant_features_logit.index else None
        glm_coef = coeff_summary_glm.loc[feature, 'Coef.'] if feature in significant_features_glm.index else None
        logit_p = coeff_summary_logit.loc[feature, 'P>|z|'] if feature in significant_features_logit.index else None
        glm_p = coeff_summary_glm.loc[feature, 'P>|z|'] if feature in significant_features_glm.index else None
        
        print(f"\nFeature: {feature}")
        if logit_coef is not None and glm_coef is not None:
            print(f"  Logit Coefficient: {logit_coef}, P-value: {logit_p}")
            print(f"  GLM Coefficient: {glm_coef}, P-value: {glm_p}")
            if abs(logit_coef) > abs(glm_coef):
                print("  The Logit model suggests a stronger impact.")
            else:
                print("  The GLM model suggests a stronger impact.")
        elif logit_coef is not None:
            print("  Only significant in the Logit model.")
        else:
            print("  Only significant in the GLM model.")
        
        best_feature = feature
        if logit_p < 0.05 and glm_p < 0.05:
            if abs(logit_coef) > abs(glm_coef):
                impact = "positive" if logit_coef > 0 else "negative"
                print(f"  Best feature based on impact and significance: {best_feature} (Logit) with a {impact} impact.")
            else:
                impact = "positive" if glm_coef > 0 else "negative"
                print(f"  Best feature based on impact and significance: {best_feature} (GLM) with a {impact} impact.")
        elif logit_p < 0.05:
            impact = "positive" if logit_coef > 0 else "negative"
            print(f"  Best feature based on impact and significance: {best_feature} (Logit) with a {impact} impact.")
        elif glm_p < 0.05:
            impact = "positive" if glm_coef > 0 else "negative"
            print(f"  Best feature based on impact and significance: {best_feature} (GLM) with a {impact} impact.")


def plot_coefficients(coeff_summary):
    plt.figure(figsize=(12, 6))
    coeff_summary = coeff_summary.reset_index()
    sns.barplot(x='index', y='Coef.', data=coeff_summary)
    plt.xticks(rotation=90)
    plt.title('Coefficients from Logistic Regression')
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.show()

def plot_coefficient_significance(coeff_summary):
    coeff_summary['p-value'] = coeff_summary['P>|z|']
    coeff_summary['significant'] = coeff_summary['p-value'] < 0.05
    plt.figure(figsize=(12, 6))
    sns.barplot(x=coeff_summary.index, y='Coef.', hue='significant', data=coeff_summary)
    plt.xticks(rotation=90)
    plt.title('Coefficient Significance')
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.legend(title='Significant')  
    plt.show()

def plot_feature_importance(coeff_summary):
    coeff_summary['Importance'] = coeff_summary['Coef.'].abs()
    coeff_summary_sorted = coeff_summary.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=coeff_summary_sorted.index, y='Importance', data=coeff_summary_sorted)
    plt.xticks(rotation=90)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Absolute Coefficient Value')
    plt.show()

def plot_pairplot(data, features_to_keep):
    sns.pairplot(data[features_to_keep + ['BBS']], hue='BBS')
    plt.title('Pairwise Relationships')
    plt.show()

def plot_roc_curve(X_test, y_test, model):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()



def perform_inference_logit(X_train, y_train, feature_names):
    X_train_sm = sm.add_constant(X_train)
    logit_model = sm.Logit(y_train, X_train_sm)
    result = logit_model.fit()
    print("\nLogit Model Coefficients:")
    coeff_summary = result.summary2().tables[1]
    coeff_summary.index = ['const'] + feature_names
    print(coeff_summary)
    return coeff_summary

def plot_coefficients_with_ci_enhanced(coeff_summary):
    plt.style.use('ggplot')
    errors = [coeff_summary['Coef.'] - coeff_summary['[0.025'], coeff_summary['0.975]'] - coeff_summary['Coef.']]
    errors = np.abs(errors)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.errorbar(x=range(len(coeff_summary)), y=coeff_summary['Coef.'], yerr=errors, fmt='o', ecolor='darkred', capsize=5, linestyle='None', markersize=8, color='darkblue')
    
    significant = coeff_summary['P>|z|'] < 0.05
    ax.scatter(range(len(coeff_summary)), coeff_summary['Coef.'], color=np.where(significant, 'darkred', 'darkblue'), s=100, edgecolors='black')
    
    ax.axhline(y=0, linestyle='--', color='grey', linewidth=0.7)
    ax.set_xticks(range(len(coeff_summary)))
    ax.set_xticklabels(coeff_summary.index, rotation=45, ha="right")
    
    ax.set_title('Coefficient Significance with Confidence Intervals', fontsize=16, fontweight='bold')
    ax.set_xlabel('Variables', fontsize=14, fontweight='bold')
    ax.set_ylabel('Coefficient Value', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def main_analysis(data, filepath):
    feature_columns = data.columns.difference(['Participant', 'Sensor', 'Task', 'RMS Acceleration', 'BBS', 'Mean Acceleration X', 'Box Area', 'Ellipse Area'])
    data = standardize_features(data, feature_columns)
    X = data[feature_columns].values
    y = data['BBS'].values
    model, X_test, X_train, y_test, cv_scores, accuracy, mse = build_and_evaluate_model(X, y, feature_columns.tolist())
    
    print("\nComparing Inference Methods:")
    coeff_summary_logit = perform_inference_logit(X, y, feature_columns.tolist())
    coeff_summary_glm = perform_inference_glm(X, y, feature_columns.tolist())
    coeff_summary = perform_inference_logit(X, y, feature_columns.tolist())
    
    plot_coefficient_significance(coeff_summary)
    plot_feature_importance(coeff_summary)
    plot_roc_curve(X_test, y_test, model)
    
    print("\nComparing Inference Methods:")
    compare_and_discuss(coeff_summary_logit, coeff_summary_glm)
    plot_coefficients(coeff_summary_logit)
    
    actual_feature_names = ["const", "Age", 'Jerk X', "Jerk Y", "Jerk Z", "Area", "poids", "Gender", "taille", "RMS Angular Velocity", "Total Pathway Length", "volume", "Mean Acceleration Y", "Mean Acceleration Z"]
    coeff_summary_actual = coeff_summary_logit.copy()
    
    plot_coefficients_with_ci_enhanced(coeff_summary_actual)
    
    save_results_to_excel(coeff_summary_logit, coeff_summary_glm, cv_scores, accuracy, mse, filepath)

def save_results_to_excel(coeff_summary_logit, coeff_summary_glm, cv_scores, accuracy, mse, filepath):
    with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
        coeff_summary_logit.to_excel(writer, sheet_name='Logit Coefficients')
        coeff_summary_glm.to_excel(writer, sheet_name='GLM Coefficients')
        model_evaluation = pd.DataFrame({
            'Metric': ['Cross-validation scores', 'Accuracy', 'Mean Squared Error'],
            'Value': [np.mean(cv_scores), accuracy, mse]
        })
        model_evaluation.to_excel(writer, sheet_name='Model Evaluation')

filepath = 'C:/Users/youss/Downloads/result/result2/lowerback-analysis_summary.xlsx'
data = load_and_preprocess_data('C:/Users/youss/Downloads/result/result2/result4.csv')
data_clean = refine_outlier_removal(data)
data_clean.drop('z_score_max', axis=1, inplace=True, errors='ignore')
print("Analysis for Lower Back Sensor Data:")
main_analysis(data_clean, filepath)
