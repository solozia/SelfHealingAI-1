import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 0. Setup Environment ---
if not os.path.exists('logs'): os.makedirs('logs')
if not os.path.exists('logs/figures'): os.makedirs('logs/figures')
KNOWLEDGE_BASE_PATH = 'logs/knowledge_base.csv'

# --- 1. Shield and Recovery Systems ---

def shield_check(predicted_treatment, patient_record):
    """Checks a predicted treatment against safety rules."""
    if 'Amoxicillin' in predicted_treatment and ('penicillin' in patient_record['ALLERGIES'].lower() or 'amoxicillin' in patient_record['ALLERGIES'].lower()):
        return False, "Rule 1: Penicillin Allergy", patient_record['ALLERGIES']
    if 'sulfamethoxazole' in predicted_treatment.lower() and 'sulfa' in patient_record['ALLERGIES'].lower():
        return False, "Rule 2: Sulfa Allergy", patient_record['ALLERGIES']
    if predicted_treatment.lower() in ['aspirin', 'ibuprofen', 'naproxen']:
        if any(keyword in patient_record['ALLERGIES'].lower() for keyword in ['nsaid', 'aspirin', 'ibuprofen', 'naproxen']):
            return False, "Rule 3a: NSAID Allergy", patient_record['ALLERGIES']
        if 'kidney' in patient_record['CONDITIONS'].lower():
            return False, "Rule 3b: NSAID Contraindication (Kidney)", patient_record['CONDITIONS']
    if 'simvastatin' in predicted_treatment.lower() and ('hepatic' in patient_record['CONDITIONS'].lower() or 'liver' in patient_record['CONDITIONS'].lower()):
        return False, "Rule 4: Statin Contraindication (Liver)", patient_record['CONDITIONS']
    return True, None, None

def r1_rollback_recovery(patient_record_encoded):
    """R1 Recovery: Finds a safe treatment from a log of past successful cases."""
    if not os.path.exists(KNOWLEDGE_BASE_PATH): return None
    kb = pd.read_csv(KNOWLEDGE_BASE_PATH)
    kb_features_encoded = kb.drop('TREATMENT', axis=1)
    similarity_scores = cosine_similarity(patient_record_encoded, kb_features_encoded)
    if np.max(similarity_scores) > 0.95:
        most_similar_idx = np.argmax(similarity_scores)
        return kb.iloc[most_similar_idx]['TREATMENT']
    return None

def predict_with_rollback(model, patient_encoded, patient_unencoded, treatment_mapping):
    """Runs the full R0 and R1 recovery process."""
    probabilities = model.predict_proba(patient_encoded)[0]
    ranked_predictions = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)
    base_pred_name = treatment_mapping[ranked_predictions[0][0]]
    
    # Attempt R0 Recovery (Probabilistic Rollback)
    for class_index, _ in ranked_predictions:
        treatment_name = treatment_mapping[class_index]
        is_safe, _, _ = shield_check(treatment_name, patient_unencoded)
        if is_safe:
            return treatment_name, "R0 Recovery" if treatment_name != base_pred_name else "Allowed", base_pred_name if treatment_name != base_pred_name else None

    # Attempt R1 Recovery (Case-Based Rollback)
    r1_recommendation = r1_rollback_recovery(patient_encoded)
    if r1_recommendation:
        return r1_recommendation, "R1 Recovery", base_pred_name

    return "No Recommendation", "R1 Failure", base_pred_name

# --- 2. Data Loading and Preparation ---
def load_and_prepare_data():
    print("Step 1: Loading all data files...")
    try:
            patients = pd.read_csv('/content/drive/MyDrive/syntheaDataset/patients.csv')
            encounters = pd.read_csv('/content/drive/MyDrive/syntheaDataset/encounters.csv')
            medications = pd.read_csv('/content/drive/MyDrive/syntheaDataset/medications.csv')
            allergies = pd.read_csv('/content/drive/MyDrive/syntheaDataset/allergies.csv')
            observations = pd.read_csv('/content/drive/MyDrive/syntheaDataset/observations.csv')
            conditions = pd.read_csv('/content/drive/MyDrive/syntheaDataset/conditions.csv')
            devices = pd.read_csv('/content/drive/MyDrive/syntheaDataset/devices.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}.")
        return None, None
    print("All data loaded successfully.")
    print("Step 2: Preparing the dataset...")
    # (Data prep logic remains the same)
    patients['PATIENT_ID'] = patients['Id']
    merged_df = pd.merge(encounters, patients, left_on='PATIENT', right_on='Id', how='left', suffixes=('_enc', '_pat'))
    final_df = pd.merge(merged_df, medications, left_on='Id_enc', right_on='ENCOUNTER', how='inner', suffixes=('', '_med'))
    final_df = final_df[['PATIENT_ID', 'PATIENT', 'ENCOUNTERCLASS', 'REASONDESCRIPTION', 'BIRTHDATE', 'GENDER', 'RACE', 'ETHNICITY', 'DESCRIPTION_med']]
    final_df = final_df.rename(columns={'ENCOUNTERCLASS': 'VISIT_TYPE', 'REASONDESCRIPTION': 'DIAGNOSIS', 'DESCRIPTION_med': 'TREATMENT'})
    patient_allergies = allergies.groupby('PATIENT')['DESCRIPTION'].apply(lambda x: ', '.join(x)).reset_index()
    patient_allergies.rename(columns={'DESCRIPTION': 'ALLERGIES'}, inplace=True)
    final_df = pd.merge(final_df, patient_allergies, on='PATIENT', how='left')
    final_df['ALLERGIES'].fillna('No_Allergy_Reported', inplace=True)
    patient_conditions = conditions.groupby('PATIENT')['DESCRIPTION'].apply(lambda x: ', '.join(x)).reset_index()
    patient_conditions.rename(columns={'DESCRIPTION': 'CONDITIONS'}, inplace=True)
    final_df = pd.merge(final_df, patient_conditions, on='PATIENT', how='left')
    final_df['CONDITIONS'].fillna('No_Conditions_Reported', inplace=True)
    final_df['BIRTHDATE'] = pd.to_datetime(final_df['BIRTHDATE'], errors='coerce')
    final_df['AGE'] = (pd.to_datetime('today') - final_df['BIRTHDATE']).dt.days // 365
    final_df = final_df.drop(columns=['BIRTHDATE', 'PATIENT'])
    final_df.dropna(subset=['DIAGNOSIS', 'TREATMENT', 'AGE'], inplace=True)
    top_diagnoses = final_df['DIAGNOSIS'].value_counts().nlargest(10).index
    df_filtered = final_df[final_df['DIAGNOSIS'].isin(top_diagnoses)]
    top_treatments = df_filtered['TREATMENT'].value_counts().nlargest(5).index
    df_filtered = df_filtered[df_filtered['TREATMENT'].isin(top_treatments)]
    return df_filtered.drop('TREATMENT', axis=1), df_filtered['TREATMENT']

# --- 3. Adversarial Injection and Knowledge Base Creation ---
def setup_adversarial_test_set(X_test, model, X_test_encoded, treatment_mapping, injection_rate=0.5):
    print(f"\nInjecting adversarial cases into the test set (Rate: {injection_rate:.0%})...")
    X_test_adversarial = X_test.copy()
    predictions = model.predict(X_test_encoded)
    injected_count = 0
    for i in range(len(X_test_adversarial)):
        if np.random.rand() < injection_rate:
            patient_record = X_test_adversarial.iloc[i]
            predicted_treatment = treatment_mapping[predictions[i]]
            injected = False
            if 'Amoxicillin' in predicted_treatment and 'penicillin' not in patient_record['ALLERGIES'].lower():
                X_test_adversarial.at[patient_record.name, 'ALLERGIES'] = 'Allergy to penicillin'
                injected = True
            elif 'Ibuprofen' in predicted_treatment and 'kidney' not in patient_record['CONDITIONS'].lower():
                X_test_adversarial.at[patient_record.name, 'CONDITIONS'] = 'Chronic kidney disease'
                injected = True
            if injected:
                injected_count += 1
    print(f"Injected {injected_count} unsafe cases.")
    return X_test_adversarial

# --- 4. Main Experimental Framework ---
def run_experiments(X, y):
    print("\nStep 3: Training the model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train_encoded = pd.get_dummies(X_train.drop('PATIENT_ID', axis=1))
    X_test_encoded = pd.get_dummies(X_test.drop('PATIENT_ID', axis=1))
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    
    label_encoder = LabelEncoder().fit(y)
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    treatment_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_encoded, y_train_encoded)
    print("Model training complete.")

    # Create Knowledge Base from safe training cases
    knowledge_base_df = X_train_encoded.copy()
    knowledge_base_df['TREATMENT'] = y_train.values
    knowledge_base_df.to_csv(KNOWLEDGE_BASE_PATH, index=False)

    # Create Adversarial Test Set
    X_test_adversarial = setup_adversarial_test_set(X_test, model, X_test_encoded, treatment_mapping)

    print("\nStep 4: Running experiments on the ADVERSARIAL test set...")
    log_columns = ['scenario_id', 'setup', 'patient_id', 'unsafe_drug', 'rule_triggered', 'action_taken', 'final_recommendation', 'recovery_time_ms', 'is_unsafe_baseline']
    experiment_log = []
    
    for i in range(len(X_test_adversarial)):
        patient_record_adv = X_test_adversarial.iloc[i]
        patient_encoded = X_test_encoded.iloc[i].values.reshape(1, -1)
        base_pred_encoded = model.predict(patient_encoded)[0]
        base_pred_name = treatment_mapping[base_pred_encoded]
        is_safe_baseline, rule_base, _ = shield_check(base_pred_name, patient_record_adv)
        
        # --- Baseline ---
        experiment_log.append([i, 'Baseline', patient_record_adv['PATIENT_ID'], base_pred_name if not is_safe_baseline else None, rule_base, 'None', base_pred_name, 0, not is_safe_baseline])

        # --- Shield ---
        start_time_shield = time.perf_counter()
        final_rec_shield = base_pred_name if is_safe_baseline else "No Recommendation"
        overhead_shield = (time.perf_counter() - start_time_shield) * 1000
        experiment_log.append([i, 'Shield', patient_record_adv['PATIENT_ID'], base_pred_name if not is_safe_baseline else None, rule_base, 'Blocked' if not is_safe_baseline else 'Allowed', final_rec_shield, overhead_shield, not is_safe_baseline])

        # --- Rollback ---
        start_time_rb = time.perf_counter()
        final_rec_rb, action_rb, unsafe_drug_rb = (base_pred_name, 'Allowed', None) if is_safe_baseline else predict_with_rollback(model, patient_encoded, patient_record_adv, treatment_mapping)
        recovery_time_rb = (time.perf_counter() - start_time_rb) * 1000
        experiment_log.append([i, 'Rollback', patient_record_adv['PATIENT_ID'], unsafe_drug_rb, rule_base if not is_safe_baseline else None, action_rb, final_rec_rb, recovery_time_rb, not is_safe_baseline])

    log_df = pd.DataFrame(experiment_log, columns=log_columns)
    log_df.to_csv('logs/shield_logs.csv', index=False)
    print("Experimental logs saved to 'logs/shield_logs.csv'")
    return log_df, y_test_encoded, treatment_mapping, len(X_test)

# --- 5. Analysis and Visualization ---
def analyze_and_visualize(log_df, y_test_encoded, treatment_mapping, total_test_cases):
    print("\nStep 5: Analyzing results and generating report...")
    results = {}
    
    for setup in ['Baseline', 'Shield', 'Rollback']:
        setup_df = log_df[log_df['setup'] == setup].copy()
        
        # Unsafe Rate
        if setup == 'Baseline': unsafe_rate = setup_df['is_unsafe_baseline'].sum() / total_test_cases * 100
        else: unsafe_rate = (setup_df['final_recommendation'] == 'No Recommendation').sum() / total_test_cases * 100
        
        # Recovery Time
        recovery_time = setup_df[setup_df['is_unsafe_baseline']]['recovery_time_ms'].mean() if setup == 'Rollback' else np.nan
        
        # Accuracy
        correct, total_valid = 0, 0
        for i, row in setup_df.iterrows():
            if row['final_recommendation'] != 'No Recommendation':
                total_valid += 1
                rec_encoded = [k for k, v in treatment_mapping.items() if v == row['final_recommendation']]
                if rec_encoded and rec_encoded[0] == y_test_encoded[row['scenario_id']]: correct += 1
        accuracy = (correct / total_valid) * 100 if total_valid > 0 else 0
        
        # Overhead
        overhead = setup_df['recovery_time_ms'].mean()
        
        # False Positives
        fp_df = log_df[log_df['setup'] == 'Shield']
        fp_rate = len(fp_df[(fp_df['is_unsafe_baseline'] == False) & (fp_df['action_taken'] == 'Blocked')]) / total_test_cases * 100
        
        results[setup] = {"Unsafe Rec. Rate (%)": unsafe_rate, "Recovery Time (avg ms)": recovery_time,
                          "Accuracy after Safety Filters (%)": accuracy, "Overhead (avg ms)": overhead,
                          "False Positives (%)": fp_rate}

    results_df = pd.DataFrame(results).T
    print("\n--- Healthcare Results Table ---")
    print(results_df.to_string(formatters={'Unsafe Rec. Rate (%)':'{:.2f}'.format, 'Recovery Time (avg ms)':'{:.2f}'.format, 'Accuracy after Safety Filters (%)':'{:.2f}'.format, 'Overhead (avg ms)':'{:.2f}'.format, 'False Positives (%)':'{:.2f}'.format}))
    
    # Visualizations
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6)); sns.barplot(x=results_df.index, y='Unsafe Rec. Rate (%)', data=results_df, palette='viridis')
    plt.title('Unsafe Recommendation Rate Comparison', fontsize=16); plt.ylabel('Final Unsafe/No Recommendation Rate (%)', fontsize=12)
    plt.savefig('logs/figures/unsafe_rate_comparison.png')
    print("\nSaved unsafe rate bar chart to 'logs/figures/unsafe_rate_comparison.png'")

    plt.figure(figsize=(10, 6)); sns.barplot(x=['Rollback'], y=[results_df.loc['Rollback', 'Recovery Time (avg ms)']], palette='plasma')
    plt.title('Mean Recovery Time for Unsafe Cases', fontsize=16); plt.ylabel('Time (ms)', fontsize=12)
    plt.savefig('logs/figures/recovery_time.png')
    print("Saved recovery time bar chart to 'logs/figures/recovery_time.png'")

# --- Main Execution ---
if __name__ == "__main__":
    X_data, y_data = load_and_prepare_data()
    if X_data is not None:
        log_dataframe, y_test_data, treat_map, test_size = run_experiments(X_data, y_data)
        if log_dataframe is not None:
            analyze_and_visualize(log_dataframe, y_test_data, treat_map, test_size)
