"""
========================================================================================
PROJECT TITLE: Neuro_Pulse: Mental Health & Behavioral Biomarker Screener using Python ML
ACADEMIC CONTEXT: B.Tech Biotechnology Practical Lab Exam - AI & Bioinformatics Applications
========================================================================================

Description:
A 100% pure Python Machine Learning application that analyzes behavioral biomarkers 
(Sleep hours, Screen time, Physical activity) and physiological indicators 
(Heart Rate Variability - HRV, Resting Heart Rate) along with clinical psychometric 
scales (PHQ-9 Depression, GAD-7 Anxiety) to classify psychiatric risk levels and 
provide biotechnological / neuro-physiological recommendations directly in the terminal.

Libraries Used:
1. numpy (Numerical matrix operations)
2. pandas (Data loading, cleaning, and preprocessing)
3. matplotlib & seaborn (Bio-statistical visualization and plot saving)
4. scikit-learn (Feature scaling, Model training, Cross-validation & Evaluation)
5. joblib (Model serialization)
========================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib

# Set random seed for scientific reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Clinical Target Categories (Severity Levels)
TARGET_CLASSES = ["Minimal Risk", "Mild Risk", "Moderate Risk", "Severe Risk"]

# Configure visualization aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid", palette="deep")


# ======================================================================================
# SECTION 1: SYNTHETIC BIOMARKER DATASET GENERATOR
# ======================================================================================
def generate_synthetic_biomarker_dataset(n_samples=1200, filepath="mental_health_biomarkers.csv"):
    """
    Generates a realistic clinical-lifestyle dataset based on bioinformatics
    literature correlating sleep, HRV, physical activity, and psychometric scores.
    """
    print("\n[STEP 1] Generating Synthetic Mental Health Biomarker Dataset for Neuro_Pulse...")
    
    # 1. Demographic Features
    age = np.random.randint(18, 66, size=n_samples)
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.48, 0.52])
    
    # 2. Lifestyle & Physiological Biomarkers
    latent_stress = np.random.beta(a=2, b=2, size=n_samples)
    
    # Sleep hours: Inversely related to stress (Low: ~4h, Optimal: ~8h)
    sleep_hours = np.round(9.5 - (latent_stress * 5.0) + np.random.normal(0, 0.4, n_samples), 1)
    sleep_hours = np.clip(sleep_hours, 3.5, 10.0)
    
    # Screen time: Positively related to stress
    screen_time = np.round(2.0 + (latent_stress * 8.5) + np.random.normal(0, 0.6, n_samples), 1)
    screen_time = np.clip(screen_time, 1.0, 14.0)
    
    # Physical activity (mins/day): Inversely related to stress
    phys_act = np.round(75.0 - (latent_stress * 65.0) + np.random.normal(0, 8.0, n_samples), 0)
    phys_act = np.clip(phys_act, 0.0, 120.0)
    
    # HRV (Heart Rate Variability in ms - RMSSD): Low HRV reflects chronic sympathetic overdrive
    hrv = np.round(78.0 - (latent_stress * 52.0) + np.random.normal(0, 5.0, n_samples), 1)
    hrv = np.clip(hrv, 15.0, 95.0)
    
    # Resting Heart Rate (BPM): Elevated under autonomic stress
    resting_hr = np.round(62.0 + (latent_stress * 28.0) + np.random.normal(0, 3.5, n_samples), 0)
    resting_hr = np.clip(resting_hr, 52.0, 108.0)
    
    # 3. Clinical Psychometric Scores
    # PHQ-9 (Depression screener: 0 to 27)
    phq9 = np.round((latent_stress * 23.0) + np.random.normal(1.5, 2.0, n_samples), 0)
    phq9 = np.clip(phq9, 0, 27).astype(int)
    
    # GAD-7 (Anxiety screener: 0 to 21)
    gad7 = np.round((latent_stress * 18.0) + np.random.normal(1.0, 1.8, n_samples), 0)
    gad7 = np.clip(gad7, 0, 21).astype(int)
    
    # 4. Target Severity Calculation
    # 0 = Minimal Risk, 1 = Mild Risk, 2 = Moderate Risk, 3 = Severe Risk
    severity = []
    for p, g, h in zip(phq9, gad7, hrv):
        composite_score = max(p, g * 1.25)
        if h < 30.0:  # Low HRV acts as physiological amplifier of risk
            composite_score += 2.0
            
        if composite_score < 5.0:
            severity.append(0) # Minimal
        elif composite_score < 10.0:
            severity.append(1) # Mild
        elif composite_score < 15.0:
            severity.append(2) # Moderate
        else:
            severity.append(3) # Severe
            
    df = pd.DataFrame({
        "Patient_ID": [f"NP-BIO-{i+1001}" for i in range(n_samples)],
        "Age": age,
        "Gender": gender,
        "Sleep_Hours": sleep_hours,
        "Physical_Activity_Min": phys_act,
        "Screen_Time_Hours": screen_time,
        "HRV_ms": hrv,
        "Resting_HR_BPM": resting_hr,
        "PHQ9_Score": phq9,
        "GAD7_Score": gad7,
        "Risk_Severity": severity
    })
    
    df.to_csv(filepath, index=False)
    print(f"[SUCCESS] Dataset created and saved to '{filepath}' ({df.shape[0]} records).")
    return df


# ======================================================================================
# SECTION 2: DATA PREPROCESSING & FEATURE PIPELINE
# ======================================================================================
def load_and_preprocess_data(filepath="mental_health_biomarkers.csv"):
    """
    Loads data using pandas, cleans missing values, encodes categories,
    normalizes features with StandardScaler, and splits into train/test sets.
    """
    print("\n[STEP 2] Loading & Preprocessing Biomarker Data...")
    
    if not os.path.exists(filepath):
        df = generate_synthetic_biomarker_dataset(filepath=filepath)
    else:
        df = pd.read_csv(filepath)
        
    print(f"- Dataset Shape: {df.shape[0]} rows, {df.shape[1]} features.")
    
    # Check and handle missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"  * Handling {missing_count} missing values with median imputation...")
        df.fillna(df.median(numeric_only=True), inplace=True)
    else:
        print("- Missing Values Check: 0 missing cells found (Clean dataset).")
        
    feature_cols = [
        "Age", "Gender_Code", "Sleep_Hours", "Physical_Activity_Min",
        "Screen_Time_Hours", "HRV_ms", "Resting_HR_BPM", "PHQ9_Score", "GAD7_Score"
    ]
    
    # Encode Gender (Male -> 1, Female -> 0)
    le = LabelEncoder()
    df["Gender_Code"] = le.fit_transform(df["Gender"])
    
    X = df[feature_cols]
    y = df["Risk_Severity"]
    
    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    
    # Feature Scaling using StandardScaler (z = (x - u) / s)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, "biomarker_scaler.pkl")
    joblib.dump(le, "gender_encoder.pkl")
    
    print(f"- Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples.")
    print("- Feature scaler saved to 'biomarker_scaler.pkl'.")
    
    return df, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler


# ======================================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA) & BIO-VISUALIZATIONS
# ======================================================================================
def generate_eda_visualizations(df, output_dir="."):
    """
    Generates and saves publication-quality bio-statistical plots:
    1. Correlation Heatmap (Biomarkers vs Psychometrics)
    2. Sleep Duration vs Psychiatric Severity (Boxplot + Strip plot)
    3. Autonomic Vagal Tone (HRV) vs Screen Time (Scatter plot)
    """
    print("\n[STEP 3] Generating Publication-Style EDA Bio-Visualizations for Neuro_Pulse...")
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = ["Age", "Sleep_Hours", "Physical_Activity_Min", "Screen_Time_Hours", 
                    "HRV_ms", "Resting_HR_BPM", "PHQ9_Score", "GAD7_Score", "Risk_Severity"]
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
        vmax=1.0, vmin=-1.0, linewidths=1.0, mask=mask,
        cbar_kws={'label': 'Pearson Correlation Coefficient (r)'}
    )
    plt.title("Neuro_Pulse: Biomarker Correlation Matrix\n(Physiological, Behavioral & Psychometric Indices)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    corr_path = os.path.join(output_dir, "eda_correlation_heatmap.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"  [+] Saved Correlation Heatmap: '{corr_path}'")
    
    # 2. Sleep Duration vs Severity Boxplot
    plt.figure(figsize=(9, 6))
    palette = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]
    sns.boxplot(
        x="Risk_Severity", y="Sleep_Hours", data=df, hue="Risk_Severity",
        palette=palette, legend=False, width=0.5, boxprops=dict(alpha=0.8)
    )
    sns.stripplot(x="Risk_Severity", y="Sleep_Hours", data=df, color="black", alpha=0.25, jitter=0.2, size=4)
    plt.xticks(ticks=[0, 1, 2, 3], labels=TARGET_CLASSES, fontweight="bold")
    plt.title("Neuro_Pulse: Circadian Sleep Duration vs Psychiatric Severity Tier", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Clinical Risk Tier", fontsize=11, fontweight="bold")
    plt.ylabel("Sleep Duration (Hours / Night)", fontsize=11, fontweight="bold")
    plt.axhline(y=7.0, color='blue', linestyle='--', alpha=0.7, label='Clinical Optimum (7.0h)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    sleep_path = os.path.join(output_dir, "eda_sleep_vs_severity.png")
    plt.savefig(sleep_path, dpi=300)
    plt.close()
    print(f"  [+] Saved Sleep vs Severity Boxplot: '{sleep_path}'")
    
    # 3. Autonomic Vagal Tone (HRV) vs Screen Time Scatter
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        x="Screen_Time_Hours", y="HRV_ms", hue="Risk_Severity", 
        palette=palette, data=df, alpha=0.85, s=45
    )
    plt.title("Neuro_Pulse: Autonomic Tone (HRV ms) vs Screen Time", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Screen Time Exposure (Hours / Day)", fontsize=11, fontweight="bold")
    plt.ylabel("Heart Rate Variability - RMSSD (ms)", fontsize=11, fontweight="bold")
    plt.axhline(y=35.0, color='red', linestyle=':', label='Sympathetic Overdrive Threshold (<35ms)')
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=TARGET_CLASSES + ['Sympathetic Overdrive (<35ms)'], loc='upper right', title="Risk Category")
    plt.tight_layout()
    hrv_path = os.path.join(output_dir, "eda_hrv_vs_screen_time.png")
    plt.savefig(hrv_path, dpi=300)
    plt.close()
    print(f"  [+] Saved HRV vs Screen Time Plot: '{hrv_path}'")


# ======================================================================================
# SECTION 4: MACHINE LEARNING MODEL TRAINING & COMPARATIVE EVALUATION
# ======================================================================================
def train_and_evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, output_dir="."):
    """
    Trains and compares:
    - Random Forest Classifier (Non-linear Ensemble Bagging)
    - Gradient Boosting Classifier (Sequential Ensemble Boosting)
    - Logistic Regression (Linear Regularized Classifier)
    
    Evaluates using 5-Fold Cross Validation, Accuracy, F1-Score, and Confusion Matrix.
    """
    print("\n[STEP 4] Training Machine Learning Models for Neuro_Pulse Screening...")
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=RANDOM_SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=RANDOM_SEED),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=RANDOM_SEED)
    }
    
    results = {}
    best_acc = 0.0
    best_model_name = ""
    best_model_obj = None
    
    print("\n" + "="*75)
    print(f"{'Model Name':<22} | {'5-Fold CV Accuracy':<18} | {'Test Accuracy':<14} | {'Macro F1':<10}")
    print("="*75)
    
    for name, model in models.items():
        train_x = X_train_scaled if name == "Logistic Regression" else X_train
        test_x = X_test_scaled if name == "Logistic Regression" else X_test
        
        # 5-Fold Cross Validation
        cv_scores = cross_val_score(model, train_x, y_train, cv=5, scoring="accuracy")
        
        # Train model
        model.fit(train_x, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(test_x)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        
        results[name] = {
            "model": model, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
            "test_acc": acc, "f1_score": f1, "y_pred": y_pred
        }
        
        print(f"{name:<22} | {cv_scores.mean():.4f} (+/- {cv_scores.std():.3f})  | {acc:.4f} ({acc*100:.1f}%)   | {f1:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model_obj = model
            
    print("="*75)
    print(f"\n[SELECTED CHAMPION MODEL]: {best_model_name} with Test Accuracy = {best_acc*100:.2f}%\n")
    
    best_pred = results[best_model_name]["y_pred"]
    print("Detailed Classification Report (Precision, Recall, F1-Score per Severity Tier):")
    print("-" * 65)
    print(classification_report(y_test, best_pred, target_names=TARGET_CLASSES))
    print("-" * 65)
    
    # Save the Champion Model to disk
    model_save_path = os.path.join(output_dir, "neuropulse_model.pkl")
    joblib.dump(best_model_obj, model_save_path)
    print(f"[SUCCESS] Trained model serialized and saved to: '{model_save_path}'")
    
    # Confusion Matrix Plot
    plt.figure(figsize=(7, 6))
    cm = confusion_matrix(y_test, best_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES)
    plt.title(f"Neuro_Pulse Confusion Matrix: {best_model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Risk Tier", fontsize=11, fontweight="bold")
    plt.ylabel("Clinical Ground Truth", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "model_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  [+] Saved Confusion Matrix: '{cm_path}'")
    
    # Feature Importance Plot
    rf_model = results["Random Forest"]["model"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    clean_feature_names = [f.replace("_", " ") for f in feature_cols]
    sorted_features = [clean_feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Blues_r", len(sorted_features))
    sns.barplot(x=sorted_importances, y=sorted_features, hue=sorted_features, palette=colors, legend=False)
    plt.title("Neuro_Pulse: Bio-Marker Gini Feature Importance Ranking (Random Forest)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Relative Predictive Importance Weight", fontsize=11, fontweight="bold")
    plt.ylabel("Biomarker Feature", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fi_path = os.path.join(output_dir, "model_feature_importance.png")
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"  [+] Saved Feature Importance Chart: '{fi_path}'")
    
    return best_model_obj, best_model_name, results


# ======================================================================================
# SECTION 5: BIOMARKER DIAGNOSTIC & BIOTECHNOLOGY RECOMMENDATION ENGINE
# ======================================================================================
def diagnose_biomarker_drivers(age, gender, sleep_hours, physical_activity, screen_time, hrv, resting_hr, phq9, gad7):
    """
    Bioinformatics rule-based diagnostic engine.
    Maps lifestyle/physiological deviations to neuro-biological mechanisms and interventions.
    """
    primary_drivers = []
    bio_recommendations = []
    
    # 1. Circadian Rhythm & Sleep Homeostasis
    if sleep_hours < 6.0:
        primary_drivers.append("Chronic Sleep Deprivation & Impaired Glymphatic Clearance (<6h sleep)")
        bio_recommendations.append(
            "- [Circadian Regulation]: Sleep < 6h disrupts cortisol rhythm and hippocampal BDNF. "
            "Adopt consistent 10:30 PM - 6:30 AM sleep-wake window; avoid late-night caffeine to preserve slow-wave delta sleep."
        )
    elif sleep_hours > 9.5:
        primary_drivers.append("Excessive Hypersomnia / Disrupted Circadian Entrainment (>9.5h sleep)")
        bio_recommendations.append(
            "- [Sleep Homeostasis]: Long sleep duration without restorative rest may indicate atypical depressive lethargy. "
            "Implement morning 10,000-lux bright light exposure to reset the Suprachiasmatic Nucleus (SCN)."
        )
        
    # 2. Digital Blue Light Exposure & Suprachiasmatic Desynchronization
    if screen_time > 7.5:
        primary_drivers.append(f"Excessive Screen Exposure ({screen_time}h/day - Blue Light Melatonin Suppression)")
        bio_recommendations.append(
            "- [Neuro-Photobiology]: High 450-480nm blue spectrum exposure suppresses pineal melatonin secretion. "
            "Enable amber blue-light filtration 2 hours prior to sleep and institute a 60-minute digital sunset."
        )
        
    # 3. Autonomic Vagus Tone & Sympathovagal Balance (HRV)
    if hrv < 35.0:
        primary_drivers.append(f"Depressed Parasympathetic Vagal Tone (HRV {hrv} ms < 35 ms threshold)")
        bio_recommendations.append(
            "- [Autonomic Nervous System]: Low Heart Rate Variability indicates chronic sympathetic overdrive. "
            "Practice 0.1 Hz resonance-frequency breathing (5.5s inhale, 5.5s exhale for 10 mins) to stimulate vagal nerve acetylcholine release."
        )
    elif hrv >= 60.0:
        bio_recommendations.append(
            "- [Physiological Resilience]: Healthy parasympathetic HRV (>60ms) reflects robust neurocardiac adaptability."
        )
        
    # 4. Metabolic & Neurotrophic Factor Upregulation (Physical Activity)
    if physical_activity < 25.0:
        primary_drivers.append(f"Sedentary Lifestyle ({int(physical_activity)} mins/day physical activity)")
        bio_recommendations.append(
            "- [Neurogenesis & BDNF]: Physical activity < 25 min/day limits PGC-1alpha and Brain-Derived Neurotrophic Factor. "
            "Target 30 minutes of moderate aerobic exercise (Zone 2 cardio) 4x/week to promote hippocampal neuroplasticity."
        )
        
    # 5. Psychometric Threshold Alerts
    if phq9 >= 15:
        primary_drivers.append(f"Elevated PHQ-9 Depressive Symptomatology (Score: {phq9}/27)")
        bio_recommendations.append(
            "- [Clinical Attention]: High depressive symptom burden detected. Prioritize clinical consultation alongside lifestyle regulation."
        )
        
    if gad7 >= 15:
        primary_drivers.append(f"High GAD-7 Generalized Anxiety Metric (Score: {gad7}/21)")
        bio_recommendations.append(
            "- [HPA Axis Modulation]: High anxiety signals chronic hypothalamic-pituitary-adrenal axis stimulation. Consider mindfulness-based stress reduction (MBSR)."
        )
        
    if not primary_drivers:
        primary_drivers.append("Optimal Homeostatic Balance (All lifestyle biomarkers within clinical target range)")
        bio_recommendations.append("- [Maintenance]: Excellent biomarker stability. Continue current circadian and metabolic practices.")
        
    return primary_drivers, bio_recommendations


def predict_patient_risk(model, patient_dict, feature_cols, scaler=None):
    """
    Formats features, runs model inference, and returns predicted risk tier,
    probabilities, and bio-recommendations.
    """
    gender_code = 1 if str(patient_dict.get("Gender", "Male")).lower() in ["male", "m", "1"] else 0
    
    input_df = pd.DataFrame([{
        "Age": float(patient_dict["Age"]),
        "Gender_Code": gender_code,
        "Sleep_Hours": float(patient_dict["Sleep_Hours"]),
        "Physical_Activity_Min": float(patient_dict["Physical_Activity_Min"]),
        "Screen_Time_Hours": float(patient_dict["Screen_Time_Hours"]),
        "HRV_ms": float(patient_dict["HRV_ms"]),
        "Resting_HR_BPM": float(patient_dict["Resting_HR_BPM"]),
        "PHQ9_Score": float(patient_dict["PHQ9_Score"]),
        "GAD7_Score": float(patient_dict["GAD7_Score"])
    }])[feature_cols]
    
    pred_class = model.predict(input_df)[0]
    pred_tier = TARGET_CLASSES[pred_class]
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
    else:
        probs = [0.0, 0.0, 0.0, 0.0]
        probs[pred_class] = 1.0
        
    drivers, recommendations = diagnose_biomarker_drivers(
        age=patient_dict["Age"],
        gender=patient_dict["Gender"],
        sleep_hours=patient_dict["Sleep_Hours"],
        physical_activity=patient_dict["Physical_Activity_Min"],
        screen_time=patient_dict["Screen_Time_Hours"],
        hrv=patient_dict["HRV_ms"],
        resting_hr=patient_dict["Resting_HR_BPM"],
        phq9=patient_dict["PHQ9_Score"],
        gad7=patient_dict["GAD7_Score"]
    )
    
    return {
        "predicted_class_id": int(pred_class),
        "predicted_risk_tier": pred_tier,
        "confidence_probabilities": {TARGET_CLASSES[i]: round(probs[i] * 100, 1) for i in range(len(TARGET_CLASSES))},
        "primary_drivers": drivers,
        "recommendations": recommendations
    }


# ======================================================================================
# SECTION 6: PRACTICAL LAB CLI & INTERACTIVE EXAMINER INFERENCE
# ======================================================================================
def print_clinical_report(patient_name, patient_data, result):
    """Clean Formatted Terminal Output for Lab Viva presentation."""
    print("\n" + "="*80)
    print(f"       NEURO_PULSE: CLINICAL BIOMARKER & PSYCHIATRIC RISK REPORT")
    print("="*80)
    print(f"Patient / Candidate Profile : {patient_name}")
    print(f"Age: {patient_data['Age']} yrs | Gender: {patient_data['Gender']} | Resting HR: {patient_data['Resting_HR_BPM']} BPM")
    print(f"Sleep Duration              : {patient_data['Sleep_Hours']} hrs/day")
    print(f"Physical Activity           : {patient_data['Physical_Activity_Min']} mins/day")
    print(f"Screen Time Exposure        : {patient_data['Screen_Time_Hours']} hrs/day")
    print(f"Heart Rate Variability (HRV): {patient_data['HRV_ms']} ms (RMSSD Vagal Index)")
    print(f"Psychometric Scales         : PHQ-9 = {patient_data['PHQ9_Score']}/27 | GAD-7 = {patient_data['GAD7_Score']}/21")
    print("-" * 80)
    
    tier = result["predicted_risk_tier"]
    badge_colors = {
        "Minimal Risk": "\033[92m[MINIMAL RISK - NORMAL]\033[0m",
        "Mild Risk": "\033[93m[MILD RISK - WATCHLIST]\033[0m",
        "Moderate Risk": "\033[91m[MODERATE RISK - INTERVENTION NEEDED]\033[0m",
        "Severe Risk": "\033[95m[SEVERE RISK - CLINICAL PRIORITY]\033[0m"
    }
    badge = badge_colors.get(tier, f"[{tier.upper()}]")
    
    print(f"--> AI PREDICTED RISK LEVEL : {badge} ({tier})")
    print("--> Class Probability Distribution:")
    for cls_name, prob in result["confidence_probabilities"].items():
        bar = "#" * int(prob / 4)
        print(f"    - {cls_name:<14} : {prob:>5.1f}% | {bar}")
        
    print("-" * 80)
    print("--> IDENTIFIED HIGH-RISK BIOMARKER DRIVERS:")
    for idx, d in enumerate(result["primary_drivers"], 1):
        print(f"    {idx}. {d}")
        
    print("\n--> BIOTECHNOLOGY & PHYSIOLOGICAL INTERVENTIONS:")
    for rec in result["recommendations"]:
        print(f"    {rec}")
    print("="*80 + "\n")


def run_demo_cases(model, feature_cols):
    """Runs standard pre-configured student/patient cases for quick viva demo."""
    demo_profiles = [
        {
            "name": "Case A: Healthy Bio-Engineering Student (Normal Baseline)",
            "data": {
                "Age": 21, "Gender": "Male", "Sleep_Hours": 8.0, "Physical_Activity_Min": 50.0,
                "Screen_Time_Hours": 3.5, "HRV_ms": 72.0, "Resting_HR_BPM": 64.0,
                "PHQ9_Score": 2, "GAD7_Score": 3
            }
        },
        {
            "name": "Case B: Exam Stress & Screen Fatigue (Moderate Anxiety)",
            "data": {
                "Age": 22, "Gender": "Female", "Sleep_Hours": 5.2, "Physical_Activity_Min": 15.0,
                "Screen_Time_Hours": 9.5, "HRV_ms": 32.0, "Resting_HR_BPM": 86.0,
                "PHQ9_Score": 8, "GAD7_Score": 13
            }
        },
        {
            "name": "Case C: Chronic Burnout & Sympathetic Overdrive (Severe Depression Risk)",
            "data": {
                "Age": 28, "Gender": "Male", "Sleep_Hours": 4.0, "Physical_Activity_Min": 5.0,
                "Screen_Time_Hours": 11.0, "HRV_ms": 21.0, "Resting_HR_BPM": 95.0,
                "PHQ9_Score": 19, "GAD7_Score": 17
            }
        }
    ]
    
    print("\n" + "="*80)
    print("   RUNNING AUTOMATED VIVA DEMONSTRATION CASES (Pre-configured Profiles)")
    print("="*80)
    
    for case in demo_profiles:
        result = predict_patient_risk(model, case["data"], feature_cols)
        print_clinical_report(case["name"], case["data"], result)


def interactive_cli_session(model, feature_cols):
    """Interactive command-line interface for practical exam evaluation."""
    print("\n" + "="*80)
    print("   NEURO_PULSE: INTERACTIVE PRACTICAL EXAM INFERENCE CLI")
    print("="*80)
    print("Enter candidate lifestyle and biomarker parameters for live AI risk assessment.\n")
    
    try:
        age_str = input("1. Enter Age (years, e.g. 21) [Default: 22]: ").strip()
        age = float(age_str) if age_str else 22.0
        
        gender = input("2. Enter Gender (Male/Female) [Default: Female]: ").strip() or "Female"
        
        sleep_str = input("3. Enter Average Sleep Duration (hours/night, e.g. 6.5) [Default: 6.0]: ").strip()
        sleep_hours = float(sleep_str) if sleep_str else 6.0
        
        act_str = input("4. Enter Physical Activity (minutes/day, e.g. 30) [Default: 20]: ").strip()
        act = float(act_str) if act_str else 20.0
        
        screen_str = input("5. Enter Screen Time Exposure (hours/day, e.g. 8.0) [Default: 7.5]: ").strip()
        screen = float(screen_str) if screen_str else 7.5
        
        hrv_str = input("6. Enter Heart Rate Variability - HRV (RMSSD in ms, e.g. 40.0) [Default: 38.0]: ").strip()
        hrv = float(hrv_str) if hrv_str else 38.0
        
        rhr_str = input("7. Enter Resting Heart Rate (BPM, e.g. 72) [Default: 76]: ").strip()
        rhr = float(rhr_str) if rhr_str else 76.0
        
        phq_str = input("8. Enter PHQ-9 Depression Score (0 - 27) [Default: 7]: ").strip()
        phq9 = int(phq_str) if phq_str else 7
        
        gad_str = input("9. Enter GAD-7 Anxiety Score (0 - 21) [Default: 8]: ").strip()
        gad7 = int(gad_str) if gad_str else 8
        
        custom_patient = {
            "Age": age, "Gender": gender, "Sleep_Hours": sleep_hours,
            "Physical_Activity_Min": act, "Screen_Time_Hours": screen,
            "HRV_ms": hrv, "Resting_HR_BPM": rhr,
            "PHQ9_Score": phq9, "GAD7_Score": gad7
        }
        
        result = predict_patient_risk(model, custom_patient, feature_cols)
        print_clinical_report("Live Practical Exam Interactive Sample", custom_patient, result)
        
    except KeyboardInterrupt:
        print("\nInteractive CLI session closed by user.")
    except Exception as e:
        print(f"\n[Error parsing input]: {e}. Running default demo instead.")


# ======================================================================================
# MAIN EXECUTION ORCHESTRATOR
# ======================================================================================
def main():
    print("="*80)
    print("        NEURO_PULSE: MENTAL HEALTH & BIOMARKER SCREENER (PYTHON ML)")
    print("       B.Tech Biotechnology Practical Lab Exam AI Application Suite")
    print("="*80)
    
    csv_file = "mental_health_biomarkers.csv"
    
    # 1. Pipeline: Load or Generate Dataset and Preprocess
    df, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler = load_and_preprocess_data(csv_file)
    
    # 2. EDA: Generate Publication-Style Bio-Plots (.png)
    generate_eda_visualizations(df)
    
    # 3. Model Training & Comparative Evaluation
    best_model, best_name, results = train_and_evaluate_models(
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    )
    
    # 4. Automated Demo Inference Cases for Lab Viva
    run_demo_cases(best_model, feature_cols)
    
    # 5. Interactive CLI Prompt Option
    print("\n" + "="*80)
    print("Options:")
    print("  [1] Launch Interactive Live Examiner CLI Input Session")
    print("  [2] Exit (Plots and models generated successfully)")
    print("="*80)
    
    if sys.stdin.isatty():
        choice = input("\nEnter choice (1 or 2) [Default: 2]: ").strip()
        if choice == "1":
            interactive_cli_session(best_model, feature_cols)
        else:
            print("\nExecution complete. All artifacts and figures are saved and ready for viva examination.")
    else:
        print("\n[INFO] Running in automated pipeline mode. All figures, dataset, and models are ready.")


if __name__ == "__main__":
    main()
