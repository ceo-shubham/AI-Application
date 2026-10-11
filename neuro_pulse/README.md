# Neuro_Pulse: Mental Health & Behavioral Biomarker Screener using Python ML

**Academic Subject**: B.Tech Biotechnology — Practical Lab Exam on AI & Bioinformatics Applications  
**Domain**: Bioinformatics, Computational Neuroscience & Machine Learning  
**Core Technologies**: Pure Python (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`) — 100% Terminal-Based.

---

## 📌 Project Overview

**Neuro_Pulse** is an end-to-end Python Machine Learning application that bridges **digital behavioral biomarkers** (sleep duration, digital screen exposure, physical activity) and **physiological indices** (Heart Rate Variability - HRV RMSSD, Resting Heart Rate) with standard clinical psychometric scales (**PHQ-9** for Major Depressive Disorder, **GAD-7** for Generalized Anxiety Disorder).

The system classifies psychiatric risk into 4 stratified tiers (**Minimal, Mild, Moderate, Severe**), identifies primary physiological/lifestyle stress drivers, and delivers targeted **biotechnological and neuro-physiological interventions** directly in the terminal.

---

## 🧬 Biological & Clinical Significance of Parameters

| Parameter | Unit / Scale | Normal Baseline | Biological & Neuro-Physiological Relevance |
|---|---|---|---|
| **Sleep Hours** | Hours / night | 7.0 – 9.0 hrs | **Circadian & Glymphatic Clearance**: Sleep <6h impairs brain glymphatic beta-amyloid clearance, elevates cortisol, and reduces hippocampal synaptic plasticity. |
| **Heart Rate Variability (HRV)** | RMSSD (ms) | 45.0 – 85.0 ms | **Autonomic Vagus Tone**: Higher HRV reflects parasympathetic vagal flexibility. HRV <35ms indicates chronic sympathetic overdrive and allostatic stress load. |
| **Physical Activity** | Mins / day | >30 mins/day | **Neurogenesis & BDNF**: Aerobic activity up-regulates Brain-Derived Neurotrophic Factor (BDNF) and stimulates adult hippocampal neurogenesis. |
| **Screen Time Exposure** | Hours / day | <5.0 hrs/day | **Melatonin & Suprachiasmatic Nucleus**: Blue light (450–480 nm) suppresses pineal melatonin secretion and delays REM sleep onset. |
| **Resting Heart Rate** | Beats / min (BPM) | 60 – 80 BPM | Indicator of cardiovascular stress and sympathovagal homeostasis. |
| **PHQ-9 Score** | 0 – 27 | 0 – 4 (Minimal) | Gold standard clinical psychometric instrument for depression screening. |
| **GAD-7 Score** | 0 – 21 | 0 – 4 (Minimal) | Validated clinical psychometric scale for anxiety severity screening. |

---

## 🚀 Quick Installation & Setup

### Install Required Libraries
Run the standard pip command in your terminal / command prompt:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

---

## 💻 How to Run the Project

Execute the main script in your terminal:

```bash
python main.py
```

### What Happens When You Run `main.py`:
1. **Dataset Pipeline**: Creates/loads `mental_health_biomarkers.csv` (1,200 patient records).
2. **Data Preprocessing**: Handles missing values, encodes categories, and standardizes continuous features using `StandardScaler`.
3. **Bio-Visualizations**: Generates and saves publication-quality `.png` charts.
4. **Machine Learning Training**: Trains & evaluates **Random Forest**, **Gradient Boosting**, and **Logistic Regression** with 5-fold cross-validation.
5. **Automated Demo Profiles**: Runs 3 real-world patient cases (Healthy Student, Exam Burnout, Severe Depression Risk).
6. **Interactive CLI Prompt**: Allows the examiner to enter real-time parameters for instant risk prediction!

---

## 📊 Generated Output Files

| File Name | Description |
|---|---|
| `main.py` | Complete runnable Python application script with detailed viva comments. |
| `mental_health_biomarkers.csv` | Synthetic dataset of 1,200 records with clinical & lifestyle features. |
| `eda_correlation_heatmap.png` | Pearson correlation matrix between physiological metrics and psychometric scores. |
| `eda_sleep_vs_severity.png` | Boxplot + stripplot showing sleep duration distribution across severity tiers. |
| `eda_hrv_vs_screen_time.png` | Autonomic tone (HRV ms) vs screen exposure scatter plot. |
| `model_feature_importance.png` | Random Forest Gini feature importance ranking. |
| `model_confusion_matrix.png` | Multi-class confusion matrix on unseen test data. |
| `neuropulse_model.pkl` | Serialized trained champion model for instant inference. |
| `biomarker_scaler.pkl` | Fitted StandardScaler object for feature normalization. |

---

## 🎓 Viva Voce Q&A Cheat Sheet (Top Examiner Questions)

### Q1: Why is Heart Rate Variability (HRV) an important biomarker in mental health AI?
> **Answer**: HRV measures beat-to-beat variations in heart rate (RMSSD in milliseconds). It reflects the balance between the **parasympathetic (vagal)** and **sympathetic** branches of the Autonomic Nervous System (ANS). A low HRV (<35 ms) is a physiological indicator of chronic sympathetic stress and impaired emotional regulation.

### Q2: Why did we choose Random Forest over a single Decision Tree?
> **Answer**: Random Forest is an **ensemble bagging algorithm** that builds multiple de-correlated decision trees with feature bootstrapping. It reduces variance, prevents overfitting, and provides reliable feature importance rankings.

### Q3: What is the purpose of `StandardScaler` in the pipeline?
> **Answer**: `StandardScaler` standardizes continuous features so that they have a mean ($\mu$) of 0 and a standard deviation ($\sigma$) of 1:
> $$z = \frac{x - \mu}{\sigma}$$
> This prevents features with larger numerical magnitudes (e.g. Physical Activity in minutes) from dominating distance- and gradient-based algorithms (like Logistic Regression).

### Q4: What are PHQ-9 and GAD-7?
> **Answer**: 
> - **PHQ-9** (Patient Health Questionnaire-9) is a 9-item scale (0–27 points) assessing depressive symptom severity.
> - **GAD-7** (Generalized Anxiety Disorder-7) is a 7-item scale (0–21 points) assessing anxiety disorder severity.

### Q5: How does the recommendation engine connect machine learning to biotechnology?
> **Answer**: When a high-risk tier is predicted, the diagnostic engine maps abnormal feature thresholds to physiological pathways—such as prescribing **Zone 2 aerobic exercise** to stimulate **BDNF (Brain-Derived Neurotrophic Factor)** synthesis, **0.1 Hz resonance breathing** to stimulate **vagal nerve acetylcholine release**, and **blue light reduction** to restore **circadian melatonin entrainment**.
