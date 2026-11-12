# SMART HEALTH MONITORING SYSTEM FOR BLOOD SUGAR AND BLOOD PRESSURE PREDICTION USING MACHINE LEARNING

> **AI-Powered Blood Sugar and Blood Pressure Monitoring and Risk Analysis**  
> *Developed for an ECE DevOps & Machine Learning College Assignment*

---

## 1. Project Overview

The **Smart Health Monitoring System** is a responsive, cloud-deployable healthcare analytics platform. It enables users to monitor, analyze, and track longitudinal trends for two vital physiological indicators:
1. **Blood Sugar (Glycemic Health):** Fasting, Post-meal, and Random glucose measurements with ADA guideline mapping and Random Forest classification.
2. **Blood Pressure & Pulse Rate (Cardiovascular Health):** Systolic, Diastolic, and Resting Heart Rate analysis with AHA/ACC hypertension staging and an automatic **Hypertensive Crisis Safety Alert System**.

The platform is engineered using modern DevOps methodologies, version-controlled with Git/GitHub, and deployed serverless on Vercel.

---

## 2. Key Features

- **Blood Sugar Monitoring:**
  - Evaluates Fasting, Post-meal (2hr), and Random blood sugar levels.
  - Automatic Hypoglycemia detection (< 70 mg/dL) with "Rule of 15" emergency guidance.
  - Clinical category assignment based on American Diabetes Association (ADA) standards.
- **Blood Pressure & Pulse Monitoring:**
  - Classifies readings into AHA/ACC 2017 categories: Normal, Elevated, Stage 1, Stage 2, and Crisis.
  - Resting pulse rate evaluation (Bradycardia, Normal, Tachycardia).
  - **Urgent Crisis Safety Protocol:** Real-time alert triggers when Systolic > 180 mmHg and/or Diastolic > 120 mmHg, advising a 1-minute rest retest and immediate emergency care.
- **Machine Learning Risk Estimation:**
  - Two independent Scikit-Learn **Random Forest Classifiers** trained on clinically benchmarked distributions.
  - Stratified train-test split (80/20) with cross-validation to prevent data leakage.
  - Model confidence scores and class probability distributions.
- **Unified Health Dashboard:**
  - Summary vital stat cards (latest glucose, BP, pulse, and total records).
  - Interactive line charts powered by Chart.js.
  - Timeline filtering (All time, 30 days, 7 days).
- **Patient History & Tools:**
  - LocalStorage persistence with multi-tab synchronizer.
  - Search history by ID, status, or date.
  - Filter by record type (Blood Sugar vs. Blood Pressure).
  - **Side-by-Side Reading Comparison** modal.
  - Individual record deletion and complete history purge.
  - **CSV Report Download** for clinical consults.
- **Modern Responsive Design:**
  - Dark navy blue (`#0a1128`), medical blue (`#0077b6`), cyan (`#00b4d8`), and white theme.
  - Fully responsive across desktop, tablet, and mobile devices.

---

## 3. Technology Stack

- **Frontend:**
  - HTML5 & Modern Semantic Elements
  - CSS3 with Glassmorphism, CSS Variables, and Flexbox/Grid
  - Vanilla JavaScript (ES6+)
  - [Chart.js](https://www.chartjs.org/) for reactive health trend charting
  - [FontAwesome 6](https://fontawesome.com/) for healthcare icons
- **Backend:**
  - Python 3.10+
  - Flask 3.x (WSGI Microframework)
- **Machine Learning:**
  - Scikit-Learn (`RandomForestClassifier`)
  - Pandas & NumPy for data preparation
  - Joblib for model serialization
- **DevOps & Cloud:**
  - Git for distributed version control
  - GitHub for remote repository management
  - Vercel for automated serverless deployments

---

## 4. Project Folder Structure

```
Smart-Health-Monitoring-System/
├── api/
│   └── index.py             # Flask application entry point & API endpoints for Vercel WSGI
├── model/
│   ├── train_model.py       # Random forest training & evaluation script
│   ├── glucose_model.joblib # Serialized blood sugar classifier
│   ├── bp_model.joblib      # Serialized blood pressure classifier
│   └── model_metadata.json  # Model accuracy, precision, recall, F1, and feature schema
├── data/
│   ├── README.md            # Dataset provenance, schema, and clinical reference
│   ├── generate_dataset.py  # Synthetic benchmark dataset generator (seed=42)
│   ├── glucose_dataset.csv  # 2,000 synthetic glucose training samples
│   └── bp_dataset.csv       # 2,000 synthetic blood pressure training samples
├── templates/
│   ├── base.html            # Core layout, navbar, footer, scripts
│   ├── index.html           # Landing page with complete screening form & vitals summary
│   ├── sugar.html           # Blood sugar monitoring form, results, and trend
│   ├── blood_pressure.html  # Blood pressure form, crisis alert, and trend
│   ├── dashboard.html       # Unified patient vitals dashboard & timeline charts
│   ├── history.html         # Full history table, search, compare, & CSV export
│   ├── about.html           # ECE DevOps context, architecture, & ML model metrics
│   ├── contact.html         # Contact, academic support, and emergency helpline
│   └── disclaimer.html      # Medical safety disclaimer & clinical criteria tables
├── static/
│   ├── css/
│   │   └── style.css        # Premium dark navy & cyan medical styling
│   ├── js/
│   │   └── script.js        # Data store, Chart.js integrations, APIs, and modal logic
│   └── images/
│       └── logo.svg         # SVG healthcare pulse & cross brand logo
├── app.py                   # Local development execution script
├── requirements.txt         # Pinned Python package dependencies
├── vercel.json              # Vercel serverless build and routing configuration
├── .gitignore               # Excludes bytecode, temporary caches, and virtual envs
└── README.md                # Comprehensive documentation
```

---

## 5. Machine Learning Models & Evaluation

Two independent Random Forest ensembles are utilized to prevent feature interference and maintain domain separation:

### 1. Blood Glucose Model
- **Algorithm:** Random Forest Classifier (120 estimators, max depth 8)
- **Input Features (Strict Order):** `['age', 'gender', 'glucose_level', 'measurement_type']`
- **Target Classes:**
  - Class 0: `Normal / Low Risk`
  - Class 1: `Moderate / Prediabetes Risk`
  - Class 2: `High / Diabetic Range Risk`
- **Validation Metrics (Test Split: 400 samples):**
  - **Accuracy:** 99.50%
  - **Precision (Weighted):** 99.50%
  - **Recall (Weighted):** 99.50%
  - **F1 Score (Weighted):** 99.50%

### 2. Blood Pressure Model
- **Algorithm:** Random Forest Classifier (150 estimators, max depth 9)
- **Input Features (Strict Order):** `['age', 'gender', 'systolic', 'diastolic', 'pulse_rate']`
- **Target Classes:**
  - Class 0: `Normal`
  - Class 1: `Elevated / Stage 1 Hypertension`
  - Class 2: `Stage 2 Hypertension`
  - Class 3: `Hypertensive Crisis (Critical Emergency)`
- **Validation Metrics (Test Split: 400 samples):**
  - **Accuracy:** 99.75%
  - **Precision (Weighted):** 99.76%
  - **Recall (Weighted):** 99.75%
  - **F1 Score (Weighted):** 99.75%

---

## 6. How to Run Locally

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12 installed on your system
- Git installed (optional, for version control)

### Step 1: Clone or Open Workspace
Open your terminal (PowerShell, Command Prompt, or Bash) in the project folder:
```bash
cd "SMART HEALTH MONITERING SYSTEM"
```

### Step 2: (Optional) Create and Activate Virtual Environment
```bash
# Windows PowerShell / CMD
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: (Optional) Re-generate Datasets and Train Models
The repository already includes pre-trained models. If you wish to retrain from scratch:
```bash
# 1. Generate datasets
python data/generate_dataset.py

# 2. Train models and generate metadata
python model/train_model.py
```

### Step 5: Start the Flask Web Server
Run either runner:
```bash
python app.py
# OR
python api/index.py
```

### Step 6: Access the Application
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 7. How to Upload Files to GitHub

Follow these standard Git commands to push your project to a remote GitHub repository:

### Step 1: Initialize Git Repository
```bash
git init
```

### Step 2: Add All Project Files
```bash
git add .
```

### Step 3: Create Initial Commit
```bash
git commit -m "feat: complete Smart Health Monitoring System with ML and Vercel support"
```

### Step 4: Set the Default Branch to `main`
```bash
git branch -M main
```

### Step 5: Link Your Remote GitHub Repository
*(Create a new empty repository on [GitHub](https://github.com/new) named `Smart-Health-Monitoring-System`, then copy your repo URL)*:
```bash
git remote add origin https://github.com/YOUR_USERNAME/Smart-Health-Monitoring-System.git
```

### Step 6: Push Code to GitHub
```bash
git push -u origin main
```

---

## 8. How to Deploy on Vercel

The application is pre-configured with `vercel.json` for seamless serverless deployment.

### Option A: Via GitHub (Recommended)
1. Go to [Vercel](https://vercel.com/) and sign in with your GitHub account.
2. Click **Add New...** &rarr; **Project**.
3. Select your `Smart-Health-Monitoring-System` repository.
4. Leave the Framework Preset as **Other** (Vercel automatically detects `vercel.json`).
5. Click **Deploy**.
6. Within 60 seconds, your application will be live at `https://your-project.vercel.app`.

### Option B: Via Vercel CLI
```bash
# Install Vercel CLI globally
npm install -g vercel

# Deploy directly from terminal
vercel
```

---

## 9. Verification & Testing Checklist

- [x] **Home Route (`/`):** Verifies hero display, navigation, and vitals summary cards.
- [x] **Blood Sugar Route (`/sugar`):** Tests inputs (Fasting 95 mg/dL &rarr; Normal; 115 mg/dL &rarr; Prediabetes; 180 mg/dL &rarr; High Risk; 60 mg/dL &rarr; Hypoglycemia Warning).
- [x] **Blood Pressure Route (`/blood-pressure`):** Tests inputs (118/75 &rarr; Normal; 135/85 &rarr; Stage 1; 155/95 &rarr; Stage 2; **190/125 &rarr; Urgent Hypertensive Crisis Alert Banner**).
- [x] **Dashboard Route (`/dashboard`):** Verifies longitudinal trend chart rendering and timeline date filters.
- [x] **History Route (`/history`):** Verifies searching, type filtering, side-by-side comparison modal, and CSV export.
- [x] **About Route (`/about`):** Verifies dynamic model metrics, confusion matrix display, and DevOps architecture.
- [x] **Contact Route (`/contact`):** Verifies developer info, inquiry form, and emergency contacts.
- [x] **Disclaimer Route (`/disclaimer`):** Verifies ADA & AHA clinical tables and emergency symptoms protocol.
- [x] **Automated Unit Tests (`python test_app.py`):** 20 comprehensive automated unit tests covering all routes, ML predictions, safety alerts, and CSV exports.

---

## 10. Medical Safety & Ethical Disclaimer

> **IMPORTANT CLINICAL NOTICE:**  
> This software is an **academic educational demonstration and screening tool**. It is **NOT** a medical diagnostic device and does not substitute for professional medical consultation, diagnosis, or treatment. 
> 
> The datasets utilized for model training are simulated benchmark distributions adhering to ADA and AHA statistical boundaries; they do not represent identifiable patient records. Never alter prescription medications or disregard professional clinical advice based on predictions generated by this software.

