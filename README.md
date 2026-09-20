# 🧬 AMR-UTI Prediction Engine
## Fault-Tolerant Hybrid Quantum-Classical AI for Antimicrobial Resistance Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/PennyLane-Quantum%20ML-6C3483?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/XGBoost-Classical%20ML-EC6C00?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/AWS%20Braket-Quantum%20Cloud-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
</p>

<p align="center">
  <b>AI × Quantum Computing × Clinical Data × Antimicrobial Stewardship</b>
</p>

<p align="center">
  A research-oriented hybrid quantum-classical framework for early antimicrobial resistance risk estimation in urinary tract infections.
</p>

---

# 📌 Table of Contents

- [About the Project](#-about-the-project)
- [The Core Challenge](#-the-core-challenge)
- [The Solution](#-the-solution)
- [Key Innovations](#-key-innovations)
- [Project Objectives](#-project-objectives)
- [Tech Stack and Architecture](#-tech-stack-and-architecture)
- [System Architecture](#-system-architecture)
- [Architecture Layers](#-architecture-layers)
- [User & Frontend Layer](#1--user--frontend-layer)
- [Backend & ETL Layer](#2--backend--etl-layer)
- [Feature Reduction Layer](#3--feature-reduction-layer)
- [Quantum State Preparation](#4--quantum-state-preparation)
- [Quantum Execution and Fault Tolerance](#-quantum-execution-and-fault-tolerance)
- [Clinical Output Layer](#-clinical-output-layer)
- [How We Used AWS](#-how-we-used-aws)
- [Dataset and Data Pipeline](#-dataset-and-data-pipeline)
- [Machine Learning Strategy](#-machine-learning-strategy)
- [Performance and Evaluation](#-performance-and-evaluation)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Running the Project](#-running-the-project)
- [End-to-End Workflow](#-end-to-end-workflow)
- [Research Motivation](#-research-motivation)
- [Learning and Growth](#-learning-and-growth)
- [Limitations](#-limitations)
- [Future Work](#-future-work)
- [Safety and Intended Use](#-safety-and-intended-use)
- [Hackathon Alignment](#-hackathon-alignment)
- [Contributors](#-contributors)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---

# 🧬 About the Project

The **AMR-UTI Prediction Engine** is a fault-tolerant hybrid quantum-classical artificial intelligence framework designed to investigate early prediction of **antimicrobial resistance (AMR)** in **urinary tract infections (UTIs)**.

The project combines longitudinal electronic health record (EHR) information, classical machine learning, tensor-network feature compression, quantum machine learning, probability calibration, and uncertainty quantification into a single decision-support pipeline.

Instead of treating AMR prediction as a conventional binary classification problem alone, the framework is designed to answer three connected questions:

1. **What is the patient's estimated resistance risk?**
2. **How reliable is the probability produced by the model?**
3. **When is the model sufficiently uncertain that definitive laboratory testing should remain the source of truth?**

The architecture is explicitly fault-tolerant. Quantum execution is not treated as a single point of failure. Depending on optimization stability and runtime conditions, inference can dynamically route between a Variational Quantum Neural Network (QNN), a Quantum Support Vector Machine (QSVM), and a classical XGBoost fallback.

> **Core Research Question**
>
> Can a fault-tolerant hybrid quantum-classical architecture provide calibrated and uncertainty-aware antimicrobial resistance predictions from longitudinal clinical data?

---

# 🚨 The Core Challenge

In emergency and clinical medicine, diagnosis and treatment of urinary tract infections can involve **empirical prescribing** while clinicians wait for definitive laboratory culture and antimicrobial susceptibility results.

The practical challenge is an information gap:

```text
Patient Presents
       │
       ▼
Clinical Assessment
       │
       ▼
Empirical Treatment
       │
       │
       │   ⏳ Laboratory Processing
       │
       ▼
Culture + Susceptibility Results
       │
       ▼
Targeted Treatment
```

The conventional workflow can create a substantial delay before definitive susceptibility information becomes available.

During this period, clinical decisions may need to be made using incomplete information. This creates an opportunity for computational systems to estimate resistance risk using data that is already available.

The project therefore investigates whether longitudinal EHR information can provide an **early resistance-risk estimate** before definitive laboratory susceptibility results become available.

---

# 💡 The Solution

The **AMR-UTI Prediction Engine** is designed as a layered AI system that transforms longitudinal clinical information into a calibrated antimicrobial resistance estimate.

The framework incorporates:

- longitudinal EHR-derived features,
- temporal feature representation,
- Matrix Product State (MPS) compression,
- quantum feature embedding,
- Variational Quantum Neural Networks,
- Quantum Support Vector Machines,
- XGBoost fallback inference,
- SMOTE for class imbalance,
- Platt Scaling / probability calibration,
- Precision-Recall threshold optimization,
- and Shannon Entropy-based uncertainty quantification.

The overall concept is:

```text
Longitudinal EHR Data
         │
         ▼
   ETL & Cleaning
         │
         ▼
Temporal Representation
         │
         ▼
      MPS Layer
         │
         ▼
 Quantum State Preparation
         │
         ▼
 ┌────────────────────────┐
 │ Quantum Health Check   │
 │ Gradient Variance      │
 └────────────┬───────────┘
              │
       ┌──────┼───────┐
       │      │       │
       ▼      ▼       ▼
      QNN    QSVM   XGBoost
       │      │       │
       └──────┼───────┘
              ▼
    Probability Calibration
              │
              ▼
     Entropy / Uncertainty
              │
              ▼
      Clinical Risk Output
```

---

# 🔬 Key Innovations

## 1. SMOTE for Severe Class Imbalance

AMR datasets can contain substantially more susceptible observations than resistant observations.

This creates a class-imbalance problem in which a model may achieve high aggregate accuracy while performing poorly on the clinically important minority class.

The framework therefore incorporates **SMOTE (Synthetic Minority Oversampling Technique)** to improve representation of the resistant class during training.

---

## 2. Platt Scaling for Probability Calibration

A classification probability is not automatically a reliable estimate of real-world event frequency.

The framework incorporates probability calibration using **Platt Scaling** so that the downstream system can work with more meaningful probability estimates.

This is especially important because these probabilities are subsequently used for:

- risk interpretation,
- threshold decisions,
- and entropy-based uncertainty estimation.

---

## 3. Precision-Recall Threshold Optimization

Instead of assuming that the default threshold of `0.50` is optimal, the framework analyzes the Precision-Recall relationship and tunes the classification threshold.

The objective is to explicitly consider the trade-off between:

- minority-class recall,
- precision,
- false positives,
- false negatives,
- and F1 score.

---

## 4. Shannon Entropy Uncertainty Quantification

The system does not only ask whether a case is classified as resistant.

It also asks:

> **How uncertain is the model about this prediction?**

For binary probability \(p\), Shannon entropy is:

\[
H = -\left[p\log_2(p) + (1-p)\log_2(1-p)\right]
\]

where:

- \(p\) = calibrated probability of resistance,
- \(H\) = Shannon entropy in bits.

The framework uses:

\[
H > 0.40 \text{ bits}
\]

as a high-uncertainty condition.

Such predictions are flagged for additional attention and definitive laboratory confirmation.

---

## 5. Fault-Tolerant Quantum-Classical Execution

The framework does not depend on the success of a single quantum model.

It dynamically routes inference across:

```text
Healthy Quantum Optimization
          │
          ▼
         QNN

Barren Plateau / Unstable Gradient
          │
          ▼
         QSVM

Quantum Runtime / Hardware Failure
          │
          ▼
       XGBoost
```

This provides a classical safety path when quantum execution becomes unreliable or unavailable.

---

# 🎯 Project Objectives

The project is designed around four primary objectives.

### Objective 1 — Early Risk Estimation

Estimate antimicrobial resistance risk from information available before complete susceptibility results.

### Objective 2 — Fault-Tolerant AI

Maintain inference capability when quantum optimization or quantum runtime execution becomes unstable.

### Objective 3 — Calibrated Predictions

Convert raw model outputs into probability estimates that are more suitable for downstream decision support.

### Objective 4 — Uncertainty-Aware Decision Support

Identify cases where the model is uncertain and should defer to definitive laboratory evidence.

---

# ⚙️ Tech Stack and Architecture

## Languages & Core Libraries

| Technology | Purpose |
|---|---|
| **Python 3.11+** | Core development and execution environment |
| **PyTorch** | Tensor operations, neural components, automatic differentiation |
| **PennyLane** | Quantum circuits and hybrid quantum-classical execution |
| **XGBoost** | Classical fallback prediction engine |
| **Scikit-Learn** | Preprocessing, scaling, metrics, evaluation |
| **NumPy** | Numerical computation |
| **Pandas** | Tabular data processing and ETL |

---

## Quantum Computing Stack

| Technology | Purpose |
|---|---|
| **PennyLane** | Variational quantum circuits and QML |
| **AWS Braket** | Cloud quantum execution infrastructure |
| **Matrix Product States** | Tensor-network temporal feature compression |
| **Quantum Kernel / QSVM** | Non-variational quantum fallback |

---

## Frontend

| Technology | Purpose |
|---|---|
| **Streamlit** | Interactive dashboard, patient intake, dynamic risk evaluation |

---

# 🏗️ System Architecture

The framework is organized into four primary layers:

```text
┌───────────────────────────────────────────────────────────┐
│                  1. USER / FRONTEND LAYER                │
│                                                           │
│  Streamlit Dashboard                                      │
│  • Patient intake                                         │
│  • Clinical variables                                     │
│  • Chronological history                                  │
│  • Dynamic lookback windows                               │
└───────────────────────────────┬───────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────┐
│                  2. BACKEND / ETL LAYER                   │
│                                                           │
│  • Data ingestion                                         │
│  • Leakage prevention                                     │
│  • Missing-value processing                               │
│  • Sequential interpolation                               │
│  • Standardization                                        │
└───────────────────────────────┬───────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────┐
│                 3. AI / ML PROCESSING LAYER               │
│                                                           │
│  • MPS temporal compression                              │
│  • Quantum state preparation                             │
│  • Variational QNN                                       │
│  • QSVM                                                  │
│  • XGBoost fallback                                      │
│  • Gradient variance monitoring                           │
└───────────────────────────────┬───────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────┐
│                 4. CLINICAL OUTPUT LAYER                 │
│                                                           │
│  • Resistance probability                                │
│  • Susceptibility score                                  │
│  • Probability calibration                               │
│  • Shannon entropy                                       │
│  • Threshold-based output                                │
└───────────────────────────────────────────────────────────┘
```

---

# 🧩 Architecture Layers

# 1. 👨‍⚕️ User & Frontend Layer

The primary user interacts with a **Streamlit clinical decision-support dashboard**.

The interface is designed to capture patient-level information and provide a dynamic representation of chronological history.

The system can work with temporal lookback windows such as:

```text
ALL
180 days
90 days
30 days
14 days
7 days
```

These windows allow the longitudinal history to be processed before entering the temporal feature-reduction layer.

---

# 2. 🔄 Backend & ETL Layer

## Data Ingestion

Raw EHR feature records and resistance-label records are merged using:

```text
example_id
```

The ingestion process also removes metadata or cohort-splitting variables that may create leakage between training and inference.

Examples include:

```text
is_train
uncomplicated
```

This prevents model training from exploiting information about how the dataset itself was constructed.

---

## Sequential Imputation

Longitudinal healthcare data frequently contains temporal gaps.

The preprocessing pipeline performs **linear interpolation across sequential time steps** where applicable.

Residual missing events are assigned zero values in the relevant representation so that the chronological structure is preserved without flattening the patient history.

---

## StandardScaler Normalization

Numerical features are standardized before downstream dimensionality reduction.

\[
z_i = \frac{x_i - \mu}{\sigma}
\]

where:

- \(x_i\) = original feature,
- \(\mu\) = feature mean,
- \(\sigma\) = feature standard deviation,
- \(z_i\) = standardized value.

---

# 3. 🧬 Feature Reduction Layer

## Matrix Product State (MPS) Tensor Network

Longitudinal clinical data can have a very large feature dimension.

The framework therefore applies a **Matrix Product State (MPS)** representation to compress temporal information before quantum encoding.

The patient's history is organized into chronological windows:

```text
Patient Timeline
│
├── 180 days
├── 90 days
├── 30 days
├── 14 days
└── 7 days
```

The MPS representation iteratively contracts temporal features while maintaining a hidden memory state that is propagated across successive windows.

Conceptually:

```text
Temporal Features
       │
       ▼
 ┌─────────────┐
 │   Window 1  │
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │   Window 2  │
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │   Window 3  │
 └──────┬──────┘
        │
        ▼
 Hidden Temporal State
```

This reduces the dimensionality of the input while maintaining temporal information.

---

# 4. ⚛️ Quantum State Preparation

The compressed classical temporal representation is transformed into a quantum feature representation.

The features are scaled into:

\[
[0,\pi]
\]

and then incorporated through **angle-based quantum feature embedding**.

The resulting representation becomes the input to the variational quantum execution stage.

---

# ⚛️ Quantum Execution and Fault Tolerance

## Gradient Variance Monitoring

Variational quantum circuits can suffer from optimization instability associated with **barren plateaus**.

The framework monitors the variance of the circuit gradients:

\[
\operatorname{Var}(\nabla_\theta)
\]

The system considers the quantum optimization landscape unstable when:

\[
\operatorname{Var}(\nabla_\theta)<10^{-4}
\]

This condition becomes a routing signal for the fault-tolerant execution layer.

---

# 🛣️ Three-Path Execution Cascade

```text
                    Quantum Feature State
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Gradient Variance   │
                  │      Monitor        │
                  └─────────┬───────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
       Healthy          Barren            Runtime /
       Gradient         Plateau            Hardware Error
          │                 │                 │
          ▼                 ▼                 ▼
       QNN               QSVM             XGBoost
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
                            ▼
                    Probability Output
```

---

## Path A — Variational Quantum Neural Network

When the quantum circuit demonstrates healthy gradient behaviour, inference follows:

**Variational Quantum Neural Network (QNN)**.

This combines:

- PyTorch,
- PennyLane,
- parameterized quantum circuits,
- automatic differentiation,
- gradient-based parameter optimization.

The system can optionally fuse additional static clinical features into the quantum embedding.

---

## Path B — Quantum Support Vector Machine

When gradient variance indicates a barren plateau or unstable variational optimization, the framework switches to:

**Quantum Support Vector Machine (QSVM)**.

The QSVM uses a precomputed quantum-kernel Gram matrix, reducing dependence on unstable variational training dynamics.

---

## Path C — Penalized XGBoost

When quantum execution encounters:

- hardware errors,
- execution timeouts,
- backend availability issues,
- or other quantum runtime failures,

the system falls back to:

**Penalized XGBoost**.

The classical fallback uses L1/L2 regularization to provide a robust alternative when quantum execution is unavailable.

---

# 🏥 Clinical Output Layer

## Resistance Probability

The model generates:

\[
P(Y=1\mid X)
\]

where:

- \(Y=1\) = resistant phenotype,
- \(X\) = patient feature representation.

---

## Susceptibility Score

The framework converts resistance probability into a complementary susceptibility score:

\[
S(X)=1-P(Y=1\mid X)
\]

where:

- \(P(Y=1\mid X)\) = predicted probability of resistance,
- \(S(X)\) = estimated susceptibility score.

---

# 📐 Probability Calibration

The model output is calibrated before uncertainty assessment.

The project uses **Platt Scaling / temperature-based probability calibration** to reduce probability distortion and make the output more suitable for threshold and uncertainty analysis.

The conceptual goal is:

```text
Raw Model Probability
        │
        ▼
Probability Calibration
        │
        ▼
Calibrated Probability
        │
        ├───────────────┐
        ▼               ▼
 Decision Threshold   Entropy
```

---

# 🌡️ Shannon Entropy Uncertainty Budget

For a binary probability \(p\), Shannon entropy is:

\[
H = -\left[p\log_2(p)+(1-p)\log_2(1-p)\right]
\]

The resulting quantity is measured in **bits**.

A higher entropy indicates that the model's probability distribution is more uncertain.

The project uses:

\[
H>0.40\text{ bits}
\]

as the threshold for high uncertainty.

When the prediction exceeds this uncertainty budget, the system recommends obtaining **definitive laboratory culture / susceptibility evidence** rather than relying on the model alone.

---

# 🚨 Clinical Alert Decision

The calibrated resistance probability is compared against a safety-oriented decision threshold.

The final output combines:

```text
Calibrated Resistance Probability
                +
       Decision Threshold
                +
       Shannon Entropy
                ↓
      Decision-Support Output
```

Possible outputs can be presented as resistance-risk and uncertainty information rather than an autonomous treatment recommendation.

---

# ☁️ How We Used AWS

## AWS Braket Integration

AWS is used as the cloud quantum-computing layer through **Amazon Braket**.

The framework is designed to decouple local development from the execution backend.

```text
                    Hybrid Application
                           │
                           ▼
                  Quantum Abstraction
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
      Local Simulation              AWS Braket
                                         │
                                ┌────────┴────────┐
                                ▼                 ▼
                              SV1               QPU
```

---

## Cloud Quantum Execution

The system can leverage AWS Braket simulators such as **SV1** and cloud-accessible quantum processing units for quantum circuit execution.

This creates a path from local experimentation toward cloud-accessible quantum computing.

---

## Why AWS Braket?

The main architectural motivation is **decoupling**.

Rather than hard-coding the entire application around a single local quantum simulator, the architecture allows the quantum execution layer to communicate with a cloud quantum backend.

This is important because quantum workloads can require:

- additional compute capacity,
- specialized simulators,
- access to real QPUs,
- and experimentation across different execution environments.

---

## NISQ-Era Readiness

The architecture is designed with the **Noisy Intermediate-Scale Quantum (NISQ)** era in mind.

The system does not assume:

- quantum hardware is always available,
- optimization is always stable,
- execution is always fast,
- or quantum hardware never fails.

Instead, it creates explicit fallback mechanisms:

```text
Quantum Model
     │
     ├── Healthy → QNN
     │
     ├── Barren Plateau → QSVM
     │
     └── Runtime Failure → XGBoost
```

This provides a more resilient path toward future quantum-enabled healthcare experimentation.

---

# 📊 Dataset and Data Pipeline

The framework works with longitudinal UTI-related clinical data and antimicrobial resistance labels.

The raw-data layer contains:

```text
data/
├── raw/
│   ├── all_uti_features.csv
│   ├── all_uti_resist_labels.csv
│   ├── all_prescriptions.csv
│   └── data_dictionary.csv
│
└── processed/
    └── processed_clinical_data.csv
```

The high-level data pipeline is:

```text
Raw EHR Features
       +
Resistance Labels
       +
Prescription Information
       │
       ▼
Data Integration
       │
       ▼
Leakage Prevention
       │
       ▼
Sequential Imputation
       │
       ▼
Standardization
       │
       ▼
Temporal Window Construction
       │
       ▼
MPS Representation
       │
       ▼
Quantum / Classical Inference
```

---

# ⚖️ Machine Learning Strategy

## Why Accuracy Alone Is Not Enough

Medical datasets can be highly imbalanced.

If susceptible cases substantially outnumber resistant cases, a classifier can obtain apparently strong accuracy while failing to identify the minority resistant phenotype.

For this reason, the project emphasizes:

- minority-class recall,
- precision,
- F1 score,
- Precision-Recall analysis,
- probability calibration,
- and uncertainty.

---

## SMOTE

**SMOTE** is used to improve representation of the minority class during training.

The conceptual workflow is:

```text
Original Training Data
        │
        ▼
Class Distribution Analysis
        │
        ▼
      SMOTE
        │
        ▼
Balanced Training Representation
        │
        ▼
Model Training
```

SMOTE is applied as part of the training workflow rather than to the held-out evaluation population.

---

## Threshold Optimization

The default classification threshold is:

\[
0.50
\]

The framework also analyzes the Precision-Recall curve and selects an optimized decision threshold based on the project objective.

This creates a deliberate balance between:

```text
False Negatives
       ↕
False Positives
       ↕
Minority Recall
       ↕
Precision
       ↕
F1 Score
```

---

# 📈 Performance and Evaluation

## Test Cohort

The reported test cohort contains:

**22,153 patient records**

| Configuration | Value |
|---|---:|
| Test cohort | **22,153 records** |
| Default threshold | **0.50** |
| Optimized threshold | Precision-Recall tuned |
| Optimization objective | **F1 + minority-class recall** |

---

## Classification Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| **Susceptible (0)** | 0.89 | 0.85 | 0.87 | 17,476 |
| **Resistant (1)** | 0.53 | 0.61 | 0.57 | 4,677 |
| **Overall Accuracy** | — | — | **80.0%** | **22,153** |

---

## Key Result

The threshold-tuned system reports:

- **80.0% overall accuracy**
- **61% recall for the Resistant minority class**

The result reflects an intentional trade-off between false negatives and false positives rather than optimizing accuracy alone.

> **Clinical Note:** These results represent a research and decision-support evaluation. They should not be interpreted as clinical validation or as evidence that the system can replace laboratory susceptibility testing.

---

# 📏 Evaluation Metrics

| Metric | Purpose |
|---|---|
| **Accuracy** | Overall prediction correctness |
| **Precision** | Reliability of positive resistance predictions |
| **Recall** | Ability to identify resistant cases |
| **F1 Score** | Balance between precision and recall |
| **ROC-AUC** | Overall discrimination across thresholds |
| **PR-AUC** | Performance under class imbalance |
| **Confusion Matrix** | Detailed prediction error analysis |
| **Shannon Entropy** | Prediction uncertainty |

---

# 📁 Project Structure

```text
AMR-UTI-First-Commit-Hackathon/
│
├── data/
│   ├── raw/
│   │   ├── all_uti_features.csv
│   │   ├── all_uti_resist_labels.csv
│   │   ├── all_prescriptions.csv
│   │   └── data_dictionary.csv
│   │
│   └── processed/
│       └── processed_clinical_data.csv
│
├── models/
│   ├── standard_scaler.pkl
│   ├── quantum_state_scaler.pkl
│   ├── xgb_baseline.json
│   ├── hybrid_mps_qnn.pt
│   ├── hybrid_mps_qnn_meta.json
│   ├── qsvm_weights.pkl
│   ├── qsvm_train_embeddings.npy
│   ├── qsvm_mps_embedder.pt
│   └── qsvm_mps_embedder_meta.json
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   │
│   ├── module1_etl/
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   ├── imputation.py
│   │   └── distribution.py
│   │
│   ├── module2_features/
│   │   ├── __init__.py
│   │   ├── reduction.py
│   │   └── state_prep.py
│   │
│   └── module3_execution/
│       ├── __init__.py
│       ├── quantum_circuit.py
│       ├── qsvm_fallback.py
│       └── xgboost_engine.py
│
├── apps.py
├── main.py
├── evaluate_metrics.py
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

## Prerequisites

Install:

- Python **3.11+**
- Git
- compatible PyTorch
- PennyLane
- XGBoost
- Scikit-Learn
- Streamlit
- AWS Braket SDK when cloud quantum execution is enabled

---

## 1. Clone the Repository

```bash
git clone https://github.com/KanakDharamthok/AMR-UTI-First-Commit-Hackathon.git
cd AMR-UTI-First-Commit-Hackathon
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Launch the Streamlit Dashboard

```bash
streamlit run apps.py
```

The interface should be available at:

```text
http://localhost:8501
```

---

## Run the Main Pipeline

```bash
python main.py
```

---

## Evaluate the Trained Models

```bash
python evaluate_metrics.py
```

---

# ⚛️ Quantum Execution Strategy

The complete execution hierarchy is:

```text
                         ┌──────────────────────┐
                         │   Clinical Patient   │
                         │        Input         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   ETL & Imputation   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    MPS Reduction     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Quantum State Prep  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Gradient Variance   │
                         │        Check         │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┼───────────────────┐
                 │                  │                   │
                 ▼                  ▼                   ▼
          Healthy Gradient     Barren Plateau      Hardware Error
                 │                  │                   │
                 ▼                  ▼                   ▼
          ┌────────────┐       ┌──────────┐       ┌──────────┐
          │    QNN     │       │   QSVM   │       │ XGBoost  │
          └─────┬──────┘       └────┬─────┘       └────┬─────┘
                │                   │                  │
                └───────────────────┼──────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Probability          │
                         │ Calibration          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Entropy & Threshold  │
                         │ Analysis             │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Clinical Decision    │
                         │ Support              │
                         └──────────────────────┘
```

---

# 🔬 Research Motivation

Antimicrobial resistance is a major challenge for empirical treatment of bacterial infections.

For UTIs, definitive laboratory susceptibility information can become available only after laboratory processing. During this period, risk estimation from routinely available clinical data could potentially provide earlier evidence.

This framework therefore investigates whether the combination of:

- longitudinal EHR representations,
- tensor-network feature compression,
- quantum machine learning,
- classical machine learning,
- probability calibration,
- uncertainty quantification,
- and safety-oriented threshold optimization

can produce useful early resistance-risk estimates while retaining an explicit fallback mechanism.

The central research direction is:

> **Can a fault-tolerant hybrid quantum-classical architecture provide calibrated, uncertainty-aware antimicrobial resistance predictions from longitudinal clinical data?**

---

# 📚 Learning and Growth

## 1. Bridging Quantum Computing with Clinical Reality

Building this system provided experience in translating abstract quantum-computing concepts into a practical healthcare-oriented application.

The implementation involved concepts such as:

- angle embedding,
- parameterized quantum circuits,
- parameter-shift gradients,
- tensor networks,
- Matrix Product States,
- quantum kernels,
- variational optimization,
- barren-plateau monitoring.

The most important lesson was that technical novelty by itself is insufficient in high-stakes domains.

Healthcare AI also requires attention to:

- reliability,
- safety,
- interpretability,
- calibration,
- uncertainty,
- reproducibility.

---

## 2. Handling Extreme Data Imbalance

Medical datasets can be heavily skewed toward susceptible cases.

Working on AMR prediction demonstrated why accuracy alone can produce a misleading picture.

Combining:

```text
SMOTE
   +
Precision-Recall Analysis
   +
Threshold Optimization
   +
Probability Calibration
```

provided a stronger understanding of minority-class modelling and decision thresholds.

This shifted the design objective from:

> "Which model gives the highest accuracy?"

to:

> **"Which modelling strategy provides useful and reliable resistance-risk estimates for the target problem?"**

---

## 3. Uncertainty Quantification

A major conceptual step was moving from simple binary classification toward:

```text
Prediction
    +
Probability
    +
Uncertainty
```

The entropy-based uncertainty budget showed how an AI system can explicitly identify cases in which its prediction may not be sufficiently decisive.

The threshold:

\[
H>0.40\text{ bits}
\]

creates an explicit uncertainty signal that can trigger further laboratory confirmation.

A key learning was:

> **A useful clinical AI system should know when it is uncertain.**

---

## 4. Fault-Tolerant AI Architecture

Quantum machine learning introduces unique failure modes, including:

- barren plateaus,
- unstable gradient landscapes,
- runtime failures,
- execution timeouts,
- backend availability constraints.

Rather than allowing the quantum pipeline to become a single point of failure, the architecture introduces:

```text
QNN
 ↓
QSVM
 ↓
XGBoost
```

This helped build an understanding of resilient AI systems where emerging technology is surrounded by reliable fallback mechanisms.

---

## 5. AWS and Cloud Quantum Computing

Integrating AWS Braket provided experience in separating:

```text
Application Logic
       │
       ▼
Quantum Abstraction
       │
       ▼
Cloud Execution
```

It introduced practical exposure to cloud quantum infrastructure and demonstrated how a local experimental workflow can be connected to cloud-accessible quantum simulators and QPUs.

---

## 6. From Model Experimentation to an End-to-End System

The project required moving beyond isolated experiments and integrating:

```text
Data
+
ETL
+
Feature Engineering
+
Machine Learning
+
Quantum Computing
+
Cloud Infrastructure
+
Frontend
+
Calibration
+
Uncertainty
+
Safety
```

This created a broader understanding of AI engineering as a system-building discipline rather than only a model-training exercise.

---

# 🧠 What We Learned Technically

| Area | Learning |
|---|---|
| Clinical ML | Evaluation must consider class imbalance and domain-specific risk |
| Data Engineering | Leakage prevention is essential before training |
| Temporal ML | Longitudinal history can be represented through chronological windows |
| Tensor Networks | MPS can compress structured high-dimensional temporal representations |
| QML | Quantum models introduce optimization and runtime constraints |
| Fault Tolerance | Classical fallback paths can improve system resilience |
| Calibration | Raw confidence scores are not automatically reliable probabilities |
| Uncertainty | Entropy provides an explicit measure of prediction ambiguity |
| Cloud | AWS Braket enables cloud-accessible quantum execution |
| Product Engineering | A research model must be integrated into a complete usable pipeline |

---

# ⚠️ Limitations

## Dataset Limitations

Performance depends on:

- dataset quality,
- cohort definition,
- population characteristics,
- missingness,
- and availability of longitudinal information.

## Generalization

A model trained on a particular cohort should not automatically be considered generalizable to other hospitals, geographies, demographics, or healthcare systems.

## Quantum Hardware Constraints

Real quantum hardware may introduce:

- noise,
- latency,
- device availability limitations,
- optimization instability,
- hardware-specific execution behaviour.

## Clinical Validation

The reported test metrics represent model-development evaluation and should not be interpreted as clinical validation.

## Calibration

Probability calibration should be assessed independently across different cohorts before any clinical use.

---

# 🚀 Future Work

## 1. External Validation

Evaluate the model on independent datasets from other healthcare systems.

## 2. Prospective Temporal Evaluation

Study how performance changes on future patient cohorts.

## 3. Antibiotic-Specific Predictions

Extend the framework from a general resistance outcome to antibiotic-specific susceptibility predictions.

## 4. Explainable AI

Add feature-level and patient-level explanation mechanisms.

## 5. Advanced Calibration

Evaluate calibration stability across populations, sites, and temporal periods.

## 6. Larger Quantum Experiments

Expand quantum execution to additional AWS Braket simulators and hardware platforms.

## 7. Model Drift Monitoring

Introduce cloud-based monitoring for data drift and model performance drift.

## 8. Clinical Workflow Integration

Investigate how the prediction layer could integrate into research or clinical decision-support workflows under appropriate validation and governance.

---

# 🏆 Hackathon Alignment

This project is structured to directly demonstrate the four core dimensions of the challenge.

## 1. About the Project

The project addresses a real-world healthcare problem:

**early antimicrobial resistance risk estimation for urinary tract infections.**

It combines longitudinal clinical data with AI and uncertainty-aware decision support.

---

## 2. Tech Stack and Architecture

The system demonstrates a complete architecture spanning:

- Python,
- PyTorch,
- PennyLane,
- XGBoost,
- Scikit-Learn,
- Streamlit,
- Matrix Product States,
- QSVM,
- probability calibration,
- uncertainty quantification.

The architecture is modular and fault-tolerant, allowing inference to transition between quantum and classical execution paths.

---

## 3. How We Used AWS

AWS Braket serves as the cloud quantum infrastructure.

It enables access to:

- quantum simulators,
- cloud-accessible QPUs,
- and an abstraction layer between local application execution and quantum backends.

This provides a pathway toward scalable quantum experimentation.

---

## 4. Learning and Growth

The project demonstrates growth across:

- machine learning,
- clinical data processing,
- quantum machine learning,
- tensor networks,
- probability calibration,
- uncertainty quantification,
- fault-tolerant architecture,
- cloud quantum infrastructure,
- and responsible AI.

---

# 🛡️ Safety and Intended Use

> **IMPORTANT:** This project is a research and decision-support prototype.

It is **not** intended to:

- diagnose patients autonomously,
- prescribe antibiotics,
- replace microbiological culture,
- replace antimicrobial susceptibility testing,
- replace physicians or infectious-disease specialists,
- or function as an independent clinical decision-maker.

Predictions generated by the system should be treated as **computational estimates** and not as medical advice.

Definitive clinical decisions must remain grounded in:

- clinical evaluation,
- microbiological culture,
- antimicrobial susceptibility testing,
- qualified medical judgment,
- and institutional antimicrobial stewardship protocols.

Any real-world deployment would require extensive:

- external validation,
- prospective clinical evaluation,
- regulatory review,
- privacy and security controls,
- clinical governance,
- and monitoring.

---

# 👥 Contributors

### Core Team

- **Kanak Dharamthok**
- **Khushbu Gupta**
- **Nandini Jaiswal**
- **Sanvi Goja**

---

# 🙏 Acknowledgements

This project integrates concepts and technologies from:

- PyTorch
- PennyLane
- XGBoost
- Scikit-Learn
- AWS Braket
- Streamlit
- Tensor Networks
- Matrix Product State methods
- Antimicrobial Resistance research
- Clinical Machine Learning
- Probability Calibration
- Uncertainty Quantification

---

# 📜 License

Add the applicable project license here.

Example:

```text
MIT License
```

---

# 🌟 Project Vision

The long-term vision of **AMR-UTI** is to investigate how **AI, quantum computing, cloud infrastructure, and uncertainty-aware modelling** can contribute to earlier antimicrobial resistance risk assessment.

```text
                    ┌────────────────────┐
                    │   Clinical Data     │
                    └──────────┬─────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │     AI / ML        │
                    └──────────┬─────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │ Quantum + Classical│
                    │      Inference     │
                    └──────────┬─────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │ Calibration +      │
                    │ Uncertainty        │
                    └──────────┬─────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │ Decision Support   │
                    └──────────┬─────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │ Antimicrobial      │
                    │ Stewardship         │
                    └────────────────────┘
```

<p align="center">
  <b>🧬 AMR-UTI Prediction Engine</b>
  <br>
  <sub>Researching the intersection of Antimicrobial Resistance, Quantum ML, Clinical AI, and AWS.</sub>
</p>
