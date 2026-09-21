
# AMR-UTI Prediction Engine

## Hybrid Quantum-Classical Antimicrobial Resistance Decision-Support Prototype

The **AMR-UTI Prediction Engine** is a research-oriented machine-learning and quantum-computing project for antimicrobial resistance (AMR) modelling in urinary tract infection (UTI) data.

The project combines longitudinal clinical-data processing, classical machine learning, quantum-machine-learning experiments, uncertainty analysis, model certification, and a clinician-oriented decision-support interface.

The current certified inference system exposes a **788-feature input contract** and produces susceptibility probabilities for four antibiotic targets:

* **NIT** — Nitrofurantoin
* **SXT** — Trimethoprim/Sulfamethoxazole
* **CIP** — Ciprofloxacin
* **LVX** — Levofloxacin

The project is intended for **research and demonstration purposes**. It is not a clinically validated diagnostic or treatment system.

---

## Table of Contents

* [Overview](#overview)
* [Research Objective](#research-objective)
* [System Architecture](#system-architecture)
* [Current Certified Inference System](#current-certified-inference-system)
* [Quantum and Classical Modelling](#quantum-and-classical-modelling)
* [Model Certification](#model-certification)
* [Frontend and API](#frontend-and-api)
* [API Endpoints](#api-endpoints)
* [Local Demonstration](#local-demonstration)
* [Cloud Deployment](#cloud-deployment)
* [Project Structure](#project-structure)
* [Reproducibility and Integrity](#reproducibility-and-integrity)
* [Evaluation](#evaluation)
* [Limitations](#limitations)
* [Safety and Intended Use](#safety-and-intended-use)
* [Future Work](#future-work)
* [Contributors](#contributors)
* [License](#license)

---

# Overview

Antimicrobial resistance is a major challenge in the treatment of bacterial infections. For urinary tract infections, susceptibility information may not be immediately available when an initial clinical assessment is performed.

This project investigates whether machine-learning models can provide **computational susceptibility estimates from structured clinical information**, while explicitly preserving uncertainty and clinician oversight.

The research system explores both classical and quantum approaches rather than assuming that quantum computation is always preferable or always available.

The resulting architecture is therefore designed around three principles:

1. **Model integrity** — the deployed model must correspond to a locked and identifiable artifact.
2. **Reproducibility** — model files, feature schemas, manifests, and deterministic inference behaviour are explicitly checked.
3. **Human oversight** — model outputs are presented as decision-support information rather than autonomous clinical decisions.

---

# Research Objective

The central research question is:

> **Can a hybrid quantum-classical modelling pipeline provide reproducible and uncertainty-aware antimicrobial resistance estimates from longitudinal clinical data?**

The broader research workflow investigates:

* longitudinal clinical-data representation,
* feature engineering,
* classical machine learning,
* quantum machine learning,
* tensor-network representations,
* probability outputs,
* uncertainty quantification,
* model integrity,
* and fault-tolerant inference strategies.

The project deliberately treats quantum computation as an experimental component rather than making it a mandatory dependency for every inference path.

---

# System Architecture

At a high level, the project follows:

```text
                 Clinical / Structured Data
                           │
                           ▼
                  Data Processing / ETL
                           │
                           ▼
                 Feature Representation
                           │
                           ▼
                Classical / Quantum Models
                           │
                           ▼
                  Certified Model Artifact
                           │
                           ▼
                  Production Inference API
                           │
                           ▼
                    Frontend Interface
                           │
                           ▼
                 Human / Clinician Review
```

The current production-oriented implementation separates the model artifact from the user interface.

```text
┌─────────────────────────────────────────────┐
│                 Frontend                    │
│                                             │
│  React + TypeScript                         │
│  Patient / scenario interface               │
│  Results visualization                       │
└───────────────────┬─────────────────────────┘
                    │
                    │ HTTP / JSON
                    ▼
┌─────────────────────────────────────────────┐
│              FastAPI Backend                │
│                                             │
│  /health                                    │
│  /model-info                                │
│  /predict                                   │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│       Certified Inference Engine             │
│                                             │
│  Script 36 — Production Inference           │
│  Script 37 — Clinician Decision Support     │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│          Certified Model Artifact             │
│                                             │
│  FullHybridModel                            │
│  52,978 parameters                           │
│  788 input features                          │
│  4 output targets                            │
└─────────────────────────────────────────────┘
```

---

# Current Certified Inference System

The current certified production artifact is a:

```text
FullHybridModel
```

with:

| Property                            |        Certified value |
| ----------------------------------- | ---------------------: |
| Input features                      |                **788** |
| Output targets                      |                  **4** |
| Model parameters                    |             **52,978** |
| Targets                             | **NIT, SXT, CIP, LVX** |
| Model mode                          | Evaluation / inference |
| Threshold                           |                **0.5** |
| Model training during inference     |                     No |
| Official test data during inference |                     No |

The model artifact is identified by the following SHA-256 digest:

```text
db708f2c18a9390d72af1c520dab460374a737ced00e8c7ec76e3ac315bcc318
```

This digest is used as an integrity identifier for the certified model file.

---

# Quantum and Classical Modelling

The broader research pipeline explores hybrid quantum-classical modelling.

The experimental architecture includes components such as:

```text
Clinical Features
       │
       ▼
Feature Representation
       │
       ▼
Quantum / Classical Processing
       │
 ┌─────┼─────────┐
 │     │         │
 ▼     ▼         ▼
QNN   QSVM    XGBoost
 │     │         │
 └─────┼─────────┘
       ▼
Prediction / Probability
       │
       ▼
Uncertainty Analysis
```

Quantum experiments are implemented using tools including:

* PyTorch
* PennyLane
* quantum simulators
* quantum kernels
* tensor-network methods
* AWS Braket experimentation

Classical components include:

* XGBoost
* Scikit-Learn
* NumPy
* Pandas
* PyTorch

The classical path is important because quantum execution can be subject to optimisation instability, simulator/runtime limitations, backend availability, or hardware constraints.

---

# Model Certification

A major part of the project is the **certification and provenance layer**.

The certification workflow verifies that the model used by the application is the expected model artifact rather than an independently reconstructed or silently modified version.

The certification process checks:

* model SHA-256,
* parameter count,
* state-dict entry count,
* state-dict keys,
* state-dict tensor shapes,
* tensor dtypes,
* total weight element count,
* model metadata,
* Script 33 manifest,
* Script 35 certification report,
* forward-output contract,
* deterministic inference behaviour.

The certified model contains:

```text
52,978 parameters
11 state-dict entries
4 output values
```

The recorded model SHA-256 is:

```text
db708f2c18a9390d72af1c520dab460374a737ced00e8c7ec76e3ac315bcc318
```

The Script 30 SHA-256 is:

```text
12d2d201c9153253632383c598b8e360bcc5dc61b0492bf4233b406d7b6cdb3a
```

The certification workflow also verifies deterministic inference.

The recorded self-test reports:

```text
Forward contract: PASS
Deterministic inference: PASS
Exact repeat equality: True
Maximum absolute difference: 0
```

These checks are intended to establish **artifact integrity and reproducibility**, not clinical validity.

---

# Feature Contract

The production inference system requires exactly:

```text
788 features
```

The feature schema is locked in:

```text
modeling_contract/feature_names.json
```

The repository also maintains supporting contract artifacts:

```text
modeling_contract/
├── feature_names.json
├── locked_feature_manifest.csv
├── modeling_contract.json
├── target_names.json
├── target_summary.csv
├── test_data.npz
└── train_data.npz
```

The application intentionally refuses to invent or reconstruct the feature schema if the locked feature contract is unavailable.

This behaviour is deliberate: silently generating a different feature ordering could cause an apparently valid model request to use semantically incorrect inputs.

---

# Frontend and API

The current user interface is a React/TypeScript frontend.

The frontend communicates with the backend through HTTP requests.

The frontend API layer is located at:

```text
frontend/src/api/backend.ts
```

The primary operations are:

```text
GET  /health
GET  /model-info
POST /predict
```

The frontend sends prediction requests as JSON.

The certified prediction interface expects:

```json
{
  "features": [
    0.0,
    0.0,
    0.0
  ]
}
```

with exactly **788 numeric feature values**.

The frontend validates this requirement before sending the request.

---

# API Endpoints

## `GET /health`

Returns service and model health information.

Example:

```json
{
  "status": "ok",
  "application_version": "37.1-certified-clinician-cds",
  "model_version": "script33-final-certified",
  "model_sha256": "db708f2c18a9390d72af1c520dab460374a737ced00e8c7ec76e3ac315bcc318",
  "feature_count": 788,
  "targets": [
    "NIT",
    "SXT",
    "CIP",
    "LVX"
  ]
}
```

---

## `GET /model-info`

Returns metadata describing the certified inference model.

Example:

```json
{
  "application_version": "37.1-certified-clinician-cds",
  "model_version": "script33-final-certified",
  "engine_version": "script36-production",
  "model_class": "FullHybridModel",
  "parameter_count": 52978,
  "feature_count": 788,
  "targets": [
    "NIT",
    "SXT",
    "CIP",
    "LVX"
  ],
  "threshold": 0.5,
  "threshold_optimization": false,
  "calibration": false,
  "model_training": false,
  "official_test_usage": false,
  "model_sha256": "db708f2c18a9390d72af1c520dab460374a737ced00e8c7ec76e3ac315bcc318",
  "script30_sha256": "12d2d201c9153253632383c598b8e360bcc5dc61b0492bf4233b406d7b6cdb3a"
}
```

---

## `POST /predict`

Runs inference using the certified model.

Request:

```json
{
  "features": [788 numeric values]
}
```

The service validates:

1. the request structure,
2. feature count,
3. numerical finiteness,
4. the locked feature contract,
5. and the certified model artifact.

The response contains susceptibility-related prediction information for:

```text
NIT
SXT
CIP
LVX
```

The frontend normalizes the returned payload into its internal `CertifiedPrediction` representation.

---

# Authentication

The API supports clinician/API-key authentication for protected endpoints.

The local demonstration uses:

```text
X-Clinician-Api-Key
```

For example:

```bash
curl -fsS \
  -H "X-Clinician-Api-Key: test-only-key" \
  http://127.0.0.1:8080/model-info
```

An unauthenticated request to a protected endpoint should return:

```text
HTTP 401
```

The authentication mechanism used in the current MVP is intended for controlled demonstration and development environments.

It should not be treated as a complete production identity-management system.

A future production deployment should use an appropriate identity and access-management layer, secret management, auditing, and role-based access control.

---

# Local Demonstration

The application can be tested locally through the Docker image.

First define the image:

```bash
export IMAGE="asia-south1-docker.pkg.dev/amr-uti-prediction-system/uti-amr-production/uti-amr-clinician:latest"
```

Pull the image:

```bash
docker pull "$IMAGE"
```

Run the container:

```bash
docker run --rm \
  -e PORT=8080 \
  -e CLINICIAN_API_KEY="test-only-key" \
  -p 8080:8080 \
  "$IMAGE"
```

The service should then listen on:

```text
http://127.0.0.1:8080
```

---

## Verify the Frontend

Open:

```text
http://127.0.0.1:8080/
```

The frontend should return the Q-ABX application.

A command-line check is:

```bash
curl -i http://127.0.0.1:8080/
```

The returned HTML should contain the application title and frontend assets.

---

## Verify Health

```bash
curl -fsS \
  http://127.0.0.1:8080/health \
  | python3 -m json.tool
```

---

## Verify Authentication

Unauthenticated:

```bash
curl -sS \
  -o /tmp/model-info-noauth.json \
  -w "HTTP_STATUS=%{http_code}\n" \
  http://127.0.0.1:8080/model-info
```

Expected:

```text
HTTP_STATUS=401
```

Authenticated:

```bash
curl -fsS \
  -H "X-Clinician-Api-Key: test-only-key" \
  http://127.0.0.1:8080/model-info \
  | python3 -m json.tool
```

---

# Testing the Prediction Endpoint

The model requires a vector of exactly 788 numerical values.

For a structural API smoke test, a JSON payload can be generated from the shell.

```bash
python3 - <<'PY'
import json

payload = {
    "features": [0.0] * 788
}

with open("/tmp/test_prediction.json", "w") as f:
    json.dump(payload, f)

print("Created /tmp/test_prediction.json")
print("Feature count:", len(payload["features"]))
PY
```

Then send the request:

```bash
curl -fsS \
  -H "Content-Type: application/json" \
  -H "X-Clinician-Api-Key: test-only-key" \
  --data-binary @/tmp/test_prediction.json \
  http://127.0.0.1:8080/predict \
  | python3 -m json.tool
```

This verifies the complete request path:

```text
JSON
 ↓
FastAPI
 ↓
Input validation
 ↓
Certified model
 ↓
Inference
 ↓
JSON response
```

For meaningful scientific evaluation, the feature vector must correspond to the locked feature schema and a valid test case. An all-zero vector is only a **transport/integration smoke test** and should not be interpreted as a clinically meaningful patient example.

---

# Cloud Deployment

The project has been containerized for Google Cloud deployment.

The relevant Google Cloud components are:

* Google Cloud Build
* Artifact Registry
* Cloud Run
* Cloud Logging

The project uses:

```text
Project:
amr-uti-prediction-system

Region:
asia-south1

Artifact Registry repository:
uti-amr-production

Cloud Run service:
uti-amr-clinician
```

The production image is stored as:

```text
asia-south1-docker.pkg.dev/amr-uti-prediction-system/uti-amr-production/uti-amr-clinician
```

---

# Container Structure

The production deployment context contains the components required by the certified application.

Conceptually:

```text
/app
│
├── cloud_run_entrypoint.py
│
├── scripts/
│   ├── 30_quantum_contribution_ablation.py
│   ├── 36_production_inference_engine.py
│   └── 37_clinician_decision_support_app.py
│
├── modeling_contract/
│   └── feature_names.json
│
├── audit_outputs/
│   └── script33_final/
│       └── final_model.pt
│
└── frontend/
    └── dist/
        ├── index.html
        └── assets/
```

The deployment image must contain both:

```text
modeling_contract/feature_names.json
```

and:

```text
audit_outputs/script33_final/final_model.pt
```

because Script 37 uses the locked feature contract and certified model artifact during startup.

---

# Deployment Integrity

A successful Docker build alone is not sufficient.

Before considering the container usable, verify that the critical artifacts exist inside the image:

```bash
docker run --rm "$IMAGE" \
  python3 -c '
from pathlib import Path

checks = [
    "/app/modeling_contract/feature_names.json",
    "/app/audit_outputs/script33_final/final_model.pt",
    "/app/frontend/dist/index.html",
    "/app/scripts/37_clinician_decision_support_app.py",
]

for path in checks:
    print(("PASS " if Path(path).exists() else "FAIL ") + path)
'
```

Expected:

```text
PASS /app/modeling_contract/feature_names.json
PASS /app/audit_outputs/script33_final/final_model.pt
PASS /app/frontend/dist/index.html
PASS /app/scripts/37_clinician_decision_support_app.py
```

---

# Important Deployment Note

Script 37 historically resolves the project root through:

```python
Path.home() / "uti_amr_quantum_pipeline"
```

while the production container uses:

```text
/app
```

Therefore, the production entrypoint and container layout must be kept consistent with the certified application's path assumptions.

A container can successfully contain the model file at:

```text
/app/modeling_contract/feature_names.json
```

and still fail during startup if Script 37 searches for:

```text
/root/uti_amr_quantum_pipeline/modeling_contract/feature_names.json
```

This is an **application path-contract issue**, not a Docker or model-integrity issue.

When diagnosing Cloud Run startup failures, inspect the revision logs first.

---

# Project Structure

The current repository contains both research and production-oriented components.

A simplified structure is:

```text
uti_amr_quantum_pipeline/
│
├── modeling_contract/
│   ├── feature_names.json
│   ├── locked_feature_manifest.csv
│   ├── modeling_contract.json
│   ├── target_names.json
│   ├── target_summary.csv
│   ├── train_data.npz
│   └── test_data.npz
│
├── scripts/
│   ├── 29_quantum_diagnostic.py
│   ├── 30_quantum_contribution_ablation.py
│   ├── 36_production_inference_engine.py
│   └── 37_clinician_decision_support_app.py
│
├── audit_outputs/
│   ├── script33_final/
│   │   └── final_model.pt
│   │
│   ├── script35_certification/
│   │
│   └── script36_production/
│       └── script36_self_test_report.json
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── dist/
│
├── cloud_run_deployment/
│   ├── Dockerfile
│   ├── cloud_run_entrypoint.py
│   └── requirements.txt
│
└── README.md
```

---

# Reproducibility and Integrity

The project treats reproducibility as a first-class engineering requirement.

The certification process records:

```text
Model SHA-256
Script SHA-256
Parameter count
State-dict structure
Tensor shapes
Tensor dtypes
Feature count
Target definitions
Forward-output shape
Deterministic inference behaviour
```

The current deterministic inference self-test reports:

```text
Output shape: (4, 4)
Exact repeat equality: True
Maximum absolute difference: 0
```

This means that, under the tested execution conditions, repeated inference produced identical tensor outputs.

This should be understood as a **software/model reproducibility check**, not as evidence of clinical reliability.

---

# Evaluation

The project contains separate research evaluation and production certification concepts.

These should not be conflated.

## Model Evaluation

Research evaluation may include:

* accuracy,
* precision,
* recall,
* F1 score,
* ROC-AUC,
* PR-AUC,
* confusion matrices,
* probability calibration,
* uncertainty analysis.

## Production Certification

Production certification focuses on:

* correct model artifact,
* correct feature contract,
* correct model metadata,
* deterministic inference,
* expected output shape,
* and integrity hashes.

The production inference endpoint does **not** retrain the model.

The certified model is loaded for inference only.

---

# What the Current System Does Not Claim

The project does **not** claim that:

* quantum computation has been proven superior to classical computation for this task;
* the model is clinically validated;
* the model is suitable for autonomous diagnosis;
* the model can independently prescribe antibiotics;
* the reported development metrics guarantee performance in another hospital or population;
* a susceptibility probability is equivalent to a laboratory susceptibility test;
* the current MVP provides a complete hospital-grade authentication system.

These distinctions are intentional.

---

# Limitations

## Dataset Generalization

Model performance can vary across:

* hospitals,
* patient populations,
* geographic regions,
* clinical workflows,
* temporal periods,
* laboratory practices,
* and data collection systems.

Independent external validation is required before any clinical interpretation.

## Data Quality

Machine-learning performance depends on:

* missing data,
* feature quality,
* label quality,
* cohort definition,
* temporal coverage,
* and preprocessing assumptions.

## Quantum Limitations

Quantum models can encounter:

* optimization instability,
* barren plateaus,
* simulator cost,
* backend availability,
* hardware noise,
* execution latency,
* and device-specific behaviour.

## Probability Interpretation

A model output labelled as a probability should not automatically be interpreted as a calibrated clinical probability unless calibration has been independently established for the relevant population.

## Clinical Validation

The project is not clinically validated.

Any future clinical application would require appropriate:

* external validation,
* prospective evaluation,
* clinical governance,
* privacy controls,
* security controls,
* regulatory assessment,
* monitoring,
* and human oversight.

---

# Safety and Intended Use

> **IMPORTANT: AMR-UTI is a research and decision-support prototype. It is not medical advice.**

The system is not intended to:

* diagnose patients autonomously;
* prescribe medication;
* determine antibiotic dosage;
* replace microbiological culture;
* replace antimicrobial susceptibility testing;
* replace physicians;
* replace infectious-disease specialists;
* or make independent clinical decisions.

The susceptibility outputs are computational estimates intended for research and demonstration.

A qualified clinician and the appropriate laboratory evidence remain responsible for real-world clinical decisions.

---

# Current MVP Scope

The current MVP focuses on demonstrating an end-to-end technical workflow:

```text
Certified Model
      │
      ▼
FastAPI Inference API
      │
      ▼
Authentication
      │
      ▼
JSON Prediction
      │
      ▼
React Frontend
      │
      ▼
Human-Readable Results
```

The MVP intentionally does not require a full clinical database.

Authentication is currently handled at the API layer for controlled demonstration.

A future version can add:

```text
Identity Provider
       │
       ▼
User Accounts
       │
       ▼
Role-Based Access
       │
       ▼
Database
       │
       ▼
Audit Logging
```

without changing the underlying certified model contract.

---

# Future Work

Planned research and engineering directions include:

1. Independent external validation.
2. Prospective temporal evaluation.
3. More rigorous probability calibration.
4. Expanded uncertainty analysis.
5. Model-drift monitoring.
6. Improved feature-level explainability.
7. Additional quantum backends.
8. Real quantum-hardware experiments.
9. Structured identity and access management.
10. Persistent experiment and audit storage.
11. Clinical workflow integration under appropriate governance.
12. More extensive automated deployment and regression testing.

---

# Research Position

The project is intentionally framed as an investigation into **hybrid AI systems**, rather than as a claim that quantum machine learning is inherently superior to classical machine learning.

The engineering question is broader:

```text
Can emerging quantum models
        +
reliable classical fallbacks
        +
model certification
        +
uncertainty analysis
        +
human oversight
        ↓
produce a reproducible research system
for antimicrobial-resistance modelling?
```

This framing keeps the research contribution focused on system design, experimentation, reliability, and reproducibility.

---

# Contributors

Core project contributors:

* **Kanak Dharamthok**
* **Khushbu Gupta**
* **Nandini Jaiswal**
* **Sanvi Goja**

---

# Technology

The project uses or experiments with:

* Python
* PyTorch
* PennyLane
* XGBoost
* Scikit-Learn
* NumPy
* Pandas
* AWS Braket
* Google Cloud
* Docker
* Google Artifact Registry
* Google Cloud Build
* Google Cloud Run
* FastAPI
* React
* TypeScript
* Vite

---

# License

The repository should include the applicable project license in a separate `LICENSE` file.

If no license has yet been selected, do not describe the project as open source until the licensing terms have been explicitly defined.

---

# Project Status

**Current status: Research / MVP demonstration**

The project currently includes:

* a certified model artifact;
* a locked 788-feature modelling contract;
* deterministic production inference;
* FastAPI inference endpoints;
* API-key protection for protected endpoints;
* a React/TypeScript frontend;
* Docker packaging;
* Google Cloud deployment artifacts;
* model and deployment integrity checks.

The system remains a **research prototype and should not be used for clinical decision-making**.

---

## Verification Summary

The current certification pipeline records:

```text
Model class:              FullHybridModel
Parameter count:          52,978
State-dict entries:       11
Input features:           788
Output targets:           4

Forward contract:         PASS
State-dict checks:        PASS
Model loading:            PASS
Model evaluation mode:   PASS
Deterministic inference: PASS

Model SHA-256:
db708f2c18a9390d72af1c520dab460374a737ced00e8c7ec76e3ac315bcc318

Script 30 SHA-256:
12d2d201c9153253632383c598b8e360bcc5dc61b0492bf4233b406d7b6cdb3a
```

These checks establish the current software/model integrity state documented by the project's certification workflow. They do not establish clinical efficacy or clinical validity.

The key change I made is **not just wording**: I removed or softened claims that your current certified deployment does not directly substantiate. For example, the README now distinguishes the **research quantum pipeline** from the **currently certified `FullHybridModel` inference service**, and it does not present the older Streamlit architecture as though it were your current React/FastAPI MVP.

Your certification record supports the model-integrity figures above, including the 52,978 parameters, 11 state-dict entries, deterministic inference result, and model SHA-256.  

For GitHub, I would keep this as the **root README** and put lengthy deployment/debugging material into `docs/` later; GitHub specifically recommends keeping the README focused on what the project does and how to get started, with longer documentation separated out. ([GitHub Docs][1])

[1]: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes?utm_source=chatgpt.com "About the repository README file - GitHub Docs"
