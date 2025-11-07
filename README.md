# CMT-Predictive-Generative-Modeler

This project is an end-to-end web application for modeling Charcot-Marie-Tooth (CMT) disease. It uses a deep learning model (CGAN) to generate high-fidelity synthetic patient data and a machine learning model (RandomForest) to predict disease severity based on genotype-phenotype data.

The application addresses the problem of data scarcity in rare disease research by creating a scalable, privacy-preserving framework for data augmentation and predictive modeling.

---

## Features

This application features a live web dashboard with two primary functions:

1.  **Synthetic Patient Generation:**
    * Uses a trained **Conditional GAN (CGAN)** to generate new, artificial patient profiles from scratch.
    * Allows you to **conditionally** request a patient with a specific target severity (e.g., "Mild", "Moderate", or "Severe").

2.  **Disease Severity Prediction:**
    * Uses a trained **RandomForest model** to predict a patient's disease severity.
    * Takes a patient's genetic data (`gene_variant_id`, `mutation_type`) and clinical data (`age_of_onset`, `motor_score`, `sensory_score`) as input.
    * Outputs a prediction with confidence scores visualized in bar charts.

---

## Project Structure

/CMT_Project/ ├── app.py # Flask web server (powers the app) ├── create_models.py # Trains the RandomForest model ├── tabular_cgan.py # Trains the CGAN model ├── cmt_synthetic_dataset.csv # The "real" seed dataset ├── requirements.txt # All Python dependencies | ├── rf_model.pkl # (Generated) The predictive model ├── encoder.pkl # (Generated) Data processor ├── scaler.pkl # (Generated) Data processor | ├── /checkpoints/ │ └── cgan_checkpoint_epoch400.pt # (Generated) The generative model | └── /templates/ └── index.html # The frontend web page


---

## How to Run

This project requires two training steps before the web application can be run.

### Step 1: Install Dependencies

Install all the required Python libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
Step 2: Train the AI Models
You must run both training scripts to generate the models that the app.py server needs.

A. Train the Predictive Model: This script reads cmt_synthetic_dataset.csv and creates rf_model.pkl, encoder.pkl, and scaler.pkl.

Bash

python create_models.py
B. Train the Generative Model: This script reads cmt_synthetic_dataset.csv, trains the CGAN, and creates the checkpoints/cgan_checkpoint_epoch400.pt file.

Bash

python tabular_cgan.py --data cmt_synthetic_dataset.csv --out cmt_synthetic_generated.csv
Step 3: Run the Web Application
Once both models are trained and their files are present, you can start the Flask web server.

Bash

python -m flask run
Open your web browser and go to http://127.0.0.1:5000 to see the application live.
