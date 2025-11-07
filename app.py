import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import torch
import sklearn.preprocessing # <-- Import sklearn

# --- Import functions from your CGAN script ---
try:
    from tabular_cgan import Generator, postprocess_generated, sample_noise, label_to_onehot
    print("Imported tabular_cgan.py functions successfully.")
except ImportError:
    print("FATAL ERROR: tabular_cgan.py not found.")
    print("Please copy tabular_cgan.py into the project folder.")
    Generator, postprocess_generated, sample_noise, label_to_onehot = None, None, None, None

# Initialize the Flask app
app = Flask(__name__)

# --- Load Models, Encoder, and Scaler for PREDICTION ---
try:
    model = joblib.load("rf_model.pkl")
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Prediction model, Encoder, and Scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"Prediction Model Error: {e}. File not found.")
    print("Please run `create_models.py` first to generate the .pkl files.")
    model, encoder, scaler = None, None, None
except Exception as e:
    print(f"Error loading prediction files: {e}")
    model, encoder, scaler = None, None, None

# --- Load Models for GENERATION ---
# These variables must be defined *outside* the try block
gan_generator, gan_encoder, gan_scaler = None, None, None
try:
    # Load the trained GAN Generator
    gan_checkpoint_path = "checkpoints/cgan_checkpoint_epoch400.pt"
    
    # --- THIS IS THE FIX ---
    # We are telling torch.load() that we trust this file and it's safe
    # to load the sklearn.preprocessing.OneHotEncoder object inside it.
    # We set weights_only=False, which was the default in older PyTorch.
    gan_checkpoint = torch.load(gan_checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    
    # These dimensions must match your trained model
    # From tabular_cgan.py: num_features=3, cat_dim=25 (20 genes + 5 mutations)
    gan_generator = Generator(latent_dim=64, condition_dim=3, num_features=3, cat_dim=25)
    gan_generator.load_state_dict(gan_checkpoint['G_state_dict'])
    gan_generator.eval() # Set model to evaluation mode
    
    # Also load the encoder/scaler from the GAN checkpoint
    gan_encoder = gan_checkpoint['enc']
    gan_scaler = gan_checkpoint['scaler']
    print("CGAN Generator loaded successfully.")

except FileNotFoundError:
    print(f"--- WARNING: GAN Checkpoint '{gan_checkpoint_path}' not found. ---")
    print("The 'Generate New Patient' feature will be disabled.")
    print("To fix this, run: python tabular_cgan.py --data cmt_synthetic_dataset.csv --out generated.csv")
    gan_generator, gan_encoder, gan_scaler = None, None, None 
except Exception as e:
    print(f"--- WARNING: Error loading GAN: {e} ---")
    print("The 'Generate New Patient' feature will be disabled.")
    gan_generator, gan_encoder, gan_scaler = None, None, None 


# --- Preprocessing Function for New Data (for Prediction) ---
def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])
    cat_cols = ["gene_variant_id", "mutation_type"]
    num_cols = ["age_of_onset", "motor_score", "sensory_score"]
    X_cat = encoder.transform(df[cat_cols])
    X_num = scaler.transform(df[num_cols].values.astype(float))
    X_processed = np.hstack([X_num, X_cat])
    return X_processed


# --- Frontend Route ---
@app.route('/')
def home():
    return render_template('index.html')


# --- API Prediction Route ---
@app.route('/api/predict', methods=['POST'])
def predict():
    if not all([model, encoder, scaler]):
        return jsonify({"error": "Prediction model not loaded. Check server logs."}), 500

    try:
        data = request.get_json()
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)
        
        output_map = {0: 'Mild', 1: 'Moderate', 2: 'Severe'}
        predicted_class = output_map.get(prediction[0], 'Unknown')
        
        return jsonify({
            "prediction_text": predicted_class,
            "raw_prediction": int(prediction[0]),
            "confidence_scores": {
                "mild": round(prediction_proba[0][0], 3),
                "moderate": round(prediction_proba[0][1], 3),
                "severe": round(prediction_proba[0][2], 3)
            }
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400


# --- API Generation Route ---
@app.route('/api/generate', methods=['POST'])
def generate():
    # This check will now work correctly
    if not all([gan_generator, gan_encoder, gan_scaler, postprocess_generated, sample_noise, label_to_onehot]):
        return jsonify({"error": "GAN generator not loaded. Check server logs for warnings."}), 500

    try:
        # 1. Get desired severity from frontend
        data = request.get_json()
        severity_map = {"mild": 0, "moderate": 1, "severe": 2}
        class_idx = severity_map.get(data.get("severity", "moderate"), 1)

        # 2. Prepare inputs for the generator
        device = torch.device('cpu')
        labels_fake = torch.tensor([class_idx], dtype=torch.long, device=device) # Generate 1 sample
        cond_fake = label_to_onehot(labels_fake, num_classes=3, device=device)
        z = sample_noise(1, 64, device) # 1 sample, 64 latent_dim

        # 3. Generate data
        with torch.no_grad():
            g_out = gan_generator(z, cond_fake).cpu().numpy()
        
        num_features = 3 # age, motor, sensory
        num_pred = g_out[:, :num_features]
        cat_logits = g_out[:, num_features:]

        # 4. Post-process the data
        df_gen = postprocess_generated(num_pred, cat_logits, gan_scaler, gan_encoder)
        
        patient_profile = df_gen.to_dict('records')[0]
        patient_profile['disease_severity'] = data.get("severity", "moderate")

        return jsonify(patient_profile)

    except Exception as e:
        print(f"Error during generation: {e}")
        return jsonify({"error": str(e)}), 400


# Run the app
if __name__ == '__main__':
    if not all([Generator, postprocess_generated, sample_noise, label_to_onehot]):
        print("\nCould not start Flask server. `tabular_cgan.py` functions not imported.")
    else:
        app.run(debug=True)