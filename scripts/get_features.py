import os
import pickle

# Construct the absolute path to the model file relative to the script location
script_dir = os.path.dirname(__file__)
model_path = os.path.join(
    script_dir, "models", "readmission_model.pkl"
)  # Path relative to script

try:
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
    else:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        if "feature_names" in model_data:
            feature_names = model_data["feature_names"]
            print("--- FEATURE NAMES START ---")
            # Print each feature name on a new line for clarity
            for name in feature_names:
                print(name)
            print("--- FEATURE NAMES END ---")
            print(f"\nTotal features: {len(feature_names)}")
        else:
            print("Error: 'feature_names' key not found in the model file.")

except Exception as e:
    print(f"An error occurred: {e}")
