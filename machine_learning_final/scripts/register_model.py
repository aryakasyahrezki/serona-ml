"""
register_model.py

PURPOSE: Manually register trained model to Weights & Biases for version tracking
WHEN TO RUN: After training model and saving to models/model.pkl
HOW TO RUN: python scripts/register_model.py (from project root)

REQUIREMENTS:
- W&B API key set in environment: export WANDB_API_KEY="your_key"
- Trained model.pkl exists at machine_learning_final/models/model.pkl
"""

import os
import sys
from datetime import datetime

import joblib
import wandb


def login_to_wandb():
    """
    Authenticate with Weights & Biases using API key from environment.

    Returns:
        bool: True if login successful, False otherwise
    """
    api_key = os.environ.get('WANDB_API_KEY')

    if api_key:
        wandb.login(key=api_key)
        print("✅ Logged in using WANDB_API_KEY from environment")
        return True

    print("❌ WANDB_API_KEY not found in environment")
    print("Set it with: export WANDB_API_KEY='your_key'")
    return False


def load_model_artifact(model_path):
    """
    Load trained model and extract metadata.

    Args:
        model_path (str): Path to model.pkl file

    Returns:
        dict: Model artifact data, or None if file not found
    """
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        print("Make sure you run this from project root directory")
        print("And that you've trained and saved model.pkl first")
        return None

    print(f"📦 Loading model from {model_path}...")
    artifact_data = joblib.load(model_path)
    return artifact_data


def print_model_metrics(artifact_data):
    """
    Print model metrics to console.

    Args:
        artifact_data (dict): Model artifact containing metrics
    """
    metrics = artifact_data.get('metrics', {})
    cv_f1_macro = metrics.get('cv_f1_macro', 0)
    random_seed = artifact_data.get('random_seed', 0)
    metadata = artifact_data.get('metadata', {})
    n_features = metadata.get('n_features_selected', 'N/A')

    print("\n📊 Model Metrics:")
    print(f"   CV F1-Macro: {cv_f1_macro*100:.2f}%")
    print(f"   Random Seed: {random_seed}")
    print(f"   Features Selected: {n_features}")


def register_model_to_wandb(artifact_data, model_path):
    """
    Register model as W&B artifact with versioning.

    Args:
        artifact_data (dict): Model artifact containing metrics and metadata
        model_path (str): Path to model.pkl file

    Returns:
        bool: True if registration successful
    """
    # Extract metrics
    metrics = artifact_data.get('metrics', {})
    cv_f1_macro = metrics.get('cv_f1_macro', 0)
    random_seed = artifact_data.get('random_seed', 0)
    metadata = artifact_data.get('metadata', {})

    # Initialize W&B run
    run = wandb.init(
        entity= "aryakasyahrezki-binus-university",
        project="serona-ml",
        job_type="model-registration",
        notes="Manual model registration after training"
    )

    # Log metrics to dashboard
    wandb.log({
        "cv_f1_macro": cv_f1_macro,
        "n_features_selected": metadata.get('n_features_selected', 0),
        "random_seed": random_seed
    })

    # Create artifact with metadata
    model_artifact = wandb.Artifact(
        name="serona-face-model",
        type="model",
        description="Face shape classification model",
        metadata={
            "cv_f1_macro": f"{cv_f1_macro*100:.2f}%",
            "created_at": datetime.now().isoformat(),
            "random_seed": random_seed,
            **metadata
        }
    )

    # Add model file and register
    model_artifact.add_file(model_path)
    run.log_artifact(model_artifact, aliases=["latest", "production"])

    # Print success message
    print("\n✅ SUCCESS! Model registered to W&B")
    print(f"   F1-Score: {cv_f1_macro*100:.2f}%")
    print("   View at: https://wandb.ai/your-username/serona-ml")
    print("\n💡 This model is tagged as 'latest' and 'production'")

    # Finish run
    run.finish()
    return True


def main():
    """Main function to orchestrate model registration."""
    # Step 1: Login to W&B
    if not login_to_wandb():
        sys.exit(1)

    # Step 2: Load model
    model_path = "machine_learning_final/models/model.pkl"
    artifact_data = load_model_artifact(model_path)

    if artifact_data is None:
        sys.exit(1)

    # Step 3: Print metrics
    print_model_metrics(artifact_data)

    # Step 4: Register to W&B
    success = register_model_to_wandb(artifact_data, model_path)

    if success:
        print("\n🎉 Model registration complete!")
        sys.exit(0)
    else:
        print("\n❌ Model registration failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()