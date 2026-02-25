import wandb
import os
import joblib
from datetime import datetime

# =============================================================
# 1. LOGIN MANUAL (HAPUS BAGIAN INI SEBELUM DI-PUSH KE GITHUB!)
# =============================================================
MY_KEY = "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"
wandb.login(key=MY_KEY)
# =============================================================

# 2. Inisialisasi Run
run = wandb.init(
    project="serona-ml",
    job_type="model-registration",
    notes="Registration from local for v1"
)

# 3. Path Model (Pastikan folder ini ada di root project kamu)
model_path = "machine_learning_final/models/model.pkl"

if os.path.exists(model_path):
    # Load model untuk ambil data akurasi & metadata dari notebook
    artifact_data = joblib.load(model_path)
    accuracy = artifact_data.get('cv_accuracy', 0)
    
    # Ambil metadata dari struktur artifact notebook kamu
    metadata_n_features = artifact_data['metadata'].get('n_features_selected', 0)

    # 4. Log Metrik ke Dashboard W&B
    wandb.log({
        "cv_accuracy": accuracy,
        "n_features_selected": metadata_n_features
    })

    # 5. Daftarkan Model sebagai Artifact (Otomatis jadi v1, v2, dst)
    model_artifact = wandb.Artifact(
        name="serona-face-model",
        type="model",
        metadata={
            "accuracy": f"{accuracy*100:.2f}%",
            "created_at": str(datetime.now()),
            **artifact_data['metadata']  # Memasukkan semua isi metadata dari notebook
        }
    )
    
    model_artifact.add_file(model_path)
    run.log_artifact(model_artifact)
    
    print(f"✅ Berhasil! Model terdaftar dengan akurasi {accuracy*100:.2f}%")
    print(f"Cek dashboard W&B kamu sekarang!")
else:
    print(f"❌ File tidak ditemukan di {model_path}!")
    print("Pastikan kamu menjalankan script ini dari folder root 'serona-ml'")

run.finish()