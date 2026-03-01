# 🧠 Serona ML - Face Shape Classification API

> Machine learning service for **Serona**, an AI-powered personal makeup assistant that provides face shape classification and skin tone analysis.

[![CI Pipeline](https://github.com/aryakasyahrezki/serona-ml/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/aryakasyahrezki/serona-ml/actions/workflows/ml_pipeline.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128.0-green)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)

**Built with:** Python 3.11 • FastAPI • scikit-learn • MediaPipe • Docker

---

## 📖 What is Serona?

**Serona** is an Android app that helps users discover makeup styles suited to their unique features through **live real-time face scanning**.

**Key Features:**
1. **Live face shape detection** — Heart, Oblong, Oval, Round, Square
2. **Skin tone analysis** — Fair Light, Medium Tan, Deep
3. **Personalized recommendations** — Makeup tutorials tailored to face shape and skin tone

🔗 [Download on Google Play Store](https://play.google.com/store/apps/details?id=com.serona.app&pcampaignid=web_share)

---

## 📱 App Screenshots

| Live Scanning | Result |
|:---:|:---:|
| ![Camera Screen](assets/ss_camera.png) | ![Result Screen](assets/ss_result.png) |
| Camera continuously scans and updates | Face shape + skin tone + recommendations |

> **Live Scanning:** Unlike a one-time capture, Serona continuously processes each camera frame in real-time. The result updates automatically — users simply look at the camera and the prediction stabilizes within seconds, creating a smooth, real-time experience.

---

## ⚡ Quick Start (This Service Only)

### Prerequisites
- Docker installed
- Port 8000 available

### Run with Docker
```bash
# Clone repository
git clone https://github.com/aryakasyahrezki/serona-ml.git
cd serona-ml

# Build Docker image
docker build -t serona-ml .

# Run container
docker run -d -p 8000:8000 --name serona-ml-api serona-ml

# Verify it's running
curl http://localhost:8000/
# Expected: {"status":"online","service":"Serona AI",...}
```

### Stop Service
```bash
docker stop serona-ml-api
docker rm serona-ml-api
```

---

## 🔗 Running Complete Serona System

> **Note:** To run the full Serona app (ML + Backend + Android), see the complete setup guide below. If you only need this ML service, the Quick Start above is sufficient.

### System Architecture
```
Android App (serona-android)
    ├─→ ML API (serona-ml) :8000
    │   └─→ Returns face shape + skin tone
    │
    └─→ Backend API (serona-backend) :8080
        └─→ Stores user data + provides articles
```

**Note:** ML API and Backend API are independent services. The Android app communicates with both separately.

---

### Prerequisites
- Docker & Docker Compose
- Android Studio (for mobile app)
- Git

---

### Step 1: Start ML Service
```bash
# Clone ML repository
git clone https://github.com/aryakasyahrezki/serona-ml.git
cd serona-ml

# Build and run
docker build -t serona-ml .
docker run -d -p 8000:8000 --name serona-ml-api serona-ml

# Verify
curl http://localhost:8000/
# Expected: {"status":"online",...}
```

---

### Step 2: Start Backend Service
```bash
# Navigate to parent directory
cd ..

# Clone backend repository
git clone https://github.com/aryakasyahrezki/serona-backend.git
cd serona-backend

# Copy environment file
copy .env.example .env

# Build and run (Laravel + MySQL)
docker compose up -d --build

# Wait ~30 seconds for migrations to complete

# Verify
curl http://localhost:8080/
# Expected: Laravel response
```

---

### Step 3: Run Android App
```bash
# Navigate to parent directory
cd ..

# Clone Android repository
git clone https://github.com/aryakasyahrezki/serona-android.git
cd serona-android

# Open in Android Studio
# See serona-android README for detailed setup
```

---

### Step 4: Setup Port Forwarding

**Android devices cannot access `localhost` directly. Run the connection script:**

**Windows:**
```bash
connect_to_docker.bat
```

**Mac/Linux:**
```bash
./connect_to_docker.sh
```

This forwards device ports to your computer:
- Port 8080 (Backend) → `http://127.0.0.1:8080`
- Port 8000 (ML API) → `http://127.0.0.1:8000`

---

### Verification Checklist

After starting all services:

- [ ] ML API running: http://localhost:8000
- [ ] Backend running: http://localhost:8080
- [ ] Port forwarding active: Run `adb reverse --list`
- [ ] Android app opens without crashes
- [ ] Face scan works and returns predictions

### Service Ports

| Service | Port | Desktop URL | Android Emulator URL |
|---------|------|-------------|---------------------|
| ML API | 8000 | http://localhost:8000 | http://127.0.0.1:8000 |
| Backend | 8080 | http://localhost:8080 | http://127.0.0.1:8080 |
| Database | 3306 | localhost:3306 | - |

---

## 📡 API Reference

### Health Check
```bash
GET /

Response:
{
  "status": "online",
  "service": "Serona AI",
  "location": "Cloud"
}
```

### Face Prediction
```bash
POST /predict
Content-Type: multipart/form-data

Parameters:
- file: image file (jpg, png)

Success Response:
{
  "status": "success",
  "shape": "Oval (87%)",
  "skintone": "Medium Tan",
  "server_inference_ms": "1.52",
  "total_request_ms": "8.34"
}

Failure Response:
{
  "status": "failed",
  "message": "No face detected"
}
```

---

## 🤖 Model Details

| Property | Value |
|----------|-------|
| **Algorithm** | Logistic Regression |
| **Feature Engineering** | Polynomial Features (degree=2) |
| **Feature Selection** | RFECV |
| **Input Features** | 8 geometric ratios |
| **Selected Features** | 10 (from 44 polynomial) |
| **Classes** | Heart, Oblong, Oval, Round, Square |
| **CV F1-Macro** | 74.21% ± 7.53% |
| **Random Seed** | 4 |

### Input Features

| Feature | Description |
|---------|-------------|
| `ratio_len_width` | Face length / width |
| `ratio_jaw_cheek` | Jaw width / cheek width |
| `ratio_forehead_jaw` | Forehead width / jaw width |
| `avg_jaw_angle` | Average jaw angle (degrees) |
| `ratio_chin_jaw` | Chin width / jaw width |
| `circularity` | Face oval circularity |
| `solidity` | Convex hull solidity |
| `extent` | Bounding box coverage |

---

## 🔗 Related Repositories

| Repository | Description | Link |
|------------|-------------|------|
| **serona-ml** | ML API for face analysis ← *You are here* | [serona-ml](https://github.com/aryakasyahrezki/serona-ml) |
| **serona-backend** | Backend API & database | [serona-backend](https://github.com/aryakasyahrezki/serona-backend) |
| **serona-android** | Android mobile app | [serona-android](https://github.com/aryakasyahrezki/serona-android) |

---

## 👥 Team

**Group 5 — DINAS**

| Name | Student ID |
|------|-----------|
| Aryaka Syahrezki | 2802540244 |
| Dea Audreyla Hadi | 2802540074 |
| I Gusti Ngurah Radithya Bagus Santosa | 2802538675 |
| Iyurichie Lay | 2802539980 |
| Shinta Aulia | 2802538731 |

---

## 📄 License

This project is for academic purposes — Bina Nusantara University.
