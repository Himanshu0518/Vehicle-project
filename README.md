# 🚗 Vehicle Insurance prediction System – MLOps End-to-End Project

This project demonstrates the **complete MLOps pipeline** for an ML model trained on vehicle-related data. From **data ingestion using MongoDB**, **model training**, **evaluation**, and **deployment on AWS EC2** using **Docker and GitHub Actions**, the project is structured to reflect **production-grade workflows**.

---

## 📌 Table of Contents

- [🎯 Project Objective](#-project-objective)
- [🚀 Features](#-features)
- [🧰 Tech Stack](#-tech-stack)
- [📁 Folder Structure](#-folder-structure)
- [⚙️ Local Setup](#️-local-setup)
- [☁️ Cloud Integration](#-cloud-integration)
- [🔍 Model Pipeline](#-model-pipeline)
- [📈 Future Scope](#-future-scope)
- [🙋‍♂️ Author](#-author)

---

## 🎯 Project Objective

Build a modular, end-to-end **MLOps pipeline** that:
- Ingests and stores vehicle data using MongoDB Atlas
- Validates, transforms, and trains models on the dataset
- Automatically deploys models via **CI/CD pipeline** with Docker & GitHub Actions
- Hosts prediction API on **AWS EC2** exposed over port `5000`

---

## 🚀 Features

- ⛓️ Modular code with `components`, `pipeline`, `configuration`
- ☁️ MongoDB Atlas for cloud-based NoSQL storage
- ⚙️ CI/CD pipeline using GitHub Actions
- 🐳 Dockerized app deployment to AWS EC2
- 🌐 Live FastAPI app on EC2 instance
- 🧠 Model registry using AWS S3
- 📉 Schema-based validation & detailed logging

---

## 🧰 Tech Stack

| Layer             | Tools & Frameworks |
|------------------|--------------------|
| Programming Lang | Python 3.10         |
| Backend/API      | FastAPI             |
| ML Framework     | Scikit-learn        |
| Cloud Storage    | MongoDB Atlas, AWS S3 |
| Deployment       | AWS EC2, Docker, GitHub Actions |
| CI/CD            | GitHub Actions, Self-hosted Runner |
| Environment Mgmt | Conda, `requirements.txt` |
| Logging & Error  | Custom Logger & Exception Classes |

---

## 📁 Folder Structure
```bash
VEHICLE-PROJECT/
│
├── .dockerignore
├── .gitignore
├── README.md
├── demo.py
├── .env
├── requirements.txt
│
├── artifact/
├── config/
├── logs/
├── notebooks/
│   ├── experiments.py
│   └── mongoDB_demo.ipynb
│
├── src/
│   ├── cloud_storage/
│   │   ├── __init__.py
│   │   └── aws_storage.py
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── data_validation.py
│   │   ├── model_evaluation.py
│   │   ├── model_pusher.py
│   │   └── model_trainer.py
│   │
│   ├── configuration/
│   │   ├── __init__.py
│   │   ├── aws_connection.py
│   │   └── mongo_db_connection.py
│   │
│   ├── constants/
│   │   └── __init__.py
│   │
│   ├── data_access/
│   │   ├── __init__.py
│   │   └── proj1_data.py
│   │
│   ├── entity/
│   │   ├── __init__.py
│   │   ├── artifact_entity.py
│   │   ├── config_entity.py
│   │   ├── estimator.py
│   │   └── s3_estimator.py
│   │
│   ├── exception/
│   │   └── __init__.py
│   │
│   ├── logger/
│   │   └── __init__.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── prediction_pipeline.py
│   │   └── training_pipeline.py
│   │
│   └── utils/
│       └── __init__.py
│       └──  main_utils.py
│
├── static/
├── templates/
```

```bash
# Clone repo
git clone https://github.com/yourusername/vehicle-project.git
cd vehicle-project

# Create and activate environment
conda create -n vehicle python=3.10 -y
conda activate vehicle

# Install dependencies
pip install -r requirements.txt
```

# 🌐 Setup Environment Variables

## MongoDB Atlas
```bash
$env:MONGODB_URL = "mongodb+srv://<user>:<pass>@cluster.mongodb.net/"

# AWS Credentials
$env:AWS_ACCESS_KEY_ID = "<your_access_key>"
$env:AWS_SECRET_ACCESS_KEY = "<your_secret_key>"
```

## ☁️ Cloud Integration

✅ MongoDB Atlas: NoSQL database for vehicle data

✅ AWS S3: Stores trained model artifacts

✅ AWS EC2: Hosts the FastAPI app on public IP

✅ AWS ECR: Container image registry


## 🔁 CI/CD Workflow

Automates deployment on each GitHub push to main.

Docker image is built via GitHub Actions

ECR repo stores the container image

EC2 self-hosted runner pulls & runs the container

Secrets managed using GitHub Secrets

  
## 🔍 Model Pipeline
Step	Tool/Module

Data Ingestion	PyMongo + MongoDB Atlas

Data Validation	schema.yaml

Data Transformation	Sklearn Preprocessors

Model Training	RandomForest

Model Evaluation	Compare with previous model

Model Pusher	Upload to AWS S3

FastAPI Endpoints	/predict


🔗 **Live Demo**: [http://18.208.198.133:5000/](http://18.208.198.133:5000/)

⚠️ **Disclaimer**: This app is currently hosted on AWS under the 12-month free tier. I may delete the services in the future to avoid charges. If the link is inactive, the deployment has likely been removed.

## 📽 Demo Video

Watch the demo here: [Click to Watch](## 📽 Demo Video

Watch the demo here: [Click to Watch](https://youtu.be/5e-8gAVst2k)
)


## 🙋‍♂️ Author
Himanshu Singh
MLOps | FastAPI | ML Engineer
📧 Email: himanshu.iiitu2027@gmail.com


