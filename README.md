# 🛍️ Amazon Review Sentiment Analysis using Machine Learning & MLOps

An end-to-end Natural Language Processing (NLP) BEGINNNER level project that analyzes Amazon customer reviews and predicts whether the sentiment is **Positive** or **Negative**. The project includes data preprocessing, machine learning model training, MLflow experiment tracking, Docker containerization, and AWS EC2 cloud deployment.

---

## 🚀 Project Overview

Customer reviews contain valuable insights about product satisfaction. Manually analyzing thousands of reviews is difficult and time-consuming.

This project builds an AI-powered sentiment analysis system that automatically classifies customer reviews using Machine Learning techniques.

The trained model is deployed as a Streamlit web application where users can enter a review and receive an instant sentiment prediction.

---

# 🏗️ Project Architecture

```
                 Amazon Reviews Dataset
                          |
                          ↓
              Text Data Preprocessing
                          |
                          ↓
                  TF-IDF Vectorization
                          |
                          ↓
              Machine Learning Model
                          |
                          ↓
                    MLflow Tracking
                          |
                          ↓
                  Docker Container
                          |
                          ↓
                    AWS EC2 Deployment
                          |
                          ↓
              Streamlit Web Application
```

---

# ✨ Features

✅ Text preprocessing using NLP techniques
✅ TF-IDF based feature extraction
✅ Machine Learning sentiment classification
✅ MLflow experiment tracking
✅ Model artifact management
✅ Docker containerization
✅ Cloud deployment using AWS EC2
✅ Interactive Streamlit user interface

---

# 🛠️ Tech Stack

## Programming Language

* Python

## Machine Learning & NLP

* Scikit-learn
* NLTK
* Pandas
* NumPy

## Model Tracking

* MLflow

## Frontend

* Streamlit

## Deployment

* Docker
* AWS EC2

## Version Control

* Git & GitHub

---

# 📂 Project Structure

```
Sentiment-Analysis-MLflow
│
├── app.py                  # Streamlit application
├── train.py                # Model training pipeline
├── predict.py              # Prediction logic
├── preprocessing.py        # Text cleaning functions
├── utils.py                # Utility functions
│
├── models/
│   └── trained model files
│
├── sentiment-model/        # ML model artifacts
├── tfidf.pkl               # TF-IDF vectorizer
│
├── Dockerfile              # Docker configuration
├── requirements.txt        # Dependencies
├── README.md
```

---

# ⚙️ Machine Learning Workflow

## 1. Data Preprocessing

The raw review text is cleaned using:

* Lowercase conversion
* Removing unwanted characters
* Tokenization
* Stopword removal
* Lemmatization

---

## 2. Feature Extraction

Since machine learning algorithms cannot understand raw text, TF-IDF Vectorization is used to convert text into numerical features.

TF-IDF assigns importance scores to words based on their frequency and uniqueness in the dataset.

---

## 3. Model Training

The model learns patterns from labeled reviews and predicts sentiment for unseen reviews.

Steps:

* Load dataset
* Preprocess text
* Convert text using TF-IDF
* Train machine learning model
* Evaluate performance
* Save model artifacts

---

# 🧪 MLflow Experiment Tracking

MLflow is used for:

* Tracking model experiments
* Logging parameters
* Recording evaluation metrics
* Managing model artifacts

This helps maintain reproducibility and compare different model versions.

---

# 🐳 Docker Deployment

The application is containerized using Docker.

Docker packages:

* Application code
* Python environment
* Required dependencies
* Model files

Build Docker image:

```bash
docker build -t amazon-review-app .
```

Run container:

```bash
docker run -d -p 8501:8501 --name sentiment-app amazon-review-app
```

---

# ☁️ AWS EC2 Deployment

The application is deployed on an AWS EC2 virtual machine.

Deployment steps:

1. Launch EC2 instance
2. Connect using SSH
3. Install Docker
4. Clone GitHub repository
5. Build Docker image
6. Run Streamlit container
7. Access application through public IP

Application URL:

```
http://<EC2-PUBLIC-IP>:8501
```

---

# 🖥️ Application Demo

The Streamlit application allows users to:

1. Enter an Amazon product review
2. Submit the review
3. Receive sentiment prediction

Example:

Input:

```
"The product quality is excellent. I really loved it."
```

Output:

```
Positive Sentiment
```

---

# 📌 Future Improvements

* Add confidence score visualization
* Improve dashboard UI
* Add sentiment analytics dashboard
* Implement CI/CD pipeline using GitHub Actions
* Deploy using AWS ECS/Kubernetes
* Upgrade model using transformer-based architectures (BERT)

---

# 👩‍💻 Author

**Himani Devadiga**

---

# ⭐ Key Learning Outcomes

Through this project, I gained practical experience in:

* Natural Language Processing
* Machine Learning model development
* MLflow for experiment tracking
* Docker containerization
* Cloud deployment using AWS EC2
* Building and deploying ML applications

 Final URL: http://13.206.85.221:8501 
