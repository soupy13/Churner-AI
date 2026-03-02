# 🎬 Churner AI - Customer Churn Predictor

Churner AI is a full-stack machine learning web application designed to predict the likelihood of a customer canceling their streaming subscription (churn). Trained on Netflix customer data, this tool provides actionable insights through a sleek, dark-themed user interface.

## ✨ Features
* **Predictive Analytics:** Uses a deep learning model (built with Keras/TensorFlow) to calculate churn probability in real-time.
* **Premium UI/UX:** A responsive, Netflix-inspired dark theme frontend built with Tailwind CSS and Phosphor Icons.
* **Robust Backend:** Powered by Flask to handle asynchronous API requests and perform on-the-fly data preprocessing (scaling and one-hot encoding).
* **Containerized:** Fully equipped with a Dockerfile and Gunicorn for easy, production-ready deployment.

## 📁 Project Structure

```text

Churner-AI/
├── app.py                      # Flask backend API & serving logic
├── churn_model_assets.pkl      # Pickled Keras model config, weights, and scaler
├── netflix_customer_churn.csv  # Original dataset for reference/training
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Containerization instructions
├── README.md                   # Project documentation
└── templates/
    └── index.html              # Frontend user interface
