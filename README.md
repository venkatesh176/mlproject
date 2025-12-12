 Student Performance Prediction
📌 Overview
This project predicts student performance scores based on multiple input features (e.g., study hours, attendance, socio-economic factors, etc.).
It demonstrates a complete ML lifecycle including:
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Logging and error handling
- Deployment via Flask API
- Containerization with Docker
- Version control with Git
- Multi-model training and comparison

🛠️ Tech Stack
- Languages: Python
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Flask
- Tools: Git, Docker
- Logging: Python logging module
- Error Handling: Custom exception classes

🔍 Features
- EDA:
- Data cleaning, visualization, and correlation analysis.
- Feature engineering for improved model accuracy.
- Model Training:
- Multiple ML models trained (Linear Regression, Random Forest, XGBoost, etc.).
- Hyperparameter tuning for optimal performance.
- Evaluation:
- Metrics: RMSE, MAE, R² score.
- Comparison across models to select the best.
- Logging & Error Handling:
- Centralized logging for debugging and monitoring.
- Graceful error handling with custom exceptions.
- Deployment:
- Flask app exposes REST API endpoints for predictions.
- Dockerized for portability and scalability.

🚀 Project Workflow
- Data Preparation & EDA
- Load dataset, clean missing values, visualize distributions.
- Identify key features influencing student scores.
- Model Training
- Train multiple models.
- Save trained models using joblib/pickle.
- Evaluation
- Compare models using performance metrics.
- Select best-performing model for deployment.
- Flask API
- Endpoint /predict accepts JSON input features.
- Returns predicted student score.
- Dockerization
- Dockerfile provided for containerized deployment.
- Run app in isolated environment.

📂 Project Structure
student-performance/
│── data/                # Raw & processed datasets
│── notebooks/           # EDA and experimentation
│── src/                 
│   ├── train.py         # Model training script
│   ├── evaluate.py      # Model evaluation
│   ├── app.py           # Flask application
│   ├── logger.py        # Logging setup
│   ├── exceptions.py    # Custom error handling
│── models/              # Saved trained models
│── requirements.txt     # Python dependencies
│── Dockerfile           # Docker setup
│── README.md            # Project documentation



⚙️ Setup & Usage
1. Clone the Repository
git clone https://github.com/your-username/student-performance.git
cd student-performance


2. Install Dependencies
pip install -r requirements.txt


3. Run Training
python src/train.py


4. Start Flask App
python src/app.py


Access API at: http://127.0.0.1:5000/predict
5. Docker Deployment
docker build -t student-performance .
docker run -p 5000:5000 student-performance



📊 Example API Request
POST /predict
{
  "study_hours": 5,
  "attendance": 90,
  "parental_support": "medium",
  "past_scores": 75
}


Response:
{
  "predicted_score": 82.5
}



🏆 Results
- Best-performing model: Linear Regression (example)
- Achieved R² = 0.89, outperforming baseline regression.
- Flask API successfully deployed with Docker for scalable predictions.

📌 Future Enhancements
- Add CI/CD pipeline for automated deployment.
- Integrate frontend dashboard for visualization.
- Extend to multi-class classification (e.g., grade prediction).









NOTES:

End to End MAchine Learning Project
Docker Build checked
Github Workflow
IAM User In AWS
Docker Setup In EC2 commands to be Executed
#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

Configure EC2 as self-hosted runner:
Setup github secrets:
AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = demo>> 566373416292.dkr.ecr.ap-south-1.amazonaws.com

ECR_REPOSITORY_NAME = simple-app
