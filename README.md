Anomaly Detection in Transactions

This project detects fraudulent or anomalous transactions using the **Isolation Forest algorithm** from Scikit-learn.  
It uses a dataset of transactions, applies preprocessing, and identifies outliers based on model predictions.  
The project is deployed as an interactive web application using **Streamlit**.

Features
- Data Upload: Upload CSV transaction datasets for analysis.
- Data Preprocessing: Missing value handling, scaling with StandardScaler.
- Anomaly Detection: Uses Isolation Forest to classify transactions as normal or anomalous.
- Visualization: Displays charts for data distribution & anomalies.
- Web App: Built with Streamlit for an interactive UI.

Tech Stack
- Python: Core programming language
- Libraries:
  - Pandas, NumPy → Data manipulation
  - Matplotlib, Seaborn → Visualization
  - Scikit-learn → Machine Learning (Isolation Forest)
  - Streamlit → Web application
- Deployment: Streamlit Cloud

Project Structure
.
├── app.py # Main Streamlit app
├── requirements.txt # Project dependencies
├── dataset.csv # Sample dataset
└── README.md # Project documentation

Installation & Setup

1. Clone the repository
bash
git clone https://github.com/YOUR-USERNAME/anomaly-detection.git
cd anomaly-detection

2.Install dependencies

pip install -r requirements.txt


3.Run the application

streamlit run app.py
