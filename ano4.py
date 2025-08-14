import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import plotly.io as pio
pio.renderers.default = "browser"

# Load and inspect data
try:
    data = pd.read_csv("transaction_anomalies_dataset.csv")
    print("Data loaded successfully. First 5 rows:")
    print(data.head())
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
    exit()

# Data quality checks
print("\nMissing values per column:")
print(data.isnull().sum())

print("\nData types and info:")
print(data.info())

print("\nStatistical summary:")
print(data.describe())

# Simple threshold-based anomaly detection
mean_amount = data['Transaction_Amount'].mean()
std_amount = data['Transaction_Amount'].std()
anomaly_threshold = mean_amount + 2 * std_amount

data['Threshold_Anomaly'] = data['Transaction_Amount'] > anomaly_threshold

# Isolation Forest model
relevant_features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']
X = data[relevant_features]
y = data['Threshold_Anomaly']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
contamination = float(y_train.mean())
model = IsolationForest(contamination=contamination, random_state=42)
model.fit(X_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_binary = np.where(y_pred == -1, 1, 0)
data_test = X_test.copy()
data_test['Predicted_Anomaly'] = y_pred_binary
data_test['Actual_Label'] = y_test.values

# Evaluation
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred_binary, target_names=['Normal', 'Anomaly']))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)
fig_conf = px.imshow(conf_matrix,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Normal', 'Anomaly'],
                    y=['Normal', 'Anomaly'],
                    title='Confusion Matrix',
                    text_auto=True)
fig_conf.show()

# Visualizations
# Scatter Plot: Transaction Amount vs Frequency colored by Prediction
fig_scatter_pred = px.scatter(data_test,
    x='Transaction_Amount',
    y='Frequency_of_Transactions',
    color='Predicted_Anomaly',
    title='Transaction Amount vs Frequency (Anomaly Detection)',
    color_discrete_map={0: 'blue', 1: 'red'},
    labels={'Predicted_Anomaly': 'Anomaly (1=True)'}
)
fig_scatter_pred.show()

# Heatmaps: Normal vs Anomalous
fig_normal = px.density_heatmap(data_test[data_test['Predicted_Anomaly'] == 0],
    x='Transaction_Amount',
    y='Frequency_of_Transactions',
    nbinsx=30, nbinsy=30,
    title="Normal Transactions Heatmap",
    color_continuous_scale='Blues')
fig_normal.show()

fig_anomalous = px.density_heatmap(data_test[data_test['Predicted_Anomaly'] == 1],
    x='Transaction_Amount',
    y='Frequency_of_Transactions',
    nbinsx=30, nbinsy=30,
    title="Anomalous Transactions Heatmap",
    color_continuous_scale='Reds')
fig_anomalous.show()

# Interactive anomaly checker
def predict_anomaly():
    print("\nEnter transaction details for anomaly detection:")
    try:
        amt = float(input("Transaction Amount: "))
        avg_amt = float(input("Average Transaction Amount: "))
        freq = float(input("Frequency of Transactions (per month): "))
        input_data = pd.DataFrame([[amt, avg_amt, freq]], columns=relevant_features)

        pred = model.predict(input_data)
        is_anomaly = pred[0] == -1

        if is_anomaly:
            print("\n⚠️ ALERT: This transaction is flagged as ANOMALOUS!")
            print(f"Anomaly score: {model.decision_function(input_data)[0]:.2f}")
        else:
            print("\n✅ This transaction appears NORMAL.")
    except ValueError:
        print("Error: Please enter valid numerical values.")

# Loop for prediction
while True:
    predict_anomaly()
    another = input("\nCheck another transaction? (y/n): ").lower()
    if another != 'y':
        break

print("\n✅ Anomaly detection completed.")
