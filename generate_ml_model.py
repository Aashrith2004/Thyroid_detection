import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("thyroid_dataset.csv")
X = data.drop("class", axis=1)
y = data["class"]

# Encode 'sex' as 0/1 if it's not already numeric
if data['sex'].dtype == 'object':
    data['sex'] = data['sex'].map({'M': 0, 'F': 1})
    X = data.drop("class", axis=1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
with open('ml_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('ml_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModel and scaler saved as 'ml_model.pkl' and 'ml_scaler.pkl'")

