import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# To Load your data Set
data = pd.read_csv('C:/Users/veera/OneDrive/Desktop/WQP/water_quality_prediction.csv')  # Update your file path when want to change dataset

# Example feature columns 
features = ['ph', 'hardness', 'solids', 'chloramines', 'sulfate', 'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity']
target = 'quality'  # Update with your actual target column

# Split data
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict for new sample
sample = pd.DataFrame([{
    'ph': 6.8,
    'hardness': 140,
    'solids': 4800,
    'chloramines': 6.5,
    'sulfate': 290,
    'conductivity': 390,
    'organic_carbon': 9,
    'trihalomethanes': 75,
    'turbidity': 2.5
}])
prediction = model.predict(sample)
print("Predicted water quality:", prediction[0])