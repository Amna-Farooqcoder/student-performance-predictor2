import pandas as pd
import numpy as np

# Create student dataset
np.random.seed(42)
n = 200

df = pd.DataFrame({
    'study_hours': np.random.uniform(1, 10, n),
    'attendance': np.random.uniform(50, 100, n),
    'previous_score': np.random.uniform(40, 100, n),
    'sleep_hours': np.random.uniform(4, 10, n),
    'final_score': None
})

# Final score based on real logic
df['final_score'] = (
    df['study_hours'] * 3 +
    df['attendance'] * 0.3 +
    df['previous_score'] * 0.5 +
    df['sleep_hours'] * 1.5 +
    np.random.normal(0, 5, n)
).clip(0, 100).round(2)

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Split data
X = df.drop('final_score', axis=1)
y = df['final_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Results
predictions = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, predictions):.2f}")
print(f"Mean Error: {mean_absolute_error(y_test, predictions):.2f} marks")

# Feature importance chart
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.barh(X.columns, model.feature_importances_, color='steelblue')
plt.xlabel('Importance')
plt.title('What affects student performance most?')
plt.tight_layout()
plt.savefig('student_importance.png')
plt.show()