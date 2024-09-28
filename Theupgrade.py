import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Load datasets
dataset = pd.read_csv('Training.csv')

# Split into features and labels
X = dataset.drop('diagnosis', axis=1)
y = dataset['diagnosis']

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

# Define models
svc = SVC(kernel='linear', probability=True, random_state=42)
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Hyperparameter tuning using GridSearchCV for SVC
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_svc = GridSearchCV(SVC(probability=True), param_grid, cv=5)
grid_svc.fit(X_train, y_train)

# Use best parameters for SVC
svc_best = grid_svc.best_estimator_

# Ensemble Learning - Voting Classifier
ensemble_model = VotingClassifier(estimators=[('svc', svc_best), ('rfc', rfc), ('gbc', gbc)], voting='soft')
ensemble_model.fit(X_train, y_train)

# Save model
pickle.dump(ensemble_model, open('models/ensemble_model.pkl', 'wb'))

# Model accuracy and confusion matrix
def evaluate_model(model):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    print(f'Model accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{cm}')

evaluate_model(ensemble_model)

# Single prediction with confidence
def predict_with_confidence(input_vector):
    predicted_prob = ensemble_model.predict_proba([input_vector])[0]
    max_prob = max(predicted_prob)
    predicted_class = ensemble_model.predict([input_vector])[0]
    return le.inverse_transform([predicted_class])[0], max_prob

# Example prediction
symptom_vector = X_test.iloc[0].values
predicted_disease, confidence = predict_with_confidence(symptom_vector)
print(f"Predicted Disease: {predicted_disease} with Confidence: {confidence*100:.2f}%")

# NLP Symptom Input Handling
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Define symptoms dictionary with more user-friendly terms
symptoms_dict = {
    'itching': 0, 'skin rash': 1, 'nodal skin eruptions': 2, 'continuous sneezing': 3,
    # Add more simplified symptom names here
}

# Symptom preprocessing function
def preprocess_symptoms(symptom_input):
    # Remove special characters, convert to lowercase
    symptom_input_clean = re.sub(r'[^a-zA-Z\s]', '', symptom_input.lower())
    # Tokenize symptoms and map to dataset format
    symptom_tokens = [symptom.strip() for symptom in symptom_input_clean.split()]
    mapped_symptoms = [symptoms_dict.get(symptom, None) for symptom in symptom_tokens if symptom in symptoms_dict]
    return mapped_symptoms

# Test NLP handling
input_symptoms = "skin rash, itching, sneezing"
user_symptoms = preprocess_symptoms(input_symptoms)
input_vector = np.zeros(len(symptoms_dict))

# Update input_vector based on user's symptoms
for symptom in user_symptoms:
    if symptom is not None:
        input_vector[symptom] = 1

predicted_disease, confidence = predict_with_confidence(input_vector)
print(f"Predicted Disease: {predicted_disease} with Confidence: {confidence*100:.2f}%")
