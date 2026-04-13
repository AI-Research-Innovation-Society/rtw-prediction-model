# TODO: see Issue #5
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
