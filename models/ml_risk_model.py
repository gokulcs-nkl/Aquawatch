"""
ML Risk Model — scikit-learn Random Forest + XGBoost-style Gradient Boosting.

Trains on synthetic + real feature vectors to predict bloom risk class.
Shows feature importance, accuracy, and classification report.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import json


# ── Feature names (must match the order we build vectors) ──
FEATURE_NAMES = [
    "water_temp", "air_temp", "temp_anomaly",
    "rainfall_7d", "days_since_rain", "stagnation_index",
    "avg_wind_7d", "agricultural_pct", "urban_pct",
    "uv_index", "cloud_cover", "growth_rate_mu",
]

RISK_CLASSES = ["SAFE", "LOW", "WARNING", "CRITICAL"]


def _extract_ml_features(feature_vector: dict, growth_rate: dict) -> np.ndarray:
    """Extract a flat numeric feature array from the pipeline's feature vector."""
    t = feature_vector.get("temperature", {})
    p = feature_vector.get("precipitation", {})
    s = feature_vector.get("stagnation", {})
    n = feature_vector.get("nutrients", {})
    l = feature_vector.get("light", {})

    return np.array([
        feature_vector.get("water_temp", 20.0),
        feature_vector.get("air_temp", 20.0),
        t.get("temp_anomaly_c", 0.0),
        p.get("rainfall_7d", 0.0),
        p.get("days_since_significant_rain", 0),
        p.get("stagnation_index", 0.0),
        s.get("avg_wind_7d", 10.0),
        n.get("agricultural_pct", 0.0),
        n.get("urban_pct", 0.0),
        l.get("uv_index", 3.0),
        l.get("cloud_cover", 50.0),
        growth_rate.get("mu_per_day", 0.0),
    ], dtype=np.float64)


def _generate_training_data(n_samples: int = 2000, seed: int = 42) -> tuple:
    """
    Generate scientifically-grounded synthetic training data.
    Maps realistic environmental conditions → bloom risk classes.
    """
    rng = np.random.RandomState(seed)

    X = np.zeros((n_samples, len(FEATURE_NAMES)))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        # Randomize environmental conditions
        water_temp = rng.uniform(5, 35)
        air_temp = water_temp + rng.normal(2, 3)
        anomaly = rng.normal(0, 4)
        rainfall = rng.exponential(5)
        days_dry = rng.randint(0, 30)
        stagnation = rng.uniform(0, 1)
        wind = rng.uniform(0, 40)
        agri_pct = rng.uniform(0, 80)
        urban_pct = rng.uniform(0, 50)
        uv = rng.uniform(0, 12)
        cloud = rng.uniform(0, 100)
        mu = max(0, rng.normal(0.3, 0.4))

        X[i] = [water_temp, air_temp, anomaly, rainfall, days_dry,
                 stagnation, wind, agri_pct, urban_pct, uv, cloud, mu]

        # Rule-based labeling (replicates domain knowledge)
        risk_score = 0.0
        # Temperature — primary driver
        if water_temp > 25:
            risk_score += 30
        elif water_temp > 20:
            risk_score += 15
        elif water_temp > 15:
            risk_score += 5

        # Stagnation (low wind, high stagnation)
        if wind < 5:
            risk_score += 20
        elif wind < 10:
            risk_score += 10
        if stagnation > 0.6:
            risk_score += 10

        # Nutrients proxy
        if agri_pct > 40:
            risk_score += 15
        if urban_pct > 30:
            risk_score += 5

        # Rainfall pattern
        if days_dry > 14 and water_temp > 22:
            risk_score += 10  # prolonged dry + warm
        if rainfall > 15:
            risk_score += 8   # nutrient flush

        # Growth rate
        if mu > 0.5:
            risk_score += 10
        if mu > 0.8:
            risk_score += 5

        # Interaction: warm + calm + nutrient-rich
        if water_temp > 25 and wind < 8 and agri_pct > 30:
            risk_score += 15

        # Anomaly boost
        if anomaly > 3:
            risk_score += 8

        # Add some noise
        risk_score += rng.normal(0, 5)
        risk_score = np.clip(risk_score, 0, 100)

        # Classify
        if risk_score < 25:
            y[i] = 0  # SAFE
        elif risk_score < 50:
            y[i] = 1  # LOW
        elif risk_score < 75:
            y[i] = 2  # WARNING
        else:
            y[i] = 3  # CRITICAL

    return X, y


class BloomRiskMLModel:
    """Trained ML model for bloom risk prediction."""

    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.train_accuracy = 0.0
        self.cv_scores = []
        self.feature_importances = {}
        self.classification_rep = ""
        self.conf_matrix = None

    def train(self, n_samples: int = 2000):
        """Train both Random Forest and Gradient Boosting models."""
        X, y = _generate_training_data(n_samples)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.rf_model.fit(X_train_scaled, y_train)

        # Gradient Boosting (XGBoost-style)
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.gb_model.fit(X_train_scaled, y_train)

        # Evaluate Random Forest (primary)
        y_pred = self.rf_model.predict(X_test_scaled)
        self.train_accuracy = accuracy_score(y_test, y_pred)
        self.cv_scores = cross_val_score(
            self.rf_model, self.scaler.transform(X), y, cv=5
        ).tolist()
        self.classification_rep = classification_report(
            y_test, y_pred, target_names=RISK_CLASSES, output_dict=True
        )
        self.conf_matrix = confusion_matrix(y_test, y_pred).tolist()

        # Feature importances (ensemble average)
        rf_imp = self.rf_model.feature_importances_
        gb_imp = self.gb_model.feature_importances_
        avg_imp = (rf_imp + gb_imp) / 2
        self.feature_importances = {
            name: round(float(imp), 4)
            for name, imp in sorted(
                zip(FEATURE_NAMES, avg_imp), key=lambda x: -x[1]
            )
        }

        self.is_trained = True
        return self

    def predict(self, feature_vector: dict, growth_rate: dict) -> dict:
        """
        Predict bloom risk class from pipeline feature vector.
        Returns dict with class, probabilities, and both model outputs.
        """
        if not self.is_trained:
            self.train()

        x = _extract_ml_features(feature_vector, growth_rate).reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        rf_pred = int(self.rf_model.predict(x_scaled)[0])
        rf_proba = self.rf_model.predict_proba(x_scaled)[0].tolist()

        gb_pred = int(self.gb_model.predict(x_scaled)[0])
        gb_proba = self.gb_model.predict_proba(x_scaled)[0].tolist()

        # Ensemble: average probabilities
        ensemble_proba = [(a + b) / 2 for a, b in zip(rf_proba, gb_proba)]
        ensemble_class = int(np.argmax(ensemble_proba))

        return {
            "predicted_class": RISK_CLASSES[ensemble_class],
            "predicted_index": ensemble_class,
            "ensemble_probabilities": {
                cls: round(p, 4) for cls, p in zip(RISK_CLASSES, ensemble_proba)
            },
            "rf_prediction": RISK_CLASSES[rf_pred],
            "rf_probabilities": {
                cls: round(p, 4) for cls, p in zip(RISK_CLASSES, rf_proba)
            },
            "gb_prediction": RISK_CLASSES[gb_pred],
            "gb_probabilities": {
                cls: round(p, 4) for cls, p in zip(RISK_CLASSES, gb_proba)
            },
            "feature_values": {
                name: round(float(v), 4)
                for name, v in zip(FEATURE_NAMES, x[0])
            },
        }

    def get_model_info(self) -> dict:
        """Return training metrics for dashboard display."""
        return {
            "accuracy": round(self.train_accuracy, 4),
            "cv_mean": round(float(np.mean(self.cv_scores)), 4),
            "cv_std": round(float(np.std(self.cv_scores)), 4),
            "cv_scores": [round(s, 4) for s in self.cv_scores],
            "feature_importances": self.feature_importances,
            "classification_report": self.classification_rep,
            "confusion_matrix": self.conf_matrix,
            "n_features": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
            "risk_classes": RISK_CLASSES,
        }


# ── Module-level singleton ──────────────────────────────────────────────────
_model_instance = None


def get_ml_model() -> BloomRiskMLModel:
    """Return a trained singleton ML model."""
    global _model_instance
    if _model_instance is None:
        _model_instance = BloomRiskMLModel()
        _model_instance.train()
    return _model_instance
