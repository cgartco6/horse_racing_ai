import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from .config import MODEL_PARAMS

class SelfHealingRacingModel:
    def __init__(self):
        self.model = GradientBoostingClassifier(**MODEL_PARAMS)
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.features = [
            'horse_age', 'jockey_win_rate', 'trainer_win_rate', 'weight', 
            'distance_suitability', 'track_record', 'going_preference',
            'days_since_last_run', 'recent_form', 'injury_risk', 
            'news_sentiment', 'course_specialty', 'class_drop', 
            'speed_rating', 'stamina_index'
        ]
        self.target = 'outcome'
        self.data = pd.DataFrame(columns=self.features + [self.target])
        self.accuracy_history = []
        self.error_analysis = []
        
    def preprocess(self, df):
        X = df[self.features]
        y = df[self.target]
        return X, y
        
    def train(self):
        if len(self.data) < 50:
            logging.warning("Insufficient data for training")
            return False
            
        X, y = self.preprocess(self.data)
        X_encoded = self.encoder.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        self.accuracy_history.append(acc)
        
        # Error analysis
        errors = X_test[y_pred != y_test]
        if len(errors) > 0:
            self.error_analysis.append({
                'timestamp': datetime.now(),
                'error_count': len(errors),
                'feature_distribution': errors.mean().to_dict()
            })
        
        logging.info(f"Model retrained | Accuracy: {acc:.2f} | Samples: {len(self.data)}")
        return True
        
    def predict(self, race_data):
        X = pd.DataFrame(race_data)[self.features]
        X_encoded = self.encoder.transform(X)
        return self.model.predict_proba(X_encoded)[:, 1]
        
    def add_results(self, results):
        self.data = pd.concat([self.data, results], ignore_index=True)
        if len(self.data) % 20 == 0:  # Retrain periodically
            self.train()
        return True
            
    def self_heal(self):
        """Analyze errors and adjust model"""
        if not self.error_analysis:
            return False
            
        # Feature importance analysis
        feature_importances = self.model.feature_importances_
        important_features = [
            self.features[i] 
            for i in np.argsort(feature_importances)[-3:]
        ]
        
        # Adjust for common error patterns
        for error in self.error_analysis[-5:]:
            if 'distance_suitability' in important_features:
                self.data['distance_suitability'] = self.data['distance_suitability'] * 1.1
            if 'recent_form' in important_features:
                self.data['recent_form'] = self.data['recent_form'] * 1.05
                
        logging.info("Self-healing applied based on error analysis")
        self.error_analysis = []  # Reset analysis
        return True
