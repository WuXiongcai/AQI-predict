import numpy as np
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import VotingRegressor, VotingClassifier

class EnsembleRegressor:
    def __init__(self, n_targets):
        self.n_targets = n_targets
        self.models = []
        for _ in range(n_targets):
            lgb = LGBMRegressor(random_state=42)
            xgb = XGBRegressor(random_state=42)
            cat = CatBoostRegressor(random_seed=42, verbose=False)
            
            ensemble = VotingRegressor([
                ('lgb', lgb),
                ('xgb', xgb),
                ('cat', cat)
            ])
            self.models.append(ensemble)
    
    def train(self, X, y):
        """训练每个目标变量的集成模型"""
        for i in range(self.n_targets):
            self.models[i].fit(X, y[:, i])
    
    def predict(self, X):
        """预测所有目标变量"""
        predictions = np.zeros((X.shape[0], self.n_targets))
        for i in range(self.n_targets):
            predictions[:, i] = self.models[i].predict(X)
        return predictions

class QualityClassifier:
    def __init__(self):
        lgb = LGBMClassifier(random_state=42)
        xgb = XGBClassifier(random_state=42)
        cat = CatBoostClassifier(random_seed=42, verbose=False)
        
        self.model = VotingClassifier([
            ('lgb', lgb),
            ('xgb', xgb),
            ('cat', cat)
        ], voting='soft')
    
    def train(self, X, y):
        """训练空气质量分类模型"""
        self.model.fit(X, y)
    
    def predict(self, X):
        """预测空气质量等级"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """预测空气质量等级概率"""
        return self.model.predict_proba(X) 