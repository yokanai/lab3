import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from grader_contracts.linear_models import (
    RegressionInput, RegressionResult,
    ClassificationInput, ClassificationResult,
    ClassificationMetrics, BaselineComparisonResult
)

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def _compute_metrics(y_true, y_pred, y_prob):
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0))
    )

def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()
    X_test = np.asarray(data.test_features, dtype=float)
    
    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []
    
    lr = data.learning_rate
    epochs = data.epochs
    
    for _ in range(epochs):
        y_pred = np.dot(X_train, weights) + bias
        
        error = y_pred - y_train
        dw = (1.0 / n_samples) * np.dot(X_train.T, error)
        db = (1.0 / n_samples) * np.sum(error)
        
        weights -= lr * dw
        bias -= lr * db
        
        mse = np.mean(error ** 2)
        loss_history.append(float(mse))
        
    test_predictions = np.dot(X_test, weights) + bias
    
    return RegressionResult(
        weights=weights.tolist(),
        bias=float(bias),
        loss_history=loss_history,
        predictions=test_predictions.tolist()
    )

def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()
    X_test = np.asarray(data.test_features, dtype=float)
    
    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []
    
    lr = data.learning_rate
    epochs = data.epochs
    
    epsilon = 1e-15
    
    for _ in range(epochs):
        linear_model = np.dot(X_train, weights) + bias
        y_prob = _sigmoid(linear_model)
        
        error = y_prob - y_train
        dw = (1.0 / n_samples) * np.dot(X_train.T, error)
        db = (1.0 / n_samples) * np.sum(error)
        
        weights -= lr * dw
        bias -= lr * db
        
        bce = -np.mean(y_train * np.log(y_prob + epsilon) + (1 - y_train) * np.log(1 - y_prob + epsilon))
        loss_history.append(float(bce))
        
    test_linear = np.dot(X_test, weights) + bias
    test_probs = _sigmoid(test_linear)
    test_preds = (test_probs >= 0.5).astype(int)
    
    return ClassificationResult(
        weights=weights.tolist(),
        bias=float(bias),
        loss_history=loss_history,
        probabilities=test_probs.tolist(),
        predictions=test_preds.tolist()
    )

def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).ravel()
    X_test = np.asarray(data.test_features, dtype=float)
    y_test = np.asarray(data.test_target, dtype=float).ravel()
    
    my_res = train_logistic_classifier(data)
    my_metrics = _compute_metrics(y_test, my_res.predictions, my_res.probabilities)
    
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    dt_probs = dt.predict_proba(X_test)[:, 1] # Вероятность класса 1
    dt_metrics = _compute_metrics(y_test, dt_preds, dt_probs)
    
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    knn_probs = knn.predict_proba(X_test)[:, 1]
    knn_metrics = _compute_metrics(y_test, knn_preds, knn_probs)
    
    return BaselineComparisonResult(
        logistic_regression=my_metrics,
        decision_tree=dt_metrics,
        knn=knn_metrics
    )