import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from linear_models import train_linear_regression, compare_classical_models
from grader_contracts.linear_models import RegressionInput, ClassificationInput

print("1. Загружаем данные...")

try:
    df_students = pd.read_csv("Student_Performance.csv")
    df_students["Extracurricular Activities"] = df_students["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    
    X = df_students.drop(columns=["Performance Index"]).values
    y = df_students["Performance Index"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    reg_input = RegressionInput(
        train_features=X_train_scaled, train_target=y_train,
        test_features=X_test_scaled, test_target=y_test,
        learning_rate=0.05, epochs=800
    )
    
    print("2. Обучаем линейную регрессию...")
    reg_res = train_linear_regression(reg_input)
    final_mse = reg_res.loss_history[-1]
    print(f"   Итоговая ошибка (MSE) на тесте: {final_mse:.4f}")
except FileNotFoundError:
    print("   Файл Student_Performance.csv не найден!")
    final_mse = None

try:
    df_german = pd.read_csv("german.csv", sep=';')
        
    target_col = df_german.columns[0]
    y_g = df_german[target_col].values
    
    unique_classes = np.unique(y_g)
    if set(unique_classes).issubset({1, 2}):
        y_g = y_g - 1
        
    X_g_df = df_german.drop(columns=[target_col])
    X_g = pd.get_dummies(X_g_df, drop_first=True).values
    
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
        X_g, y_g, test_size=0.2, random_state=42, stratify=y_g
    )
    
    scaler_g = StandardScaler()
    X_train_g_scaled = scaler_g.fit_transform(X_train_g)
    X_test_g_scaled = scaler_g.transform(X_test_g)
    
    clf_input = ClassificationInput(
        train_features=X_train_g_scaled, train_target=y_train_g,
        test_features=X_test_g_scaled, test_target=y_test_g,
        learning_rate=0.1, epochs=1000
    )
    
    print("3. Обучаем логистическую регрессию и сравниваем с деревом/KNN...")
    cmp_res = compare_classical_models(clf_input)
    
    results = {
        "linear_regression_mse": final_mse,
        "metrics": {
            "logistic_regression": {
                "roc_auc": cmp_res.logistic_regression.roc_auc,
                "accuracy": cmp_res.logistic_regression.accuracy,
                "f1": cmp_res.logistic_regression.f1
            },
            "decision_tree": {
                "roc_auc": cmp_res.decision_tree.roc_auc,
                "accuracy": cmp_res.decision_tree.accuracy,
                "f1": cmp_res.decision_tree.f1
            },
            "knn": {
                "roc_auc": cmp_res.knn.roc_auc,
                "accuracy": cmp_res.knn.accuracy,
                "f1": cmp_res.knn.f1
            }
        }
    }
    
    print("\n--- ИТОГОВЫЕ МЕТРИКИ ---")
    print(f"LogReg AUC: {cmp_res.logistic_regression.roc_auc:.4f}")
    print(f"Tree AUC:   {cmp_res.decision_tree.roc_auc:.4f}")
    print(f"KNN AUC:    {cmp_res.knn.roc_auc:.4f}")
    
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print("\nРезультаты сохранены в experiment_results.json")
    
except FileNotFoundError:
    print("   Файл german.csv не найден!")
except Exception as e:
    print(f"\nОшибка при обработке german.csv: {e}")