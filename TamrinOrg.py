import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Set up the parameter grid for Grid Search including test_size and random_state
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'metric': ['minkowski', 'euclidean', 'hamming', 'canberra'],
    'test_size': [0.2, 0.25, 0.3],  # مقادیر مختلف برای test_size
    'random_state': [42, 0, 1, 2]   # مقادیر مختلف برای random_state
}

# 3. Create a function to evaluate the model
def evaluate_model(test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
    
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)

# 4. Store results for Grid Search
results = []

for test_size in param_grid['test_size']:
    for random_state in param_grid['random_state']:
        score = evaluate_model(test_size, random_state)
        results.append({'test_size': test_size, 'random_state': random_state, 'accuracy': score})

# 5. Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results)

# 6. Find the best combination
best_result = results_df.loc[results_df['accuracy'].idxmax()]

# 7. Print best parameters
print(f"Best Test Size: {best_result['test_size']}")
print(f"Best Random State: {best_result['random_state']}")
print(f"Best Accuracy: {best_result['accuracy']:.2f}")

# 8. Perform Grid Search for KNN with best test_size and random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=best_result['test_size'], random_state=int(best_result['random_state']))

# 9. Set up the parameter grid for KNN only
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'metric': ['minkowski', 'euclidean', 'hamming', 'canberra']
}

# 10. Create a KNN classifier
knn = KNeighborsClassifier()

# 11. Perform Grid Search for KNN
grid_search = GridSearchCV(knn, knn_param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 12. Get the best parameters and score
best_knn_params = grid_search.best_params_
best_knn_score = grid_search.best_score_

# 13. Evaluate on the test set
best_knn = KNeighborsClassifier(n_neighbors=best_knn_params['n_neighbors'], metric=best_knn_params['metric'])
best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# 14. Print results
print(f"Best KNN Parameters: {best_knn_params}")
print(f"Best KNN Cross-Validation Score: {best_knn_score:.2f}")
print(f"Final Test Accuracy: {test_accuracy:.2f}")

# 15. Improved Plotting of results
plt.figure(figsize=(10, 6))

# Extracting all metrics for visualization
metrics = ['n_neighbors', 'cross_val_score', 'final_test_accuracy']
values = [
    best_knn_params['n_neighbors'],
    best_knn_score,
    test_accuracy
]

# Create a bar chart with better representation
plt.bar(metrics, values, color=['skyblue', 'salmon', 'lightgreen'])
plt.ylabel('Values')
plt.title('Best Parameters from KNN Grid Search')

# Adding test accuracy as text on the plot
plt.text(2, test_accuracy + 0.02, f'Final Test Accuracy: {test_accuracy:.2f}', ha='center')

plt.xticks(rotation=45)
plt.grid(axis='y')
plt.ylim(0, 1.1)  # Set limit for better visibility
plt.show()
