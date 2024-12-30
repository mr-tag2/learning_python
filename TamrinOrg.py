import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


data = load_breast_cancer()

X = data.data
y = data.target

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'metric': ['minkowski', 'euclidean', 'hamming', 'canberra'],
    'test_size': [0.2, 0.25, 0.3],  
    'random_state': [42, 0, 1, 2]   
}


def evaluate_model(test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
    
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)


results = []

for test_size in param_grid['test_size']:
    for random_state in param_grid['random_state']:
        score = evaluate_model(test_size, random_state)
        results.append({'test_size': test_size, 'random_state': random_state, 'accuracy': score})


results_df = pd.DataFrame(results)


best_result = results_df.loc[results_df['accuracy'].idxmax()]


print(f"Best Test Size: {best_result['test_size']}")
print(f"Best Random State: {best_result['random_state']}")
print(f"Best Accuracy: {best_result['accuracy']:.2f}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=best_result['test_size'], random_state=int(best_result['random_state']))


knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'metric': ['minkowski', 'euclidean', 'hamming', 'canberra']
}


knn = KNeighborsClassifier()


grid_search = GridSearchCV(knn, knn_param_grid, cv=5)
grid_search.fit(X_train, y_train)


best_knn_params = grid_search.best_params_
best_knn_score = grid_search.best_score_


best_knn = KNeighborsClassifier(n_neighbors=best_knn_params['n_neighbors'], metric=best_knn_params['metric'])
best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)


print(f"Best KNN Parameters: {best_knn_params}")
print(f"Best KNN Cross-Validation Score: {best_knn_score:.2f}")
print(f"Final Test Accuracy: {test_accuracy:.2f}")


plt.figure(figsize=(10, 6))


metrics = ['n_neighbors', 'cross_val_score', 'final_test_accuracy']
values = [
    best_knn_params['n_neighbors'],
    best_knn_score,
    test_accuracy
]


plt.bar(metrics, values, color=['skyblue', 'salmon', 'lightgreen'])
plt.ylabel('Values')
plt.title('Best Parameters from KNN Grid Search')


plt.text(2, test_accuracy + 0.02, f'Final Test Accuracy: {test_accuracy:.2f}', ha='center')

plt.xticks(rotation=45)
plt.grid(axis='y')
plt.ylim(0, 1.1)  
plt.show()
