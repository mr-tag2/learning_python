import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# بارگذاری داده‌های سرطان سینه از sklearn
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# تقسیم داده‌ها به مجموعه‌های آموزشی و آزمایشی
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# استانداردسازی داده‌ها
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# تعریف متریک‌های مختلف
metrics = ['minkowski', 'euclidean', 'hamming','canberra']  # متریک‌های معتبر
accuracies = []

for metric in metrics:
    # ایجاد مدل KNN با متریک‌های مختلف
    knn = KNeighborsClassifier(metric=metric)
    knn.fit(X_train, y_train)
    
    # پیش‌بینی و ارزیابی مدل
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f'Accuracy with {metric} metric: {accuracy:.2f}')

# رسم نمودار
plt.figure(figsize=(10, 6))
plt.bar(metrics, accuracies, color=['blue', 'green', 'red','purple'])
plt.xlabel('Metric')
plt.ylabel('Accuracy')
plt.title('Accuracy of KNN with Different Metrics')
plt.show()
