# Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    RocCurveDisplay)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 1) Load dataset
df = pd.read_excel("Dataset2Use_Assignment1.xlsx")

# Feature columns
feature_cols = df.columns[0:11]

#Target columns
status_col = 'ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)'
year_col = 'ΕΤΟΣ'
healthy_df = df[df[status_col] == 1]
bankrupt_df = df[df[status_col] == 2]

# 2) Figure 1. Healthy vs Bankrupt companies per year
counts_per_year = (
    df
    .groupby([year_col, status_col])
    .size()
    .unstack(fill_value=0)
)

plt.figure(figsize=(10, 6))
ax = counts_per_year.plot(kind='bar', stacked=False)

plt.title("Number of Healthy and Bankrupt Companies per Year")
plt.xlabel("Year")
plt.ylabel("Number of Companies")
plt.legend(["Healthy", "Bankrupt"])
plt.grid(axis='y')

for container in ax.containers:
    ax.bar_label(container, label_type='edge', fontsize=9)

plt.tight_layout()
plt.show()

# Figure 2. Min/Max/Mean value per indicator for healthy and bankrupt companies
def compute_stats(dataframe, features):
    return pd.DataFrame({
        'min': dataframe[features].min(),
        'mean': dataframe[features].mean(),
        'max': dataframe[features].max()
    })

feature_columns = df.columns[0:8]

healthy_stats = compute_stats(healthy_df, feature_columns)
bankrupt_stats = compute_stats(bankrupt_df, feature_columns)

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Subplot for healthy companies
ax1 = axes[0]
healthy_stats.plot(kind='bar', ax=ax1)
ax1.set_title("Healthy Companies - Indicator Statistics")
ax1.set_ylabel("Value")
ax1.grid(axis='y')

for container in ax1.containers:
    ax1.bar_label(container, label_type='edge', fontsize=8, rotation=90)

# Subplot for bankrupt companies
ax2 = axes[1]
bankrupt_stats.plot(kind='bar', ax=ax2)
ax2.set_title("Bankrupt Companies - Indicator Statistics")
ax2.set_ylabel("Value")
ax2.grid(axis='y')

for container in ax2.containers:
    ax2.bar_label(container, label_type='edge', fontsize=8, rotation=90)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3) NaNs handling
print("\nChecking for NaN values")
if df.isnull().values.any():
    print("Attention: Missing values were found (NaNs)")
    print(df.isnull().sum())
    initial_rows = df.shape[0]
    df = df.dropna()
    print(f"Deleting {initial_rows - df.shape[0]} lines containing missing values")
else:
    print("No missing values were found")

# 4) Normalization [0,1]
X = df[feature_cols].values
y = df[status_col].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("Data normalized. X shape:", X_scaled.shape)

# 5) Stratified K-Fold
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

print("StratifiedKFold setup complete with 4 splits.")

# Classifiers
classifiers = {
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machines": SVC(kernel='rbf', probability=True, random_state=42),
    "Linear Support Vector Machines": SVC(kernel='linear', probability=True, random_state=42)
}

results = []

# Plots a confusion matrix as a figure
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Bankrupt', 'Healthy'])
    plt.yticks(tick_marks, ['Bankrupt', 'Healthy'])

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], 'd'),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

aggregated_preds = {}

for clf_name in classifiers.keys():
    aggregated_preds[clf_name] = {
        "train_true": [],
        "train_pred": [],
        "test_true": [],
        "test_pred": []
    }

fold_id = 1

# 7) Start a loop over folds
for train_index, test_index in skf.split(X_scaled, y):

    print(f"\n========== Fold {fold_id} ==========")

    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 6) Print distributions
    def print_distribution(name, labels):
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique, counts))
        healthy = dist.get(1, 0)
        bankrupt = dist.get(2, 0)
        print(f"{name}: Healthy={healthy}, Bankrupt={bankrupt}")

    print_distribution("Train (before balancing)", y_train)
    print_distribution("Test", y_test)

    # Separate classes
    healthy_idx = np.where(y_train == 1)[0]
    bankrupt_idx = np.where(y_train == 2)[0]
    
    # Balance the training set
    balanced = False

    if len(healthy_idx) > 3 * len(bankrupt_idx):
        np.random.seed(42)
        selected_healthy_idx = np.random.choice(healthy_idx, size=3 * len(bankrupt_idx), replace=False)

        balanced_idx = np.concatenate([selected_healthy_idx, bankrupt_idx])

        X_train_bal = X_train[balanced_idx]
        y_train_bal = y_train[balanced_idx]
        balanced = True
    else:
        X_train_bal = X_train
        y_train_bal = y_train

    print_distribution("Train (after balancing)", y_train_bal)
    print("\n")

    # 8) Train classifiers
    for clf_name, clf in classifiers.items():

        for dataset_name, Xtr, ytr, Xte, yte in [
            ("Train", X_train_bal, y_train_bal, X_train_bal, y_train_bal),
            ("Test", X_train_bal, y_train_bal, X_test, y_test)
        ]:

            clf.fit(Xtr, ytr)

            y_pred = clf.predict(Xte)
            y_prob = clf.predict_proba(Xte)[:, 1]
            
            # 9) Evaluate performance and create confusion matrix
            if dataset_name == "Train":
                aggregated_preds[clf_name]["train_true"].extend(yte)
                aggregated_preds[clf_name]["train_pred"].extend(y_pred)
            else:
                aggregated_preds[clf_name]["test_true"].extend(yte)
                aggregated_preds[clf_name]["test_pred"].extend(y_pred)

            cm = confusion_matrix(yte, y_pred, labels=[2, 1])
            TP = cm[0, 0]
            FN = cm[0, 1]
            FP = cm[1, 0]
            TN = cm[1, 1]

            # Calculate and print metrics
            acc = accuracy_score(yte, y_pred)
            prec = precision_score(yte, y_pred, pos_label=2)
            rec = recall_score(yte, y_pred, pos_label=2)
            f1 = f1_score(yte, y_pred, pos_label=2)
            auc = roc_auc_score((yte == 2).astype(int), y_prob)

            print(f"{clf_name} | {dataset_name} | "
                  f"Acc={acc:.2f}, Prec={prec:.2f}, Recall={rec:.2f}, "
                  f"F1={f1:.2f}, AUC={auc:.2f}")
            
            print("\n")

            # Store the results
            results.append({
                "Classifier Name": clf_name,
                "Dataset": dataset_name,
                "Balanced Train Set": "Yes" if balanced else "No",
                "Number of training samples": len(y_train_bal),
                "Number of non-healthy companies": np.sum(y_train_bal == 2),
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN,
                "ROC-AUC": auc
            })

    fold_id += 1

for clf_name, data in aggregated_preds.items():
    # Confusion matrix for "Train set"
    cm_train = confusion_matrix(data["train_true"], data["train_pred"], labels=[2, 1])
    plot_confusion_matrix(cm_train, title=f"{clf_name} | Train Set")

    # Confusion matrix for "Test set"
    cm_test = confusion_matrix(data["test_true"], data["test_pred"], labels=[2, 1])
    plot_confusion_matrix(cm_test, title=f"{clf_name} | Test Set")

# 10) Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("balancedDataOutcomes.csv", index=False)

print("\nResults saved to balancedDataOutcomes.csv")