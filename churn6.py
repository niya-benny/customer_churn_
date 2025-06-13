import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.utils.class_weight import compute_class_weight

#======================================================================================
# Process 1: Load Dataset
df = pd.read_csv("train.csv")

#===================================================================================
# Process 2: Study dataset
buffer = io.StringIO()
df.info(buf=buffer)
info = buffer.getvalue()

with open("P2.study the dataset", "w") as f:
    f.write("\nHead:\n" + df.head().to_string())
    f.write("\nShape:\n" + str(df.shape))
    f.write("\nColumn info:\n" + info)
    f.write("\nUnique values in columns:\n" + df.nunique().to_string())
    f.write("\nStatistics:\n" + df.describe().to_string())
    f.write("\nChurn Distribution: \n" + df['Churn'].value_counts().to_string())

#===============================================================================
# Process 3: Basic Preprocessing
with open("P3.Basic Preprocessing", "w") as f:
    f.write("\nMissing values:\n" + df.isnull().sum().to_string())
    f.write("\nDuplicate CustomerID:\n" + str(df.duplicated(subset="CustomerID").sum()))


df.drop(columns=['CustomerID'], inplace=True)
#==========================================================================================
# Process 4: EDA

# Countplot for categorical variables
categorical_cols = df.select_dtypes(include='object').columns.tolist()
target_col = 'Churn'

n_cols = 3
n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
axes = axes.flatten() #making 1D array

for i, col in enumerate(categorical_cols):
    sns.countplot(data=df, x=col, hue=target_col, ax=axes[i])
    axes[i].set_title(f'Churn by {col}')
    axes[i].tick_params(axis='x', rotation=45)

#remove any extra (unused) subplot axes that were created but not filled with a plot
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("P4.Categorical_Countplots.png")

numerical_cols = df.select_dtypes(include=['int64', 'float64']).drop(columns=["Churn"])
plt.figure(figsize=(12, 8))
sns.heatmap(df[numerical_cols.columns.tolist() + ['Churn']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig("P4.Numerical_CorrelationMatrix.png")

#=============================================================================================
#=======================================================================================
# Process 5-6: Feature Importance Using Random Forest & Select Important Features

# Encode categorical features
df_encoded = df.copy()
encoders = {}  # Dict to store each LabelEncoder

for col in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le  # Store the encoder

X_all = df_encoded.drop(columns=['Churn'])
y_all = df_encoded['Churn']

# Train Random Forest only to get feature importances
temp_rf = RandomForestClassifier(random_state=42)
temp_rf.fit(X_all, y_all)

# Get feature importances and normalize to percentage
importances = pd.Series(temp_rf.feature_importances_, index=X_all.columns)
importances_percent = 1000 * importances / importances.sum()
important_features = importances_percent[importances_percent >= 20].sort_values(ascending=False)

# Save feature importances
with open("P5.Feature Importance", "w", encoding="utf-8") as f:
    f.write("Feature Importances (%):\n" + importances_percent.sort_values(ascending=False).to_string())
    f.write("\n\nSelected Important Features (≥20% impact):\n" + important_features.to_string())

# Keep only selected features
df_encoded = df_encoded[important_features.index.tolist() + ['Churn']]

with open("P6.Keep only Important Features", "w", encoding="utf-8") as f:
    f.write("Selected Features based on ≥20% importance:\n")
    f.write(", ".join(important_features.index.tolist()))
    f.write("\n\nNew df shape:\n" + str(df_encoded.shape))
    f.write("\nNew df head:\n" + df_encoded.head().to_string())


#==========================================================================================
# Process 7: Train-Test Split

X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)


#=====================================================================================
# # Process 8: Train and Evaluate Logistic Regression with Custom Class Weights + Threshold Tuning
#
# Compute class weights
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))
print("Class weights:", class_weight_dict)

# class_weight_dict[1]*=1.5

# Train logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=3000,class_weight=class_weight_dict)
logreg.fit(X_train, y_train)

# Predict probabilities
y_scores = logreg.predict_proba(X_test)[:, 1]

# # Tune threshold
threshold = 0.67
y_pred_logreg = (y_scores >= threshold).astype(int)

# Metrics
logreg_auc = roc_auc_score(y_test, y_scores)
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)
avg_precision = average_precision_score(y_test, y_scores)

cm = confusion_matrix(y_test, y_pred_logreg)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Unpack confusion matrix
TN,FP,FN,TP = cm.ravel()

Total=TP+TN+FP+FN
# Accuracy
accuracy =(TP+TN)/(TP+TN+FP+FN)

# Precision (Positive Predictive Value)
precision =TP/(TP+FP) if (TP+FP) != 0 else 0

# Recall (Sensitivity or True Positive Rate)
recall = TP/(TP+FN) if (TP+FN) != 0 else 0

# F1 Score
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

with open("P8.Model Evaluation", "w") as f:
    f.write("=== Logistic Regression ===\n")
    f.write(f"Threshold used: {threshold}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1_score:.4f}\n\n")
    f.write(f"classification report: \n {classification_report(y_test, y_pred_logreg, target_names=["No Churn", "Churn"])}")
    f.write(f"\nROC AUC: {logreg_auc:.4f}\n")
    f.write(f"PR AUC: {pr_auc:.4f}\n")
    f.write(f"Average Precision: {avg_precision:.4f}\n")

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}\nAvg Precision = {avg_precision:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Logistic Regression')
plt.legend(loc='lower left')
plt.grid()
plt.savefig("P8.Precision_Recall_Curve_LR.png")
plt.close()

#==========================================================================================
# Process 9: Confusion Matrix Visualization

ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test) #compute and plot the confusion matrix in one function
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig("P9.ConfusionMatrix_LR.png")
#====================================================================================
# Process 10: Save Models

# Save the encoders dictionary
joblib.dump(encoders, 'encoders6.joblib')

# Save selected feature names (X_train.columns after feature selection)
joblib.dump(list(X_train.columns), 'selected_features.joblib')

joblib.dump(logreg, 'logreg_model.joblib')
print("Logistic Regression model saved to 'logreg_model.joblib'")