***Decision Tree Classification on the Titanic Dataset (Google Colab)***

## Objective

Train a Decision Tree model to predict whether a Titanic passenger survived.

---

# Step 1: Upload the Dataset

Upload both CSV files to Google Colab.

* `TitanicSurvivorTrainingDataset.csv`
* `TitanicSurvivorTestingDataset.csv`

Verify:

```python
!ls
```

---

# Step 2: Import Pandas

```python
import pandas as pd
```

### Why?

`pandas` is used to read and manipulate tabular data.

---

# Step 3: Load the Datasets

```python
df1 = pd.read_csv("filename")
df2 = pd.read_csv("filename")
```

### Why?

* `df1` = training dataset
* `df2` = testing dataset

Check the data:

```python
df1.head()
```

---

# Step 4: Understand the Columns

Display all column names.

```python
df1.columns
```

Example:

```
Unnamed: 0
PassengerId
Survived
Sex
Age
Fare
Pclass_1
Pclass_2
Pclass_3
Family_size
Title_1
Title_2
Title_3
Title_4
Emb_1
Emb_2
Emb_3
```

---

# Step 5: Select Features (X) and Target (y)

### Target

```
Survived
```

This is what the model must predict.

### Features

Everything except:

* `Unnamed: 0`
* `PassengerId`
* `Survived`

Create X and y.

```python
X = df1.drop(columns=["Unnamed: 0","PassengerId","Survived"])
y = df1["Survived"]
```

### Why?

* `X` = input features
* `y` = correct answers

---

# Step 6: Import the Decision Tree Algorithm

```python
from sklearn.tree import DecisionTreeClassifier
```

### Why?

Imports the Decision Tree classification algorithm.

---

# Step 7: Create the Model

```python
model = DecisionTreeClassifier(random_state=42)
```

### Why?

Creates an empty Decision Tree.

`random_state=42` ensures reproducible results.

---

# Step 8: Train the Model

```python
model.fit(X, y)
```

### Why?

The model learns patterns from the training data.

Think of it as:

```
Questions (X)
        ↓
Decision Tree learns
        ↓
Correct Answers (y)
```

---

# Step 9: Prepare the Test Dataset

Create testing features.

```python
X_test = df2.drop(columns=["Unnamed: 0","PassengerId","Survived"])
y_test = df2["Survived"]
```

### Why?

The testing data must contain the same input columns used during training.

---

# Step 10: Make Predictions

```python
y_pred = model.predict(X_test)
```

### Why?

The trained model predicts whether each passenger survived.

Possible outputs:

```
0 = Did not survive

1 = Survived
```

---

# Step 11: Calculate Accuracy

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
```

Example:

```
0.84
```

Meaning:

The model correctly predicted **84%** of passengers.

---

# Step 12: Generate the Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
```

Example:

```
[[54 10]
 [ 6 30]]
```

Interpretation:

| Actual | Predicted | Meaning             |
| ------ | --------- | ------------------- |
| 54     | 0→0       | True Negative (TN)  |
| 10     | 0→1       | False Positive (FP) |
| 6      | 1→0       | False Negative (FN) |
| 30     | 1→1       | True Positive (TP)  |

Correct predictions:

```
54 + 30 = 84
```

Wrong predictions:

```
10 + 6 = 16
```

---

# Step 13: Generate the Classification Report

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

Example:

```
precision    recall    f1-score

0
1

accuracy
```

---

# Evaluation Metrics

## Accuracy

Overall percentage of correct predictions.

Formula

```
(TP + TN) / Total Predictions
```

---

## Precision

"When the model predicts **Survived**, how often is it correct?"

Formula

```
TP / (TP + FP)
```

Higher precision means fewer false alarms.

---

## Recall

"Out of all actual survivors, how many did the model find?"

Formula

```
TP / (TP + FN)
```

Higher recall means fewer missed survivors.

---

## F1-score

Balances Precision and Recall.

Formula

```
2 × Precision × Recall
-----------------------
 Precision + Recall
```

Useful when both metrics are important.

---

## Support

Number of actual samples in each class.

Example:

```
64 passengers did not survive

36 passengers survived
```

---

# Complete Workflow

```
Upload Data
      ↓
Import Pandas
      ↓
Load CSV Files
      ↓
Inspect Columns
      ↓
Create X and y
      ↓
Import DecisionTreeClassifier
      ↓
Create Model
      ↓
Train Model (fit)
      ↓
Prepare Test Data
      ↓
Predict (predict)
      ↓
Accuracy
      ↓
Confusion Matrix
      ↓
Classification Report
      ↓
Interpret Results
```

---

# Key Functions

| Function                   | Purpose                                 |
| -------------------------- | --------------------------------------- |
| `pd.read_csv()`            | Load CSV                                |
| `head()`                   | View first rows                         |
| `columns`                  | Display column names                    |
| `drop()`                   | Remove unwanted columns                 |
| `DecisionTreeClassifier()` | Create the model                        |
| `fit()`                    | Train the model                         |
| `predict()`                | Make predictions                        |
| `accuracy_score()`         | Calculate accuracy                      |
| `confusion_matrix()`       | Show prediction errors                  |
| `classification_report()`  | Generate Precision, Recall and F1-score |

---

# Final Outcome

You have learned how to:

* Load datasets
* Create features and target
* Train a Decision Tree classifier
* Predict unseen data
* Evaluate the model using:

  * Accuracy
  * Confusion Matrix
  * Precision
  * Recall
  * F1-score

This is the complete end-to-end workflow for a basic Decision Tree classification model in scikit-learn.
