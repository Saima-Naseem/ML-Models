# ML Models

Ongoing machine learning assignment repo models, code, and explanations submitted for ML coaching.

## Contents
- **linear-regression/** Linear Regression model predicting diabetes progression

---

##  Multi Linear Regression (Diabetes Dataset)

### Goal
Predict disease progression in diabetes patients using health measurements (age, BMI, blood pressure, etc.) with a simple linear regression model.

### Dataset
Built-in `diabetes` dataset from `sklearn.datasets` consisting of 442 patients, 10 health features, and 1 target (disease progression score).

### Steps Followed

1. **Load data**: Imported the diabetes dataset from sklearn.
2. **Structure data**: Converted raw data into a labeled pandas table (`X` = features, `y ' = target).
3. **Train/test split**: Split data into 80% training, 20% testing (`random_state=42` for reproducibility).
4. **Train model**: Fit a `Multi LinearRegression` model on the training data.
5. **Predict**: Tested the trained model to predict target values on unseen test data.
6. **Evaluate**: Measured performance using:
   - **MSE (Mean Squared Error)**: average squared difference between predicted and actual values (lower = better)
   - **R² Score**: how much variance in the target is explained by the model (closer to 1 = better)
7. **Visualize**: Plotted actual vs predicted values, with a red dashed diagonal line representing perfect predictions.

### Result
The model shows a clear positive trend (predictions follow actual values), with moderate accuracy, typical for a multilinear model on real-world medical data with non-linear relationships.

# Analysis Summary: Multiple Linear Regression — Diabetes Dataset

## A. Goal
Predict how much a patient's diabetes will progress using their health stats.

## B. Dataset
Used sklearn's built-in `diabetes` dataset, 442 patients, 10 health features (age, BMI, blood pressure, etc.), 1 target value (disease progression score).

## C. Loaded the Data
``` python
from sklearn.datasets import load_diabetes
data = load_diabetes()
```
Pulled the dataset into memory.

## D. Structured It
```python
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
```
- `X` = labeled table of all 10 features (inputs)
- `y` = target values (what we want to predict)

## E. Split into Train/Test
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
80% of patients (~353) for training, 20% (~89) held back for testing.

## F. Trained the Model
```python
model = LinearRegression()
model.fit(X_train, y_train)
```
Model studied the training data and learned how all 10 features together relate to disease progression, which makes it a **multiple** linear regression (10 inputs, not 1).

## G. Made Predictions
```python
y_pred = model.predict(X_test)
```
Asked the model to guess progression scores for the unseen test patients.

## H. Evaluated Accuracy
```python
mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)
```
- **MSE** ≈ measures average prediction error (lower = better)
- **R²** ≈ 0.45–0.5 model explains roughly half the variation in outcomes; decent for a linear model on complex medical data

## I. Visualized Results
```python
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
```
Plotted actual vs. predicted values, with a red dashed line marking "perfect prediction." Dots followed the trend but scattered around the line, meaning the model learned a real pattern, but it isn't fully precise.

## J. Conclusion
The model successfully captures a meaningful relationship between patient health metrics and diabetes progression, though a straight-line model has limits on real-world biological data with non-linear effects.

### How to Run this Model
1. Download `multilinear_regression.ipynb` from this repo
2. Upload it to your Google Colab
3. Run all cells in order (top to bottom)
4. View printed MSE/R² scores and the final plot
