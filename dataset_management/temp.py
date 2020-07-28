

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import csv
from sklearn import preprocessing
import numpy as np

input_file = 'data/diabetes_input.csv'
output_file = 'data/normalized.csv'

x_vals = []
y_vals = []
i=0
with open(input_file) as input_csv:
    csv_reader = csv.reader(input_csv, delimiter=',')
    for row in csv_reader:
        if i==0:
            i=1
            continue
        x_vals.append(float(row[2]))
        y_vals.append(int(row[-1]))


scaler = preprocessing.StandardScaler()
x_vals = np.array(x_vals).reshape(-1, 1)
x_scaled = scaler.fit_transform(x_vals)


# Split the data into training/testing sets
diabetes_X_train = x_scaled[:-20]
diabetes_X_test = x_scaled[-20:]

# Split the targets into training/testing sets
diabetes_y_train = y_vals[:-20]
diabetes_y_test = y_vals[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()