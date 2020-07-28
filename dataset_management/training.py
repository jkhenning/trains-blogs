from trains import Task

from sklearn import  linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

task = Task.init(project_name="diabetes prediction", task_name="train linear regressor")

args = {
    'dataset_task_id': '1367baf7933e4759a4430a12730ad389'
}

task.connect(args)
logger = task.get_logger()

dataset_task = Task.get_task(task_id=args['dataset_task_id'])
print('Input task id={} artifacts {}'.format(args['dataset_task_id'], list(dataset_task.artifacts.keys())))
# download the artifact and open it
x_train = dataset_task.artifacts['x_train'].get()
x_test = dataset_task.artifacts['x_test'].get()
y_train = dataset_task.artifacts['y_train'].get()
y_test = dataset_task.artifacts['y_test'].get()

regr = linear_model.LinearRegression()


# Train the model using the training sets
regr.fit(x_train, y_train)


# Make predictions using the testing set
y_pred = regr.predict(x_test)

joblib.dump(regr, 'model.pkl', compress=True)

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

logger.report_scalar(title='Prediction results',series="Coefficients",value=regr.coef_,iteration=1)
logger.report_scalar(title='Prediction results',series="MSE",value=mean_squared_error(y_test, y_pred),iteration=1)
logger.report_scalar(title='Prediction results',series="Coefficient of determination",value=r2_score(y_test, y_pred),iteration=1)

# Plot outputs
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
