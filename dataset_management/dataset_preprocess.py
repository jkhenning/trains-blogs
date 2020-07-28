from trains import Task
import csv

import numpy as np
from datetime import datetime
from sklearn import preprocessing

task = Task.init(project_name="diabetes prediction", task_name="ingesting dataset", task_type=Task.TaskTypes.data_processing)

args = {
    'dataset_task_id': 'c9943dc9aba047cea250364277061dbc',
    'test_size': 20
}

task.connect(args)

dataset_upload_task = Task.get_task(task_id=args['dataset_task_id'])
print('Input task id={} artifacts {}'.format(args['dataset_task_id'], list(dataset_upload_task.artifacts.keys())))
# download the csv file
source_ds = dataset_upload_task.artifacts['dataset'].get_local_copy()

x_vals = []
y_vals = []

# Converting CSV file to lists
with open(source_ds) as input_csv:
    csv_reader = csv.reader(input_csv, delimiter=',')
    next(csv_reader)
    for row in csv_reader:
        x_vals.append(float(row[2]))
        y_vals.append(int(row[-1]))


scaler = preprocessing.StandardScaler()
x_vals = np.array(x_vals).reshape(-1, 1)
x_scaled = scaler.fit_transform(x_vals)
y_vals = np.array(y_vals)

x_train = x_scaled[:-args['test_size']]
x_test = x_scaled[-args['test_size']:]

y_train = y_vals[:-args['test_size']]
y_test = y_vals[-args['test_size']:]

date= {'date': str(datetime.date(datetime.now()))}

task.upload_artifact('x_train', artifact_object=x_train, metadata=date)
task.upload_artifact('x_test', artifact_object=x_test, metadata=date)
task.upload_artifact('y_train', artifact_object=y_train, metadata=date)
task.upload_artifact('y_test', artifact_object=y_test, metadata=date)


print('all done!')
