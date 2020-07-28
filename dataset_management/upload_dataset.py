from trains import Task
from datetime import datetime

# create an dataset experiment
task = Task.init(project_name="diabetes prediction", task_name="upload dataset", output_uri="s3://allegro-datasets/blogs/data-management",task_type=Task.TaskTypes.data_processing)

date= {'date': str(datetime.date(datetime.now()))}
# add and upload local file containing our toy dataset
task.upload_artifact('dataset', artifact_object='data/diabetes_input.csv',metadata=date)

print('uploading artifacts in the background')

# we are done
print('see you next time')