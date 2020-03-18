import json
import os

import pandas as pd

FOLDER = 'results/mass/'
experiments = os.listdir(FOLDER)

experiment_performance = []
records = []
for experiment in experiments:
    experiment_folder = f"{FOLDER}{experiment}/"

    for fold in os.listdir(experiment_folder):
        fold_description = json.load(open(f"{experiment_folder}/{fold}/description.json"))
        del_fold = False
        for record, perf_on_record in fold_description["performance_per_records"].items():
            perf_on_record.update({'record': record, 'experiment': experiment})
            experiment_performance += [perf_on_record]



df = pd.DataFrame(experiment_performance)
print(df.groupby('experiment').mean().sort_values('accuracy', ascending=False))
print(df.groupby('experiment').std().sort_values('accuracy'))

import shutil
import os
path = "results/mass/"
for experiment in os.listdir(path):
    experiment_path = f"{path}/{experiment}/"
    for fold_id in os.listdir(experiment_path):
        fold_path = f"{experiment_path}/{fold_id}/"
        for file in os.listdir(fold_path):
            if file != 'description.json':
                try:
                    shutil.rmtree(f"{fold_path}/{file}")
                except NotADirectoryError:
                    os.remove(f"{fold_path}/{file}")

