## Dreem Open Dataset Evaluation (DOD-Evaluation)

DOD-Evaluation provides tools to build sleep staging consensus and compare sleep stagings to these consensus. 
Our paper CITER PAPER uses the code from this repository to computer the scorer consensus and the model performances.
The repository also contains each of the scorer's sleep staging on DOD-O and DOD-H, their consensus on both dataset and the results from all the models presented in the paper.

### Building a consensus from scorers
The individual sleep staging from each scorer for each record has to be provided in the 'scorers/dataset_name/scorer_name/' directory.
All the sleep staging are assumed to list of the same length, the sleep stages are denoted by the following numbers: 
{'Not scored': -1, 'Wake':0,'N1':1,'N2':2,'N3':3,'REM':4}.
Hence a sleep staging is simply a list of int between -1 and 4 stored in record_name.json file.

The consensus can then be built using *evaluation.ResultsEvaluation*:
```python
from evaluation import ResultsEvaluation

dataset = 'dodo'
scorers_folder = "scorers/{}/".format(dataset)

result_evaluation = ResultsEvaluation(
    scorers_folder=scorers_folder
)
result_evaluation.print_soft_agreements()
result_evaluation.print_scores()

```
Note that the consensus may be shorter than the initial staging due to unscored sleep epochs. 
Hence, the index of the first and last epoch of the consensus can be stored to truncate the PSG to truncate the PSG records properly
Consensus and start/end epoch can be saved using:
```python
import json
consensus_folder = 'consensus/dodo/'
time_reference_folder = 'time_reference/dodo/'
for record, hypno in result_evaluation.result_hypnograms_consensus.items():
    json.dump(hypno[0].tolist(),
              open('{}/hypnogram/{}.json'.format(consensus_folder, record), 'w'),
              indent=4)
    json.dump(hypno[1].tolist(),
              open('{}/probabilities/{}.json'.format(consensus_folder, record), 'w'),
              indent=4)

for record, index in result_evaluation.index.items():
    json.dump(int(index[0]),
              open(time_reference_folder + record + '/' + 'consensus_start', 'w'))
    json.dump(int(index[1]), open(time_reference_folder + record + '/' + 'consensus_end', 'w'))

```

### Extraction of the hypnograms from dreem-learning-open
dreem-learning-open can be used to generate k-folds and leave-one-out experiments, split_hypnograms.split_hypnograms contain a script to 
extract the hypnogram of each of the record from the appropriate fold and to save it into the specified folder.

### Evaluation of a sleep staging against consensus
Sleep stagings to evaluate are to be stored in "results/dataset_name/table_name/" in different folders. 
Each of the folder must contains the hypnogram outputted by the model for all the record of the consensus. 
Each of these hypnogram must have the same length as the consensus. The overall performance analysis can be performed with:

```python
dataset = 'dodo'
table = 'base_models'
scorers_folder = "scorers/{}/".format(dataset)
results_folder = f"results/{dataset}/{table + '/' if table is not None else ''}"
if dataset == 'dodo':
    result_evaluation = ResultsEvaluation(
        scorers_folder=scorers_folder,
        results_folder=results_folder

    )
elif dataset == 'dodh':
    result_evaluation = ResultsEvaluation(
        scorers_folder=scorers_folder,
        results_folder=results_folder,
        lights_off={
            "63b799f6-8a4f-4224-8797-ea971f78fb53": 60,
            "de3af7b1-ab6f-43fd-96f0-6fc64e8d2ed4": 60,
        },
        lights_on={
            "a14f8058-f636-4be7-a67a-8f7f91a419e7": 620,
        }
    )

result_evaluation.print_soft_agreements()
result_evaluation.print_scores()
```
