import json
import os
from collections import OrderedDict
from functools import partial

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score
from tabulate import tabulate


def compute_soft_agreement(hypnogram, hypnograms_consensus):
    epochs = range(len(hypnogram))
    probabilistic_consensus = np.zeros((6, len(hypnogram)))
    for hypnogram_consensus in hypnograms_consensus:
        probabilistic_consensus[np.array(hypnogram_consensus) + 1, range(len(hypnogram))] += 1
    probabilistic_consensus_normalized = probabilistic_consensus / probabilistic_consensus.max(0)
    soft_agreement = probabilistic_consensus_normalized[np.array(hypnogram) + 1, epochs].mean()
    return soft_agreement


def build_consensus_hypnogram(ranked_hypnograms_consensus):
    """In this function order matters, first hypnogram is the reference in case of ties"""
    number_of_epochs = len(ranked_hypnograms_consensus[0])
    probabilistic_consensus = np.zeros((6, number_of_epochs))
    ranked_hypnograms_consensus_array = np.array(ranked_hypnograms_consensus) + 1
    for hypnogram_consensus in ranked_hypnograms_consensus_array:
        probabilistic_consensus[np.array(hypnogram_consensus), range(number_of_epochs)] += 1

    consensus_hypnogram = np.argmax(probabilistic_consensus, 0)
    ties = (
                   probabilistic_consensus ==
                   probabilistic_consensus[consensus_hypnogram, range(number_of_epochs)]
           ).sum(0) > 1
    consensus_hypnogram[ties] = np.array(ranked_hypnograms_consensus_array[0])[ties]
    consensus_probability = (probabilistic_consensus[consensus_hypnogram, range(number_of_epochs)] /
                             len(ranked_hypnograms_consensus_array))
    consensus_hypnogram = consensus_hypnogram - 1
    return consensus_hypnogram, consensus_probability


def get_cohen_kappa(hypnogram, consensus, stage=None):
    consensus_hypnogram, consensus_probability = consensus
    mask = (consensus_hypnogram != -1)
    y1 = consensus_hypnogram[mask]
    y2 = hypnogram[mask]
    if stage is not None:
        y1 = y1 == stage
        y2 = y2 == stage
    score = cohen_kappa_score(y1, y2, sample_weight=consensus_probability[mask])
    return score


def get_f1_score(hypnogram, consensus, stage=None):
    consensus_hypnogram, consensus_probability = consensus
    mask = (consensus_hypnogram != -1)
    if stage is None:
        score = f1_score(
            consensus_hypnogram[mask],
            hypnogram[mask],
            labels=[0, 1, 2, 3, 4],
            average="weighted",
            sample_weight=consensus_probability[mask]
        )
    else:
        score = f1_score(
            consensus_hypnogram[mask],
            hypnogram[mask],
            labels=[0, 1, 2, 3, 4],
            average=None,
            sample_weight=consensus_probability[mask]
        )[stage]
    return score


def get_accuracy_score(hypnogram, consensus, stage=None):
    consensus_hypnogram, consensus_probability = consensus
    if stage is not None:
        mask = (consensus_hypnogram == stage)
    else:
        mask = (consensus_hypnogram != -1)
    y1 = consensus_hypnogram[mask]
    y2 = hypnogram[mask]
    p = consensus_probability[mask]

    score = ((y1 == y2) * p).sum() / p.sum()
    return score


get_metrics = OrderedDict({
    "f1_score": get_f1_score,
    "accuracy_score": get_accuracy_score,
    "cohen_kappa": get_cohen_kappa,
    "f1_score_0": partial(get_f1_score, stage=0),
    "accuracy_score_0": partial(get_accuracy_score, stage=0),
    "cohen_kappa_0": partial(get_cohen_kappa, stage=0),
    "f1_score_1": partial(get_f1_score, stage=1),
    "accuracy_score_1": partial(get_accuracy_score, stage=1),
    "cohen_kappa_1": partial(get_cohen_kappa, stage=1),
    "f1_score_2": partial(get_f1_score, stage=2),
    "accuracy_score_2": partial(get_accuracy_score, stage=2),
    "cohen_kappa_2": partial(get_cohen_kappa, stage=2),
    "f1_score_3": partial(get_f1_score, stage=3),
    "accuracy_score_3": partial(get_accuracy_score, stage=3),
    "cohen_kappa_3": partial(get_cohen_kappa, stage=3),
    "f1_score_4": partial(get_f1_score, stage=4),
    "accuracy_score_4": partial(get_accuracy_score, stage=4),
    "cohen_kappa_4": partial(get_cohen_kappa, stage=4),
})


class ResultsEvaluation:

    def __init__(self,
                 scorers_folder,
                 results_folder=None,
                 record_blacklist=[],
                 lights_off={},
                 lights_on={},
                 start_times=None):
        # Retrieve scorers
        self.scorers = os.listdir(scorers_folder)
        self.index = {}
        self.scorers_folder = {
            scorer: f'{scorers_folder}{scorer}/'
            for scorer in self.scorers
        }
        # Intersection of all available scored records
        self.records = sorted(list(set.intersection(*(
            {record.split(".json")[0] for record in os.listdir(self.scorers_folder[scorer]) if
             record.split(".json")[0] not in record_blacklist}
            for scorer in self.scorers
        ))))
        print(f"Found {len(self.records)} records and {len(self.scorers)} scorers.")

        # Retrieve scorer hypnograms
        self.scorer_hypnograms = {
            scorer: {
                record: np.array(
                    json.load(open(f"{self.scorers_folder[scorer]}/{record}.json", "r")))
                for record in self.records
            }
            for scorer in self.scorers
        }

        # Retrieve results hypnograms
        if results_folder is not None:
            self.results = os.listdir(results_folder)
        else:
            self.results = []

        self.results_folder = {
            result: f'{results_folder}{result}/'
            for result in self.results

        }

        self.result_hypnograms = {
            result: {
                record: np.array(
                    json.load(open(f"{self.results_folder[result]}/{record}", "r")))
                for record in os.listdir(self.results_folder[result])
            }
            for result in self.results
        }

        # Cut hypnograms to light on and off
        for record in self.records:
            hypnograms = [self.scorer_hypnograms[scorer][record] for scorer in self.scorers]
            index_min = max([np.where(np.array(hypnogram) >= 0)[0][0]
                             for hypnogram in hypnograms])
            index_min = max(index_min, lights_off.get(record, 0))
            index_max = (
                    len(hypnograms[0]) - max([np.where(np.array(hypnogram)[::-1] >= 0)[0][0]
                                              for hypnogram in hypnograms]))
            index_max = min(index_max, lights_on.get(record, np.inf))
            self.index[record] = index_min, index_max
            for result in self.results:
                try:
                    self.result_hypnograms[result][record] = self.result_hypnograms[result][record]
                except KeyError:
                    pass
            for scorer in self.scorers:
                self.scorer_hypnograms[scorer][record] = self.scorer_hypnograms[scorer][record][
                                                         index_min:index_max]

        # Build up scorer ranking
        self.scorers_ranking = {
            record: sorted(
                self.scorers,
                key=lambda scorer: -compute_soft_agreement(
                    self.scorer_hypnograms[scorer][record],
                    [self.scorer_hypnograms[other_scorer][record]
                     for other_scorer in self.scorers if other_scorer != scorer],
                )
            )
            for record in self.records
        }
        self.scorers_soft_agreement = [
            (scorer,
             np.mean([
                 compute_soft_agreement(
                     self.scorer_hypnograms[scorer][record],
                     [self.scorer_hypnograms[other_scorer][record]
                      for other_scorer in self.scorers if other_scorer != scorer]
                 ) for record in self.records
             ]))
            for scorer in sorted(self.scorers)
        ]

        # Build consensus hypnogram for scorers
        self.scorer_hypnograms_consensus = {
            scorer: {
                record: build_consensus_hypnogram(
                    [self.scorer_hypnograms[other_scorer][record] for other_scorer in
                     self.scorers_ranking[record]
                     if other_scorer != scorer]
                )
                for record in self.records
            }
            for scorer in self.scorers
        }

        # Metrics for scorers
        self.metrics = {}
        for scorer in self.scorers:
            self.metrics[scorer] = {}
            for metric_name, get_metric in get_metrics.items():
                values = [
                    get_metric(
                        self.scorer_hypnograms[scorer][record],
                        self.scorer_hypnograms_consensus[scorer][record],
                    ) for record in self.records
                ]
                self.metrics[scorer][metric_name] = (np.nanmean(values), np.nanstd(values))

        # Metrics for overall scorers
        self.metrics["Overall Scorers"] = {}
        for metric_name, get_metric in get_metrics.items():
            values = [
                get_metric(
                    self.scorer_hypnograms[scorer][record],
                    self.scorer_hypnograms_consensus[scorer][record],
                ) for record in self.records for scorer in self.scorers
            ]
            self.metrics["Overall Scorers"][metric_name] = (np.nanmean(values), np.nanstd(values))

        # Metrics for results
        self.result_hypnograms_consensus = {
            record: build_consensus_hypnogram(
                [self.scorer_hypnograms[scorer][record]
                 for scorer in self.scorers_ranking[record][:-1]]  # N - 1 scorings
            )
            for record in self.records
        }

        for result in self.results:
            self.metrics[result] = {}
            for metric_name, get_metric in get_metrics.items():
                values = [
                    get_metric(
                        self.result_hypnograms[result][record],
                        self.result_hypnograms_consensus[record[:-5]],
                    ) for record in os.listdir(self.results_folder[result])
                ]
                self.metrics[result][metric_name] = (np.nanmean(values), np.nanstd(values))

    def print_soft_agreements(self):
        print(
            tabulate(
                self.scorers_soft_agreement,
                headers=["Scorer", "SoftAgreement"],
                tablefmt="fancy_grid"
            )
        )

    def print_scores(self):
        keys = self.results + ["Overall Scorers"] + sorted(self.scorers)
        print(
            tabulate(
                [
                    [metric_key] + [
                        (f"{round(self.metrics[key][metric_key][0] * 100, 1)} ± "
                         f"{round(self.metrics[key][metric_key][1] * 100, 1)}")
                        for key in keys]
                    for metric_key in get_metrics.keys()
                ],
                headers=keys,
                tablefmt="fancy_grid"
            )
        )

    def return_scores(self):
        keys = self.results + ["Overall Scorers"] + sorted(self.scorers)
        return tabulate(
            [
                [metric_key] + [
                    (f"{round(self.metrics[key][metric_key][0] * 100, 1)} ± "
                     f"{round(self.metrics[key][metric_key][1] * 100, 1)}")
                    for key in keys]
                for metric_key in get_metrics.keys()
            ],
            headers=keys,
            tablefmt="latex_raw"
        )


if __name__ == "__main__":
    dataset = 'dodo'

    scorers_folder = "scorers/{}/".format(dataset)
    results_folder = f"results/{dataset}/learning_curve/"
    train_dataset_size = os.listdir(results_folder)
    results = {}
    for size in train_dataset_size:
        result_folder = results_folder + str(size) + '/'
        if dataset == 'dodo':
            result_evaluation = ResultsEvaluation(
                scorers_folder=scorers_folder,
                results_folder=result_folder

            )
        elif dataset == 'dodh':
            result_evaluation = ResultsEvaluation(
                scorers_folder=scorers_folder,
                results_folder=result_folder,
                lights_off={
                    "63b799f6-8a4f-4224-8797-ea971f78fb53": 60,
                    "de3af7b1-ab6f-43fd-96f0-6fc64e8d2ed4": 60,
                },
                lights_on={
                    "a14f8058-f636-4be7-a67a-8f7f91a419e7": 620,
                }

            )

        metrics_to_track = {'accuracy_score': [], 'cohen_kappa': [], "f1_score": []}
        for trial, metrics in result_evaluation.metrics.items():
            for metric in metrics_to_track:
                metrics_to_track[metric] += [metrics[metric][0] * 100]
        results[int(size)] = metrics_to_track['accuracy_score']

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    df = pd.DataFrame.from_dict(results).melt().sort_values('variable')
    df_agg = df.groupby('variable').mean()
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(12, 10))
    sns.barplot(x='variable', y='value', data=df, color="#FF0064", saturation=1, alpha=0.6)
    ax = sns.pointplot(x='variable', y='value', data=df, markers="", join=False, color='black')
    if dataset == 'dodh':
        plt.axhline(y=0.887 * 100, linestyle=':', xmin=-0.5, xmax=8.5, label='Best scorer',
                    c='black')
        plt.axhline(y=0.868 * 100, linestyle='--', xmin=-0.5, xmax=8.5, label='Avg. scorer',
                    c='black')
    elif dataset == 'dodo':
        plt.axhline(y=0.880 * 100, linestyle=':', xmin=-0.5, xmax=11.5, label='Best scorer',
                    c='black')
        plt.axhline(y=0.848 * 100, linestyle='--', xmin=-0.5, xmax=11.5, label='Avg. scorer',
                    c='black')

    labels = ["Best scorer", 'Scorers avg.']
    handles, _ = ax.get_legend_handles_labels()
    plt.xlabel('Training set size', fontsize=35)
    plt.ylabel('F1 (%)', fontsize=35)
    plt.legend(handles=handles, labels=labels, loc='lower center')
    plt.ylim(0, 100)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig(f"plots/{dataset}/learning_curve.png")
