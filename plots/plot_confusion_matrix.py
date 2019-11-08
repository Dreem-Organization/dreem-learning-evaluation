import itertools

import matplotlib.pyplot as plt
import numpy as np

from evaluation import ResultsEvaluation


def plot_confusion_matrix(model_confusion_matrix, scorer_confusion_matrix, dir='figures/'):
    data1, data2 = model_confusion_matrix, scorer_confusion_matrix
    for toto in range(2):
        if toto == 0:
            f, axes = plt.subplots(1, 1, figsize=(12, 15))
            axes = (axes,)
        else:
            f, axes = plt.subplots(2, 1, figsize=(12, 15))

        plt.subplots_adjust(hspace=0.1)

        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("lol", ["#fff7fa", "#ED5C79"])

        for index_ax, (data, ax) in enumerate(zip([data1, data2], axes)):
            percentage = np.array(data["percentage"])
            number_of_epochs = np.array(data["number_of_epochs"])
            ax.imshow(
                percentage,
                interpolation='nearest',
                # cmap=plt.cm.RdPu,
                cmap=cmap,
                alpha=0.9,
                aspect=0.5
            )

            classes = ["Wake", "N1", "N2", "N3", "REM"]
            tick_marks = np.arange(len(classes))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(classes, fontsize=21)
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(classes, fontsize=21)
            ax.tick_params(top=True, bottom=False,
                           labeltop=True, labelbottom=False)

            plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                     rotation_mode="anchor")

            ax.set_xticks(np.arange(5 + 1) - .5, minor=True)
            ax.set_yticks(np.arange(5 + 1) - .5, minor=True)

            ax.grid(which="minor", color="w", linestyle='-', linewidth=4)

            # Turn spines off and create white grid.
            for edge, spine in ax.spines.items():
                spine.set_visible(False)

            thresh = np.array(percentage).max() / 2.
            for i, j in itertools.product(range(percentage.shape[0]), range(percentage.shape[1])):
                ax.text(j, i,
                        "{0:.1f}% ({1:})".format(percentage[i, j], int(number_of_epochs[i, j])),
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="white" if percentage[i, j] > thresh else "black",
                        fontsize=15)
            ax.set_ylabel("Consensus", fontsize=20)
            ax.set_xlabel("SimpleSleepNet" if index_ax == 0 else "Scorers", fontsize=20)
            plt.grid(False)
            ax.tick_params(which="minor", bottom=False, left=False)

        plt.savefig(f"{dir}confusion_matrix{'_alone' if toto == 0 else ''}.eps",
                    bbox_inches='tight', format='eps')
        plt.savefig(f"{dir}confusion_matrix{'_alone' if toto == 0 else ''}.png",
                    bbox_inches='tight', format='png')


if __name__ == "__main__":
    dataset = 'dod'
    table = 'base_models'
    models_to_plot = ['SimpleNet']
    scorers_folder = "scorers/{}/".format(dataset)
    results_folder = f"results/{dataset}/{table}/"
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

    confusion_matrix = result_evaluation.get_confusion_matrix()

    for model in models_to_plot:
        simple_net_confusion_matrix = {
            "number_of_epochs": confusion_matrix[1][model].tolist(),
            "percentage": (confusion_matrix[1][model] / np.sum(confusion_matrix[1][model], axis=1,
                                                               keepdims=True) * 100).tolist(),
        }

        all_scorers_confusion = np.array(
            [conf for scorer, conf in confusion_matrix[0].items()]).sum(0)
        scorers_confusion_matrix = {
            "number_of_epochs": all_scorers_confusion.tolist(),
            "percentage": (all_scorers_confusion / np.sum(all_scorers_confusion, axis=1,
                                                          keepdims=True) * 100).tolist(),
        }
        from plots.plot_confusion_matrix import plot_confusion_matrix

        plot_confusion_matrix(simple_net_confusion_matrix, scorers_confusion_matrix,
                              f"plots/{dataset}/{model}_")
