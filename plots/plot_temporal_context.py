from evaluation import ResultsEvaluation

if __name__ == "__main__":
    dataset = 'dodh'
    table = 'temporal_context'

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

    metric = 'f1_score'
    results = []
    for model, performance in result_evaluation.metrics.items():
        if 'scorer' not in model and 'Scorer' not in model:
            temporal_context = model.split('_')[-1]
            results += [[int(temporal_context), performance[metric][0], performance[metric][1]]]
            print(results[-1])

    import numpy as np
    import matplotlib.pyplot as plt

    results = np.array(results)
    results = results[results[:, 0].argsort()]
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111)
    plt.bar(results[:, 0].astype(int).astype(str), results[:, 1] * 100, yerr=results[:, 2] * 100,
            color="#FF0064", alpha=0.6)
    plt.ylim(0, 100)
    plt.xlabel('Temporal context', fontsize=35)
    plt.ylabel('F1 (%)', fontsize=35)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(f"plots/{dataset}/temporal_context.png")
