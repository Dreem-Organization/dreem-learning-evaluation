if __name__ == "__main__":
    from evaluation import ResultsEvaluation
    import numpy as np

    dataset = 'dodo_to_dodh'
    results = {}
    for algo_name in ['SimpleSleepNet']:
        print(algo_name)
        results[algo_name] = {'f1_score': [[], []], 'accuracy_score': [[], []],
                              'cohen_kappa': [[], []]}
        scorers_folder = "scorers/{}/".format(dataset.split('_')[-1])
        results_folder = f"results/transfer_learning/{dataset}/{algo_name}/"
        if dataset == 'dodh_to_dodo':
            result_evaluation = ResultsEvaluation(
                scorers_folder=scorers_folder,
                results_folder=results_folder

            )
        elif dataset == 'dodo_to_dodh':
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

        for metric in ['f1_score', 'accuracy_score', 'cohen_kappa']:
            for trial in result_evaluation.metrics:
                results[algo_name][metric][0] += [result_evaluation.metrics[trial][metric][0]]
                results[algo_name][metric][1] += [result_evaluation.metrics[trial][metric][1]]
            results[algo_name][metric][0] = np.mean(results[algo_name][metric][0])
            results[algo_name][metric][1] = np.sqrt(
                np.mean(np.array(results[algo_name][metric][1]) ** 2))
    for algo, metrics in results.items():
        print('Result for:', algo)
        for metric, values in metrics.items():
            print(metric, ':', values[0], '  sd:', values[1])
        print(' ')
