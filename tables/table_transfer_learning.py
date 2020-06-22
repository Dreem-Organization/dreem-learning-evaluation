if __name__ == "__main__":
    from evaluation import ResultsEvaluation
    import numpy as np
    from evaluation import ResultsEvaluation, evaluation_bis
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
    import pandas as pd

    dataset = "dodh_to_dodo"
    results = {}
    for algo_name in ["SimpleSleepNet"]:
        print(algo_name)
        results[algo_name] = {
            "f1_score": [[], []],
            "accuracy_score": [[], []],
            "cohen_kappa": [[], []],
        }
        scorers_folder = "scorers/{}/".format(dataset.split("_")[-1])
        results_folder = f"results/transfer_learning/{dataset}/{algo_name}/"
        if dataset == "dodh_to_dodo":
            result_evaluation = ResultsEvaluation(
                scorers_folder=scorers_folder, results_folder=results_folder,

            )
            consensus_folder = "./consensus/{}/".format('dodo')
        elif dataset == "dodo_to_dodh":
            result_evaluation = ResultsEvaluation(
                scorers_folder=scorers_folder,
                results_folder=results_folder,
                lights_off={
                    "5bf0f969-304c-581e-949c-50c108f62846": 60,
                    "b5d5785d-87ee-5078-b9b9-aac6abd4d8de": 60,
                },
                lights_on={"3e842aa8-bcd9-521e-93a2-72124233fe2c": 620},
            )
            consensus_folder = "./consensus/{}/".format('dodh')
        hypnograms_model = []
        hypnograms_scorers = []
        for metric in ["f1_score", "accuracy_score", "cohen_kappa"]:
            for trial in result_evaluation.metrics:
                if trial in result_evaluation.result_hypnograms:
                    results[algo_name][metric][0] += [result_evaluation.metrics[trial][metric][0]]
                    results[algo_name][metric][1] += [result_evaluation.metrics[trial][metric][1]]
                    for record in result_evaluation.result_hypnograms[trial]:
                        hypnograms_model += result_evaluation.result_hypnograms[trial][
                            record].tolist()
                        hypnograms_scorers += result_evaluation.result_hypnograms_consensus[
                            record][0].tolist()
            results[algo_name][metric][0] = np.mean(results[algo_name][metric][0])
            results[algo_name][metric][1] = np.sqrt(
                np.mean(np.array(results[algo_name][metric][1]) ** 2)
            )
    for algo, metrics in results.items():
        print("Result for:", algo)
        for metric, values in metrics.items():
            print(metric, ":", values[0], "  sd:", values[1])
        print(" ")

    def f1_macro(x, y):
        return f1_score(x, y, average="macro", labels=[0, 1, 2, 3, 4])

    results = {}
    for metric in [f1_macro, accuracy_score, cohen_kappa_score]:
        results[metric.__name__] = evaluation_bis(
            results_folder, consensus_folder, metric
        )

    df = pd.DataFrame(results)
    df = df.iloc[[i for i, x in enumerate(df.index) if "std" not in x]]
    df = df.sort_values("f1_macro", ascending=False)
    print(df[["f1_macro"]].mean())
