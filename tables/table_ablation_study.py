from evaluation import ResultsEvaluation

if __name__ == "__main__":
    dataset = 'dodh'
    table = 'ablation_simple_net'
    scorers_folder = "./scorers/{}/".format(dataset)
    results_folder = f"./results/{dataset}/{table + '/' if table is not None else ''}"
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
