from evaluation import ResultsEvaluation


if __name__ == "__main__":

    table = 'base_models'

    for dataset in ["dodo", "dodh"]:
        scorers_folder = "./scorers/{}/".format(dataset)

        if dataset == 'dodo':
            result_evaluation = ResultsEvaluation(
                scorers_folder=scorers_folder,
                results_folder=None

            )
        elif dataset == 'dodh':
            result_evaluation = ResultsEvaluation(
                scorers_folder=scorers_folder,
                results_folder=None,
                lights_off={
                    "63b799f6-8a4f-4224-8797-ea971f78fb53": 60,
                    "de3af7b1-ab6f-43fd-96f0-6fc64e8d2ed4": 60,
                },
                lights_on={
                    "a14f8058-f636-4be7-a67a-8f7f91a419e7": 620,
                }
            )

        result_evaluation.print_demographics()
