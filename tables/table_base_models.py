from evaluation import ResultsEvaluation

if __name__ == "__main__":
    dataset = "dodh"
    table = "base_models"
    scorers_folder = "./scorers/{}/".format(dataset)
    results_folder = f"./results/{dataset}/{table + '/' if table is not None else ''}"
    consensus_folder = "./consensus/{}/".format(dataset)

    if dataset == "dodo" or dataset == "mass":
        result_evaluation = ResultsEvaluation(
            scorers_folder=scorers_folder, results_folder=results_folder
        )
    elif dataset == "dodh":
        result_evaluation = ResultsEvaluation(
            scorers_folder=scorers_folder,
            results_folder=results_folder,
            lights_off={
                "5bf0f969-304c-581e-949c-50c108f62846": 60,
                "b5d5785d-87ee-5078-b9b9-aac6abd4d8de": 60,
            },
            lights_on={"3e842aa8-bcd9-521e-93a2-72124233fe2c": 620},
        )

    result_evaluation.print_soft_agreements()
    result_evaluation.print_scores()

