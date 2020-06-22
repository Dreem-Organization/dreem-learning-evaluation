from evaluation import ResultsEvaluation, evaluation_bis
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import pandas as pd
import os

if __name__ == "__main__":
    dataset = "sleep_edf"
    table = "extended"
    results_folder = f"./results/{dataset}/{table + '/' if table is not None else ''}"
    consensus_folder = "./consensus/{}/".format(dataset)

    records = [x.replace(".json", "") for x in os.listdir(consensus_folder)]
    records.sort()
    records_sc_39, blacklist_39 = records[:39], records[39:]
    records_sc_153, blacklist_153 = records[:153], records[153:]
    blacklist_double = blacklist_153 + ["SC4131E0-PSG", "SC4362F0-PSG", "SC4522E0-PSG"]

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
    print(df[["f1_macro"]])

