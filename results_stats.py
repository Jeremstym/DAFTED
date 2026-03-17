from glob import glob

import click
import pandas as pd
from scipy import stats


def load_results(model_name: str, metric: str) -> pd.DataFrame:
    path = "/data/stympopper/didacticWORKSHOP/" + model_name + "/seed[0-9][0-9]/predictions/test_categorical_scores.csv"
    files = glob(path)
    if not files:
        raise ValueError("No files found")
    results = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file, index_col=0)
        df = df.loc[[metric]]["ht_severity_prediction"].reset_index().rename(columns={"index": "Metric"})
        results = pd.concat([results, df], axis=0)
    results = results.reset_index(drop=True)
    print(results)
    results["ht_severity_prediction"] = pd.to_numeric(results["ht_severity_prediction"], downcast="float")
    print("Mean")
    print(results.groupby("Metric").mean())
    print("Std")
    print(results.groupby("Metric").std())
    return results


@click.command()
@click.option("--model_name", help="Name of the model to load results from")
@click.option("--model_name2", help="Name of the model to load results from")
@click.option("--metric", help="Metric to load")
def ttests(model_name: str, model_name2: str, metric: str) -> pd.DataFrame:
    results1 = load_results(model_name, metric).drop(columns=["Metric"]).values.squeeze(axis=1)
    # print(f"df1 is {load_results(model_name).drop(columns=['Metric'])}")
    # print(f"auroc1 is {results1}")
    results2 = load_results(model_name2, metric).drop(columns=["Metric"]).values.squeeze(axis=1)
    # print(f"df2 is {load_results(model_name2).drop(columns=['Metric'])}")
    # print(f"auroc2 is {results2}")
    ttest = stats.ttest_rel(results1, results2)
    print("------------------------------------------------")
    print("----------------- T-TEST RESULTS ---------------")
    print(f"T-test between {model_name} and {model_name2}")
    print(f"T-statistic: {ttest.statistic}")
    print(f"P-value: {ttest.pvalue}")
    print(f"Mean difference: {100*(results1.mean() - results2.mean())}")
    print(f"ttest: {ttest}")
    return None

@click.command()
@click.option("--model_name", help="Name of the model to load results from")
@click.option("--model_name2", help="Name of the model to load results from")
@click.option("--metric", help="Metric to load")
def shapiro_tests(model_name: str, model_name2: str, metric: str) -> pd.DataFrame:
    results1 = load_results(model_name, metric).drop(columns=["Metric"]).values.squeeze(axis=1)
    results2 = load_results(model_name2, metric).drop(columns=["Metric"]).values.squeeze(axis=1)

    differences = results1 - results2
    shapiro_test = stats.shapiro(differences)
    print("------------------------------------------------")
    print("---------------- SHAPIRO TEST RESULTS ----------")
    print(f"Shapiro test for {model_name}")
    print(f"W-statistic: {shapiro_test.statistic}")
    print(f"P-value: {shapiro_test.pvalue}")
    print(f"shapiro_test: {shapiro_test}")
    return None


if __name__ == "__main__":
    # ttests()
    shapiro_tests()
