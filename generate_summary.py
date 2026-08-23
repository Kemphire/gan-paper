from typing import Tuple
from pathlib import Path
import pandas as pd


def get_result_summary(df: pd.DataFrame, regex: str) -> pd.DataFrame:
    df = df.filter(regex=regex, axis="columns")

    summary = (
        df.groupby("model_name")
        .agg(["mean", "std"])
        .stack(level=1, future_stack=True)
        .reset_index()
        .rename(columns={"level_1": "Stats"})
    )

    summary.loc[summary["model_name"].duplicated(), "model_name"] = ""

    return summary


def generate_report(
    results: list[Tuple[pd.DataFrame, str]], regex: str
) -> pd.DataFrame:
    summary = pd.DataFrame()

    for result, name in results:
        single_summary = get_result_summary(result, regex=regex)
        single_summary.insert(0, "Dataset", name)

        summary = pd.concat([summary, single_summary])

        summary.loc[summary["Dataset"].duplicated(), "Dataset"] = ""

    return summary


def main():
    pwd = Path.cwd()

    output_path = pwd / "Summaries"
    output_path.mkdir(exist_ok=True)

    results_path = pwd / "NewResults"

    results = [
        (pd.read_csv(f), f.stem.split("_")[0].title()) for f in results_path.iterdir()
    ]

    generate_report(results, regex=r"model_name|_accuracy$").to_csv(
        output_path / "accuracy.csv",
    )

    generate_report(results, regex=r"model_name|_f1_score$").to_csv(
        output_path / "f1_score.csv",
    )

    generate_report(results, regex=r"model_name|_precision$").to_csv(
        output_path / "precision.csv",
    )

    generate_report(results, regex=r"model_name|_recall$").to_csv(
        output_path / "recall.csv",
    )

    generate_report(results, regex=r"model_name|_specificity$").to_csv(
        output_path / "specificity.csv",
    )

    generate_report(results, regex=r"model_name|_matthews$").to_csv(
        output_path / "matthews.csv",
    )

    generate_report(results, regex=r"model_name|_auc$").to_csv(
        output_path / "auc.csv",
    )


if __name__ == "__main__":
    main()
