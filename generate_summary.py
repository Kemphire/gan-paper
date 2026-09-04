from typing import Tuple
from pathlib import Path
import pandas as pd
from openpyxl.styles import Font


def get_result_summary(df: pd.DataFrame, regex: str) -> pd.DataFrame:
    df = df.filter(regex=regex, axis="columns")

    summary = (
        df.groupby("model_name")
        .agg(["mean", "std"])
        .stack(level=1, future_stack=True)
        .reset_index()
        .rename(columns={"level_1": "Stats"})
        .round(4)
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

    output_path = pwd / "Summaries-2"
    output_path.mkdir(exist_ok=True)

    results_path = pwd / "CollectedResults"

    results = [
        (pd.read_csv(f), f.stem.title())
        for f in results_path.iterdir()
        if f.suffix != ".txt"
    ]

    accuracy = generate_report(results, regex=r"model_name|_accuracy$")
    f1_score = generate_report(results, regex=r"model_name|_f1_score$")
    precision = generate_report(results, regex=r"model_name|_precision$")
    recall = generate_report(results, regex=r"model_name|_recall$")
    specificity = generate_report(results, regex=r"model_name|_specificity$")
    matthews = generate_report(results, regex=r"model_name|_matthews$")
    auc = generate_report(results, regex=r"model_name|_auc$")

    with pd.ExcelWriter(output_path / "merged.xlsx", engine="openpyxl") as wrt:
        for df, sheet_name in zip(
            [accuracy, f1_score, precision, recall, specificity, matthews, auc],
            [
                "accuracy",
                "f1_score",
                "precision",
                "recall",
                "specificity",
                "matthews",
                "auc",
            ],
        ):
            df.to_excel(wrt, sheet_name=sheet_name, index=False, header=True)

        for ws in wrt.book.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    cell.font = Font(name="Times New Roman", size=10)


if __name__ == "__main__":
    main()
