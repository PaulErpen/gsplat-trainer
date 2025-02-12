import argparse
from gsplat_trainer.eval.eval_handler import EvalHandler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataDir",
        "-d",
        type=str,
        help="The path where the data is stored",
        required=True,
    )

    parser.add_argument(
        "--singleDfPathXl",
        "-s",
        type=str,
        help="The path where the data for the single results dataframe will be stored in excel format",
        required=True,
    )

    parsed_args = parser.parse_args()

    EvalHandler(parsed_args.dataDir, "cuda", 5).compute_metrics_dataframe().to_excel(
        parsed_args.singleDfPathXl
    )
