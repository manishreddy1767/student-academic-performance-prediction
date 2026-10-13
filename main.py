"""Command-line entry point for training and prediction."""

import argparse
import json
from pathlib import Path

from src.predict import load_json_record, predict
from src.train import run_training


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Student academic performance prediction")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="download data, train models, and generate reports")
    prediction = subparsers.add_parser("predict", help="predict a student's final grade")
    prediction.add_argument(
        "--model",
        choices=["with_previous_grades", "without_previous_grades"],
        default="without_previous_grades",
    )
    prediction.add_argument(
        "--input-json",
        required=True,
        help="JSON object containing the selected model's feature fields",
    )
    args = parser.parse_args()
    if args.command == "train":
        metrics = run_training(project_dir)
        print(json.dumps(metrics, indent=2))
        return
    try:
        grade = predict(project_dir, load_json_record(args.input_json), args.model)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"Estimated final grade (G3): {grade:.2f}/20")
    print("This is an estimate, not an official academic assessment.")


if __name__ == "__main__":
    main()
