# Student Academic Performance Prediction

This course-end project uses ordinary least-squares Linear Regression to estimate a
student's final mathematics grade (`G3`) from the UCI Student Performance dataset.
It demonstrates data acquisition, exploratory analysis, leakage-safe preprocessing,
model comparison, evaluation, persistence, and command-line inference.

## Dataset

The project downloads `student-mat.csv` from the [UCI Student Performance
dataset](https://archive.ics.uci.edu/dataset/320/student+performance). The UCI
repository describes the dataset as student performance data collected from two
Portuguese schools and provides it for research use under its repository terms.
The downloaded CSV is stored in `data/raw/` and is not fabricated or silently
altered. The semicolon-delimited file contains 395 rows and 33 columns.

`G3` is the final mathematics grade on a 0–20 scale. `G1` and `G2` are earlier
period grades. They are intentionally compared in two experiments: including them
usually makes prediction easier, while excluding them better demonstrates whether
other demographic, social, and study-related variables contain predictive signal.
This is observational data: correlations do not establish that a feature causes a
grade change, and results may not generalize to other schools or populations.

## Project structure

```text
data/raw/       Downloaded UCI CSV
models/         Persisted joblib model bundles
notebooks/      Educational end-to-end walkthrough
reports/        JSON metrics and generated figures
src/            Reusable loading, preprocessing, training, evaluation, prediction
tests/          pytest checks for the pipeline contracts
main.py         Train and prediction command-line interface
```

## Setup on Windows

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Git Bash uses `source .venv/Scripts/activate` after creating the same environment.
Python 3.12 is the intended course environment; the code uses portable
`pathlib` paths and also works with the tested interpreter when dependencies
support it.

## Run the project

Train both experiments, the mean baseline, save models, download the dataset, and
generate charts and measured metrics:

```powershell
python main.py train
```

The command writes `reports/metrics.json`, figures under `reports/figures/`, and
three joblib bundles under `models/`. It uses one fixed 80/20 split
(`random_state=42`) for comparable results. Metrics are never hardcoded.

Latest locally generated test-set results (from the command above) are:

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Dummy mean baseline | 3.646 | 4.550 | -0.010 |
| Linear Regression with G1/G2 | 1.647 | 2.378 | 0.724 |
| Linear Regression without G1/G2 | 3.395 | 4.196 | 0.141 |

These values are tied to the current dataset and fixed split; rerunning with a
different dataset version or split can change them.

Run tests:

```powershell
python -m pytest
```

Open and execute `notebooks/student_performance_analysis.ipynb` in Jupyter or VS
Code after running the training command.

## Make a prediction

The default model excludes `G1` and `G2`. Every field required by that model must
be supplied; `G3` is never accepted as an input. For example:

```powershell
python main.py predict --model without_previous_grades --input-json '{"school":"GP","sex":"F","age":17,"address":"U","famsize":"GT3","Pstatus":"A","Medu":4,"Fedu":4,"Mjob":"teacher","Fjob":"health","reason":"course","guardian":"mother","traveltime":1,"studytime":2,"failures":0,"schoolsup":"no","famsup":"yes","paid":"no","activities":"no","nursery":"yes","higher":"yes","internet":"yes","romantic":"no","famrel":4,"freetime":3,"goout":3,"Dalc":1,"Walc":1,"health":5,"absences":4}'
```

Use `--model with_previous_grades` only when adding numeric `G1` and `G2` values.
The result is an estimate, not an official academic assessment or guaranteed
outcome. Invalid or unexpected fields are rejected rather than guessed.

## Methodology and interpretation

The preprocessing pipeline imputes numeric values with training-set medians and
categorical values with the most frequent training value, then one-hot encodes
categorical columns. Ordinary Linear Regression does not require scaling for
correctness, so numeric scaling is intentionally omitted. The `ColumnTransformer`
and model are fitted together only on the training split, preventing test-data
leakage.

The report contains MAE (average absolute grade error), MSE (squared-error
penalty), RMSE (error in grade units with extra outlier sensitivity), and R²
(variance explained relative to a mean baseline). R² alone is not sufficient:
outliers, calibration, and the practical size of grade errors also matter.
Linear assumptions, observational sampling, omitted variables, and the strong
information in previous grades limit conclusions. The comparison is educational,
not a causal or official student assessment.

## License and future work

The project code is MIT licensed. Dataset usage remains subject to the UCI
repository's terms. Future work could add cross-validation, confidence intervals,
fairness checks across subgroups, a stronger regularized baseline, and validation
on a separate school or cohort.
