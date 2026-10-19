# Student Sleep Dashboard

An interactive dashboard that explores how university students' sleep, screen time and lifestyle habits relate to their academic performance, and predicts a student's CGPA from those habits with a trained machine learning model.

**Course:** CST8504 – Applying AI Techniques, Assignment 1
**Team:** Adam Arrhaoui and Nabil Damer
**Repository:** https://github.com/AdamArrhaoui/StudentSleepDashboard

---

## The problem

Can a student's sleep and daily habits tell us anything about their grades? We treat this as a **regression** problem: predict a student's **cumulative GPA (CGPA, 0–4 scale)** from 22 features, including weekday sleep hours, bedtime, sleep quality, screen time before bed, caffeine, exercise, study hours and class attendance.

| | |
|---|---|
| **Dataset** | [Student Sleep, Screen Time and Academic Performance](https://www.kaggle.com/datasets/aamir28/student-sleep-screen-time-and-academic-performance) (Kaggle) |
| **Size** | 500 students × 25 columns (440 after cleaning) |
| **Target** | `cgpa` |
| **Split** | 70% train / 15% validation / 15% test, fixed random seed |

## Dashboard pages

| Page | What it shows |
|---|---|
| **Overview** | Dataset size, a preview table with column descriptions, summary statistics, the cleaning steps and the CGPA distribution |
| **Explore** | Filters (major, chronotype, residence, year) and four interactive charts: relationships, group comparisons, distributions and correlations |
| **Predict** | A form for a student's habits; the trained model returns a predicted CGPA and shows where it falls among real students |
| **Model Performance** | Comparison of all models on the test set, predicted vs. actual, residuals and feature importance |

## Run with Docker (recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/). From the project folder:

```bash
docker build -t student_sleep_dashboard .
docker run --rm -p 8501:8501 --name dashboard student_sleep_dashboard
```

Then open **http://localhost:8501**. Press `Ctrl+C` to stop.

> If port 8501 is already in use, run with `-p 8502:8501` and open http://localhost:8502.

## Run locally

Requires [Anaconda](https://www.anaconda.com/download) (or Miniconda).

```bash
conda create -n ai_env python=3.10
conda activate ai_env
pip install -r requirements.txt
streamlit run app/dashboard.py
```

Run these from the project root folder (the one containing `app/` and `src/`).

## Re-create the data and model (optional)

The processed data, fitted preprocessor and trained model are already included, so this is only needed after changing the pipeline or the models.

```bash
python src/data_processor.py   # clean, split and scale the raw data; saves data/processed/
python src/model_trainer.py    # train and compare the models; saves models/best_model.pkl
```

## Project structure

```
app/
├── dashboard.py        entry point: page setup and navigation
├── components.py       shared page and section headers
├── data_loading.py     loads and cleans the data for display
├── model_loading.py    loads the trained model and its results
└── views/              one file per dashboard page
src/
├── data_processor.py   cleaning, scaling, encoding and splitting (DataProcessor)
├── model_trainer.py    training and evaluating models (ModelTrainer)
└── predictor.py        predictions for new input (Predictor)
data/
├── raw/                original Kaggle CSV (never edited by hand)
└── processed/          train / validation / test splits + fitted preprocessor
models/                 trained model and evaluation results
.streamlit/config.toml  dashboard theme
Dockerfile              container build instructions
```

## Data processing

1. Dropped `student_id` (an identifier) and `gender` (to avoid a potential source of bias)
2. Removed duplicate rows
3. Dropped rows with missing values (44 of 500 rows; too few to justify imputing)
4. Unwrapped bedtimes after midnight (00:30 → 24.5) so late bedtimes sit next to 23:30 instead of at the opposite end of the scale
5. Removed outliers more than 3 standard deviations from the mean
6. Split into train / validation / test, then fitted the scalers on the **training set only**
7. Scaled numeric features (StandardScaler, MinMaxScaler, or fixed known ranges such as 0–10 ratings) and one-hot encoded categories

## Results

*To be completed once the final evaluation on the test set is saved.*

| Model | Configuration | MAE | RMSE | R² | Training time (s) |
|---|---|---|---|---|---|
| | | | | | |

## Credits

- **Dataset:** [aamir28 on Kaggle](https://www.kaggle.com/datasets/aamir28/student-sleep-screen-time-and-academic-performance)
- **Libraries:** pandas, NumPy, scikit-learn, SciPy, Streamlit, Plotly, Matplotlib, joblib
- **AI assistance:** The dashboard's scaffolding were drafted with help from Claude (Anthropic), then reviewed and restructured by the team. See the comments in each file and the References section of the report.
