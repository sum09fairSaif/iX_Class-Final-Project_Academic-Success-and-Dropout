# iX_Class-Final-Project_Academic-Success-and-Dropout

# Academic Success and Dropout Prediction

A data analysis and machine learning project that explores why students drop out of higher education and builds models to predict student outcomes. The work covers exploratory data analysis (EDA), feature engineering, and two classification models, with a focus on identifying at-risk students early enough to support them.

## Overview

The dataset comes from a higher education institution and combines several disjoint sources. Each record describes one student and includes information known at enrollment (academic background, demographics, and social-economic factors) along with academic performance at the end of the first and second semesters. The goal is to predict each student's final outcome, framed as a three-class classification problem: **Dropout**, **Enrolled**, or **Graduate**. The classes are imbalanced, with Graduate being the most common.

The project is built around a single question: can a student's final outcome be predicted early, using a small and understandable set of features that an institution could realistically track?

## Dataset

- **Records:** 4,424 students
- **Features:** 36 original columns, expanded to 170 after one-hot encoding the categorical variables
- **Target:** Dropout, Enrolled, or Graduate (imbalanced toward Graduate)
- **Source:** Publicly available higher-education dataset (commonly found on the UCI Machine Learning Repository and Kaggle). Link: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

## Key Findings from the Exploratory Analysis

**Early academic performance is the clearest signal.** Students who drop out tend to pass very few course units in their first semester, often zero. Admission grade on its own is a weak predictor: students can enter with a strong score and still fail to progress. The number of units a student passes early in the program separates outcomes far more clearly than how they entered.

**A first-semester grade around 13 acts as a useful threshold.** Graduates tend to concentrate at or above this grade, while dropouts fall below it. However, some students with a difficult first semester still go on to graduate, so the threshold is best used as a prompt for a supportive check-in rather than a fixed judgment.

**Financial status is a strong driver of dropout.** Students who were not up to date on tuition fees dropped out at a rate of 86.6%, compared to 24.7% for those who were current. Students carrying debt dropped out at 62%, more than double the 28.3% rate of non-debtors. Scholarship holders showed a much lower dropout rate of 12.2% and a graduation rate of 76%, which suggests that financial relief is an effective retention tool.

**Evening students are a higher-risk group.** Their dropout rate was 42.9%, compared to 30.8% for daytime students. Most evening dropouts were aged 26 or older, which points to working adults balancing employment and family responsibilities rather than younger students struggling academically.

## Modeling and Results

The data was split 80/20 into training and test sets, with stratification to preserve the class balance. Two models were trained and tuned through hyperparameter sweeps:

- **Logistic Regression** — a simple, fast, and interpretable baseline (features scaled with `StandardScaler`).
- **Random Forest** — a stronger model able to capture non-linear patterns and interactions (tuned to `n_estimators=300`, `max_depth=15`).

| Model | Accuracy | Dropout F1 | Enrolled F1 | Graduate F1 |
|---|---|---|---|---|
| Logistic Regression | 76.9% | 0.78 | 0.41 | 0.86 |
| Random Forest | 77.9% | 0.80 | 0.46 | 0.85 |

Both models perform similarly overall, which shows that a simple linear model already captures most of the useful signal. Both are reliable at identifying Dropout and Graduate students but weak at the middle Enrolled class, where recall falls to roughly 0.33–0.36. This happens because Enrolled students overlap heavily with both other groups on almost every feature, leaving the models little that uniquely identifies them. The feature-importance results agree with the exploratory analysis: early approved units, first- and second-semester grades, and tuition status are among the strongest predictors.

## Conclusions

Student outcomes in this dataset are predictable, can be predicted early, and rely on a small and understandable set of academic and financial measures. Because first-semester performance carries most of the predictive weight, a model like this could run at the end of the first semester to flag at-risk students while there is still time to help them. The financial indicators add information that academic features alone do not provide, and they point to concrete actions, such as expanding scholarships or early outreach on missed payments.

## Limitations and Future Work

- Only end-of-semester grades are available, so the model cannot see the decline within a semester that often precedes disengagement. Midterm or assignment-level grades would improve both the analysis and the model.
- The class imbalance toward Graduate raises overall accuracy and is the main cause of the weak Enrolled results. A fairer evaluation would report the macro-average as the primary measure.
- Future work could apply class weighting, resampling, or a two-stage model (first separating Dropout from non-Dropout, then dividing the rest) to improve Enrolled recall.
- The data comes from a single institution, so specific thresholds such as the grade-13 line and the 86.6% figure should be treated as guidance rather than fixed rules until tested on other data.

## Tech Stack

- Python
- pandas — data handling and feature engineering
- scikit-learn — modeling, hyperparameter tuning, and evaluation
- matplotlib and seaborn — visualization

## Repository Contents

- `Academic_Success_and_Dropout_Exploration.ipynb` — the full notebook, covering data understanding, EDA, feature engineering, modeling, and evaluation.

## How to Run

1. Clone the repository.
2. Install the dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
   ```
3. Open the notebook in Jupyter, Databricks, or another notebook environment and run the cells in order.
