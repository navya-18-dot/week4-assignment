"""
Task 4: Personal Data Science Project

Problem statement
-----------------
People lose time for different reasons: social media, poor sleep, phone
distractions, money leaks, commute delays, and inconsistent fitness routines.
This project focuses on a strong personal-data-science theme:

"Can I predict my productivity from my daily habits?"

It also creates an interactive dashboard where the user can choose the issue
that frustrates them most and get a simple prediction plus recommendations.

Because this is a one-week personal project and no private user data is
available, the script creates a realistic synthetic survey dataset. In a real
project, the same columns could be collected using a Google Form or daily log.

What the script does
--------------------
1. Creates a synthetic productivity habit dataset.
2. Creates a synthetic social media overuse dataset.
3. Trains Linear Regression and Decision Tree models for productivity.
4. Trains a logistic regression model for social media procrastination risk.
5. Generates reports, charts, and a standalone interactive dashboard.
"""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


RANDOM_SEED = 42
REPORT_DIR = Path("reports")
DATA_PATH = REPORT_DIR / "social_media_overuse_survey.csv"
REPORT_PATH = REPORT_DIR / "social_media_overuse_report.txt"
PRODUCTIVITY_DATA_PATH = REPORT_DIR / "daily_productivity_habits.csv"
PRODUCTIVITY_REPORT_PATH = REPORT_DIR / "daily_productivity_report.txt"
DASHBOARD_PATH = REPORT_DIR / "personal_frustration_dashboard.html"


def sigmoid(value):
    """Convert a raw score into a probability between 0 and 1."""
    return 1 / (1 + np.exp(-value))


class SimpleDecisionTreeRegressor:
    """
    A small CART-style regression tree.

    This keeps the project runnable without scikit-learn while still showing
    the idea of a decision tree model for the assignment.
    """

    def __init__(self, max_depth=3, min_samples_leaf=8):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, features, target):
        self.feature_names = list(features.columns)
        x_values = features.to_numpy(dtype=float)
        y_values = np.asarray(target, dtype=float)
        self.tree = self._build_tree(x_values, y_values, depth=0)
        return self

    def predict(self, features):
        x_values = features.to_numpy(dtype=float)
        return np.array([self._predict_row(row, self.tree) for row in x_values])

    def _build_tree(self, x_values, y_values, depth):
        node = {"prediction": float(np.mean(y_values))}
        if depth >= self.max_depth or len(y_values) < self.min_samples_leaf * 2:
            return node

        best_feature = None
        best_threshold = None
        best_error = float("inf")

        for feature_index in range(x_values.shape[1]):
            thresholds = np.percentile(x_values[:, feature_index], [25, 50, 75])
            for threshold in np.unique(thresholds):
                left_mask = x_values[:, feature_index] <= threshold
                right_mask = ~left_mask
                if (
                    left_mask.sum() < self.min_samples_leaf
                    or right_mask.sum() < self.min_samples_leaf
                ):
                    continue
                left_error = np.var(y_values[left_mask]) * left_mask.sum()
                right_error = np.var(y_values[right_mask]) * right_mask.sum()
                total_error = left_error + right_error
                if total_error < best_error:
                    best_error = total_error
                    best_feature = feature_index
                    best_threshold = float(threshold)

        if best_feature is None:
            return node

        left_mask = x_values[:, best_feature] <= best_threshold
        node.update(
            {
                "feature_index": best_feature,
                "feature_name": self.feature_names[best_feature],
                "threshold": best_threshold,
                "left": self._build_tree(
                    x_values[left_mask],
                    y_values[left_mask],
                    depth + 1,
                ),
                "right": self._build_tree(
                    x_values[~left_mask],
                    y_values[~left_mask],
                    depth + 1,
                ),
            }
        )
        return node

    def _predict_row(self, row, node):
        if "feature_index" not in node:
            return node["prediction"]
        if row[node["feature_index"]] <= node["threshold"]:
            return self._predict_row(row, node["left"])
        return self._predict_row(row, node["right"])

    def describe(self, node=None, depth=0):
        """Return readable rules from the trained tree."""
        if node is None:
            node = self.tree
        indent = "  " * depth
        if "feature_index" not in node:
            return [f"{indent}Predict productivity score {node['prediction']:.1f}"]

        feature = node["feature_name"]
        threshold = node["threshold"]
        lines = [f"{indent}If {feature} <= {threshold:.1f}:"]
        lines.extend(self.describe(node["left"], depth + 1))
        lines.append(f"{indent}else:")
        lines.extend(self.describe(node["right"], depth + 1))
        return lines


def create_productivity_dataset(days=120):
    """
    Create a daily habit dataset for productivity prediction.

    In real use, replace this with your own 7-14 days of tracking. A larger
    synthetic dataset is used here so the model has enough rows to learn from.
    """
    rng = np.random.default_rng(RANDOM_SEED + 10)

    sleep_hours = np.clip(rng.normal(7.0, 1.2, days), 3.5, 10.0).round(1)
    wake_up_hour = np.clip(rng.normal(7.3, 1.4, days), 4.5, 11.5).round(2)
    screen_time_hours = np.clip(rng.normal(5.5, 1.8, days), 1.0, 11.0).round(1)
    social_media_minutes = np.clip(
        screen_time_hours * rng.uniform(18, 42, days) + rng.normal(0, 25, days),
        5,
        360,
    ).round()
    coffee_cups = np.clip(rng.poisson(1.4, days), 0, 5)
    exercise = rng.binomial(1, 0.45, days)
    mood_score = np.clip(
        rng.normal(6.4, 1.5, days) + 0.28 * exercise + 0.18 * (sleep_hours - 7),
        1,
        10,
    ).round(1)
    focused_work_hours = np.clip(
        rng.normal(3.8, 1.4, days)
        + 0.45 * exercise
        + 0.35 * (sleep_hours - 7)
        - 0.012 * np.maximum(social_media_minutes - 90, 0),
        0.5,
        8.5,
    ).round(1)

    productivity_score = (
        2.0
        + 0.62 * sleep_hours
        - 0.33 * np.maximum(6.5 - sleep_hours, 0)
        - 0.27 * np.maximum(wake_up_hour - 8.0, 0)
        - 0.20 * screen_time_hours
        - 0.010 * social_media_minutes
        + 0.23 * coffee_cups
        + 0.75 * exercise
        + 0.30 * mood_score
        + 0.42 * focused_work_hours
        + rng.normal(0, 0.75, days)
    )
    productivity_score = np.clip(productivity_score, 1, 10).round(1)

    return pd.DataFrame(
        {
            "sleep_hours": sleep_hours,
            "wake_up_hour": wake_up_hour,
            "screen_time_hours": screen_time_hours,
            "social_media_minutes": social_media_minutes.astype(int),
            "coffee_cups": coffee_cups.astype(int),
            "exercise": exercise.astype(int),
            "mood_score": mood_score,
            "focused_work_hours": focused_work_hours,
            "productivity_score": productivity_score,
        }
    )


def train_productivity_models(data):
    """Train Linear Regression and Decision Tree models for productivity."""
    feature_columns = [
        "sleep_hours",
        "wake_up_hour",
        "screen_time_hours",
        "social_media_minutes",
        "coffee_cups",
        "exercise",
        "mood_score",
        "focused_work_hours",
    ]
    features = data[feature_columns]
    target = data["productivity_score"]

    linear_features = sm.add_constant(features)
    linear_model = sm.OLS(target, linear_features).fit()
    tree_model = SimpleDecisionTreeRegressor(max_depth=3, min_samples_leaf=10).fit(
        features,
        target,
    )

    scored_data = data.copy()
    scored_data["linear_prediction"] = linear_model.predict(linear_features)
    scored_data["tree_prediction"] = tree_model.predict(features)

    return linear_model, tree_model, scored_data, feature_columns


def regression_metrics(actual, predicted):
    """Calculate simple regression metrics."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    total_error = np.sum((actual - actual.mean()) ** 2)
    unexplained_error = np.sum((actual - predicted) ** 2)
    r_squared = float(1 - unexplained_error / total_error)
    return {"mae": mae, "rmse": rmse, "r_squared": r_squared}


def productivity_level(score):
    if score >= 7.5:
        return "High"
    if score >= 5.5:
        return "Moderate"
    return "Low"


def predict_productivity(linear_model, tree_model, feature_columns, profile):
    row = pd.DataFrame([profile], columns=feature_columns)
    linear_row = sm.add_constant(row, has_constant="add")
    linear_prediction = float(linear_model.predict(linear_row).iloc[0])
    tree_prediction = float(tree_model.predict(row)[0])
    blended_prediction = np.clip((linear_prediction + tree_prediction) / 2, 1, 10)
    return blended_prediction, productivity_level(blended_prediction)


def build_productivity_recommendations(profile, predicted_score):
    recommendations = []

    if profile["sleep_hours"] < 6:
        recommendations.append("Increase sleep toward 7 hours before optimizing anything else.")
    if profile["wake_up_hour"] > 8:
        recommendations.append("Try waking 30 minutes earlier for one week and compare focus.")
    if profile["social_media_minutes"] > 90:
        recommendations.append("Limit social media to under 90 minutes, with no scrolling before study/work.")
    if profile["screen_time_hours"] > 6:
        recommendations.append("Create two phone-free focus blocks during your highest-energy hours.")
    if profile["exercise"] == 0:
        recommendations.append("Add a 15-20 minute walk or workout; movement often improves mood and output.")
    if profile["mood_score"] < 5:
        recommendations.append("Track mood triggers; low mood is a signal to simplify the task list.")
    if profile["focused_work_hours"] < 3:
        recommendations.append("Use a timer for two deep-work sessions instead of trying to work all day.")
    if predicted_score < 5.5:
        recommendations.append("Plan tomorrow around one important task, not a long unrealistic checklist.")

    if not recommendations:
        recommendations.append("Your routine looks balanced; keep protecting sleep, focus time, and exercise.")

    return recommendations


def make_productivity_visualizations(data):
    """Save productivity charts."""
    REPORT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    habit_columns = [
        "sleep_hours",
        "wake_up_hour",
        "screen_time_hours",
        "social_media_minutes",
        "coffee_cups",
        "exercise",
        "mood_score",
        "focused_work_hours",
        "productivity_score",
    ]

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        data[habit_columns].corr(),
        annot=True,
        fmt=".2f",
        cmap="vlag",
        linewidths=0.5,
    )
    plt.title("Correlation Heatmap: Daily Habits vs Productivity")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "productivity_correlation_heatmap.png", dpi=150)
    plt.close()

    correlations = (
        data[habit_columns].corr(numeric_only=True)["productivity_score"]
        .drop("productivity_score")
        .sort_values()
    )
    plt.figure(figsize=(8, 5))
    colors = ["#d1495b" if value < 0 else "#2a9d8f" for value in correlations]
    correlations.plot(kind="barh", color=colors)
    plt.title("Which Habits Affect Productivity Most?")
    plt.xlabel("Correlation with productivity score")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "productivity_factor_impact.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=data,
        x="productivity_score",
        y="linear_prediction",
        color="#457b9d",
        alpha=0.8,
    )
    plt.plot([1, 10], [1, 10], color="#e76f51", linestyle="--")
    plt.title("Linear Regression: Actual vs Predicted Productivity")
    plt.xlabel("Actual productivity score")
    plt.ylabel("Predicted productivity score")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "productivity_actual_vs_predicted.png", dpi=150)
    plt.close()


def write_productivity_report(data, linear_model, tree_model, metrics, feature_columns):
    """Write the main productivity project report."""
    coefficient_table = pd.DataFrame(
        {
            "feature": linear_model.params.index,
            "coefficient": linear_model.params.values,
            "p_value": linear_model.pvalues.values,
        }
    ).sort_values("coefficient", ascending=False)

    example_profile = {
        "sleep_hours": 5.8,
        "wake_up_hour": 8.7,
        "screen_time_hours": 7.2,
        "social_media_minutes": 130,
        "coffee_cups": 2,
        "exercise": 0,
        "mood_score": 5,
        "focused_work_hours": 2.4,
    }
    prediction, level = predict_productivity(
        linear_model,
        tree_model,
        feature_columns,
        example_profile,
    )
    recommendations = build_productivity_recommendations(example_profile, prediction)

    strongest_positive = (
        data[feature_columns + ["productivity_score"]]
        .corr(numeric_only=True)["productivity_score"]
        .drop("productivity_score")
        .sort_values(ascending=False)
        .head(3)
    )
    strongest_negative = (
        data[feature_columns + ["productivity_score"]]
        .corr(numeric_only=True)["productivity_score"]
        .drop("productivity_score")
        .sort_values()
        .head(3)
    )

    report = f"""
PERSONAL DATA SCIENCE PROJECT: DAILY PRODUCTIVITY PREDICTOR

Problem Statement
-----------------
Why are some days productive while others feel wasted? This project predicts a
daily productivity score from habits such as sleep, wake-up time, screen time,
social media use, coffee intake, exercise, mood, and focused work hours.

Dataset
-------
Rows: {len(data)}
Type: Synthetic daily habit log, designed like a 7-14 day personal tracking sheet.
Target variable: productivity_score, from 1 to 10.

Models
------
1. Linear Regression with statsmodels
   MAE: {metrics["linear"]["mae"]:.2f}
   RMSE: {metrics["linear"]["rmse"]:.2f}
   R-squared: {metrics["linear"]["r_squared"]:.2f}

2. Decision Tree Regressor, implemented in this file
   MAE: {metrics["tree"]["mae"]:.2f}
   RMSE: {metrics["tree"]["rmse"]:.2f}
   R-squared: {metrics["tree"]["r_squared"]:.2f}

Main Insights
-------------
Strongest positive factors:
{strongest_positive.to_string()}

Strongest negative factors:
{strongest_negative.to_string()}

Example Prediction
------------------
Example user profile:
{example_profile}

Predicted productivity score: {prediction:.1f}/10
Productivity level: {level}

Recommendations
---------------
{chr(10).join("- " + item for item in recommendations)}

Decision Tree Rules
-------------------
{chr(10).join(tree_model.describe())}

Linear Regression Coefficients
------------------------------
Positive coefficients increase productivity.
Negative coefficients reduce productivity.

{coefficient_table.to_string(index=False)}

Final Suggestion
----------------
The best first experiment is simple: sleep at least 7 hours, keep social media
below 90 minutes, and protect two focused work blocks. Track your score for
one week and compare the result with your old routine.
"""

    PRODUCTIVITY_REPORT_PATH.write_text(
        textwrap.dedent(report).strip(),
        encoding="utf-8",
    )
    return PRODUCTIVITY_REPORT_PATH


def create_synthetic_dataset(rows=500):
    """
    Create survey-style data.

    Each row represents one person's daily social media pattern. The target
    column, high_procrastination_risk, is generated from behavior that is
    commonly connected with procrastination: more screen time, more short-video
    feed exposure, more notification interruptions, late-night usage, and lower
    self-control.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    daily_minutes = np.clip(rng.normal(165, 75, rows), 20, 480).round()
    unnecessary_feed_percent = np.clip(rng.normal(52, 21, rows), 5, 98).round()
    short_video_minutes = np.clip(
        daily_minutes * rng.uniform(0.15, 0.65, rows) + rng.normal(0, 20, rows),
        0,
        300,
    ).round()
    notifications_per_day = np.clip(rng.normal(45, 28, rows), 0, 180).round()
    late_night_usage_minutes = np.clip(rng.normal(35, 32, rows), 0, 180).round()
    study_or_work_hours = np.clip(rng.normal(5.5, 1.8, rows), 1, 11).round(1)
    self_control_score = rng.integers(1, 11, rows)
    purposeful_use_percent = np.clip(
        100 - unnecessary_feed_percent + rng.normal(0, 10, rows),
        0,
        100,
    ).round()

    raw_risk_score = (
        -3.2
        + 0.010 * daily_minutes
        + 0.030 * unnecessary_feed_percent
        + 0.008 * short_video_minutes
        + 0.012 * notifications_per_day
        + 0.018 * late_night_usage_minutes
        - 0.170 * study_or_work_hours
        - 0.260 * self_control_score
        - 0.012 * purposeful_use_percent
        + rng.normal(0, 0.85, rows)
    )

    risk_probability = sigmoid(raw_risk_score)
    high_procrastination_risk = rng.binomial(1, risk_probability)

    data = pd.DataFrame(
        {
            "daily_minutes": daily_minutes.astype(int),
            "unnecessary_feed_percent": unnecessary_feed_percent.astype(int),
            "short_video_minutes": short_video_minutes.astype(int),
            "notifications_per_day": notifications_per_day.astype(int),
            "late_night_usage_minutes": late_night_usage_minutes.astype(int),
            "study_or_work_hours": study_or_work_hours,
            "self_control_score": self_control_score,
            "purposeful_use_percent": purposeful_use_percent.astype(int),
            "high_procrastination_risk": high_procrastination_risk,
        }
    )

    return data


def train_logistic_model(data):
    """Train an interpretable logistic regression model."""
    feature_columns = [
        "daily_minutes",
        "unnecessary_feed_percent",
        "short_video_minutes",
        "notifications_per_day",
        "late_night_usage_minutes",
        "study_or_work_hours",
        "self_control_score",
        "purposeful_use_percent",
    ]

    features = sm.add_constant(data[feature_columns])
    target = data["high_procrastination_risk"]
    model = sm.Logit(target, features).fit(disp=False)

    data = data.copy()
    data["predicted_risk_probability"] = model.predict(features)
    data["predicted_high_risk"] = (
        data["predicted_risk_probability"] >= 0.50
    ).astype(int)

    return model, data, feature_columns


def calculate_metrics(data):
    """Calculate simple classification metrics without extra dependencies."""
    actual = data["high_procrastination_risk"]
    predicted = data["predicted_high_risk"]

    true_positive = int(((actual == 1) & (predicted == 1)).sum())
    true_negative = int(((actual == 0) & (predicted == 0)).sum())
    false_positive = int(((actual == 0) & (predicted == 1)).sum())
    false_negative = int(((actual == 1) & (predicted == 0)).sum())

    accuracy = (true_positive + true_negative) / len(data)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def make_visualizations(data):
    """Save charts that explain behavior patterns."""
    REPORT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=data,
        x="daily_minutes",
        hue="high_procrastination_risk",
        bins=25,
        kde=True,
        palette=["#2a9d8f", "#e76f51"],
    )
    plt.title("Daily Social Media Time vs Procrastination Risk")
    plt.xlabel("Daily social media usage in minutes")
    plt.ylabel("Number of people")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "daily_usage_risk_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=data,
        x="unnecessary_feed_percent",
        y="predicted_risk_probability",
        hue="late_night_usage_minutes",
        palette="viridis",
        alpha=0.75,
    )
    plt.title("Unnecessary Feed Exposure Increases Predicted Risk")
    plt.xlabel("Unnecessary feed percentage")
    plt.ylabel("Predicted high-risk probability")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feed_exposure_predicted_risk.png", dpi=150)
    plt.close()

    correlation_columns = [
        "daily_minutes",
        "unnecessary_feed_percent",
        "short_video_minutes",
        "notifications_per_day",
        "late_night_usage_minutes",
        "study_or_work_hours",
        "self_control_score",
        "purposeful_use_percent",
        "high_procrastination_risk",
    ]
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        data[correlation_columns].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
    )
    plt.title("Correlation Heatmap of Social Media Habits")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def risk_level(probability):
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Moderate"
    return "Low"


def predict_personal_risk(model, feature_columns, user_profile):
    """Predict risk for one user profile dictionary."""
    row = pd.DataFrame([user_profile], columns=feature_columns)
    row = sm.add_constant(row, has_constant="add")
    probability = float(model.predict(row).iloc[0])
    return probability, risk_level(probability)


def build_recommendations(profile, probability):
    """Create practical recommendations from the user's risk factors."""
    recommendations = []

    if profile["daily_minutes"] > 150:
        recommendations.append(
            "Set a daily app timer near 120 minutes and reduce it gradually."
        )
    if profile["unnecessary_feed_percent"] > 45:
        recommendations.append(
            "Unfollow low-value pages and use the feed only after a planned task."
        )
    if profile["notifications_per_day"] > 35:
        recommendations.append(
            "Turn off non-essential notifications; keep only calls, messages, and study/work alerts."
        )
    if profile["late_night_usage_minutes"] > 25:
        recommendations.append(
            "Keep the phone away from bed and set a no-feed rule 45 minutes before sleep."
        )
    if profile["short_video_minutes"] > 60:
        recommendations.append(
            "Replace short-video scrolling with a fixed 10-minute break timer."
        )
    if profile["self_control_score"] <= 5:
        recommendations.append(
            "Use app blockers during deep-work blocks because willpower alone is unreliable."
        )
    if probability >= 0.70:
        recommendations.append(
            "Try a 7-day experiment: open social apps only at two fixed times per day."
        )

    if not recommendations:
        recommendations.append(
            "Current habits look balanced; keep social media purposeful and time-boxed."
        )

    return recommendations


def write_report(data, model, metrics, feature_columns):
    """Write a plain-English report for submission."""
    coefficient_table = pd.DataFrame(
        {
            "feature": model.params.index,
            "coefficient": model.params.values,
            "odds_ratio": np.exp(model.params.values),
            "p_value": model.pvalues.values,
        }
    ).sort_values("odds_ratio", ascending=False)

    high_risk_group = data[data["high_procrastination_risk"] == 1]
    low_risk_group = data[data["high_procrastination_risk"] == 0]

    example_profile = {
        "daily_minutes": 210,
        "unnecessary_feed_percent": 65,
        "short_video_minutes": 95,
        "notifications_per_day": 70,
        "late_night_usage_minutes": 55,
        "study_or_work_hours": 4.5,
        "self_control_score": 4,
        "purposeful_use_percent": 25,
    }
    probability, level = predict_personal_risk(model, feature_columns, example_profile)
    recommendations = build_recommendations(example_profile, probability)

    pros = [
        "Social media can help people learn quickly from tutorials, news, and expert creators.",
        "It supports connection with friends, communities, classmates, and professional networks.",
        "It can provide entertainment and short breaks when usage is planned.",
        "It helps creators, small businesses, and students share work with a wider audience.",
    ]

    cons = [
        "Unnecessary feed exposure increases mindless scrolling and time loss.",
        "Short-video loops can reduce attention span and make deep work feel boring.",
        "Frequent notifications interrupt study or work flow.",
        "Late-night usage can disturb sleep, which further increases procrastination.",
        "Comparison-heavy content can affect mood, confidence, and motivation.",
    ]

    pros_text = "\n".join(f"- {item}" for item in pros)
    cons_text = "\n".join(f"- {item}" for item in cons)
    recommendations_text = "\n".join(f"- {item}" for item in recommendations)
    coefficients_text = coefficient_table.to_string(index=False)

    report = f"""
PERSONAL DATA SCIENCE PROJECT: SOCIAL MEDIA OVERUSE RISK MODEL

Problem Statement
-----------------
Many people feel disgusted with unnecessary social media feed because it
wastes time and encourages procrastination. The goal of this project is to
predict whether a person's social media habits create a high
procrastination risk and to suggest practical improvements.

Dataset
-------
Rows: {len(data)}
Type: Synthetic survey-style dataset
Target variable: high_procrastination_risk

Important columns:
- daily_minutes: Total social media use per day.
- unnecessary_feed_percent: Estimated percentage of content that feels useless.
- short_video_minutes: Time spent on reels/shorts style content.
- notifications_per_day: Number of app interruptions.
- late_night_usage_minutes: Social media use close to bedtime.
- study_or_work_hours: Productive hours per day.
- self_control_score: Self-rated discipline from 1 to 10.
- purposeful_use_percent: Percentage of usage for learning, communication, or work.

Model
-----
Algorithm: Logistic Regression using statsmodels
Accuracy: {metrics["accuracy"]:.2%}
Precision: {metrics["precision"]:.2%}
Recall: {metrics["recall"]:.2%}

Confusion Matrix
----------------
True Positive: {metrics["true_positive"]}
True Negative: {metrics["true_negative"]}
False Positive: {metrics["false_positive"]}
False Negative: {metrics["false_negative"]}

Main Insights
-------------
Average daily minutes for high-risk users: {high_risk_group["daily_minutes"].mean():.1f}
Average daily minutes for low-risk users: {low_risk_group["daily_minutes"].mean():.1f}
Average unnecessary feed percentage for high-risk users: {high_risk_group["unnecessary_feed_percent"].mean():.1f}
Average unnecessary feed percentage for low-risk users: {low_risk_group["unnecessary_feed_percent"].mean():.1f}
Average late-night usage for high-risk users: {high_risk_group["late_night_usage_minutes"].mean():.1f}
Average late-night usage for low-risk users: {low_risk_group["late_night_usage_minutes"].mean():.1f}

Pros of Social Media
--------------------
{pros_text}

Cons / Negative Impacts
-----------------------
{cons_text}

Example Prediction
------------------
Example user profile:
{example_profile}

Predicted procrastination risk: {probability:.2%}
Risk level: {level}

Recommendations
---------------
{recommendations_text}

Model Coefficients
------------------
Positive coefficients increase procrastination risk.
Negative coefficients reduce procrastination risk.

{coefficients_text}

Final Suggestion
----------------
Social media does not need to be completely removed. The better solution is
controlled, purposeful usage: reduce unnecessary feed, silence distracting
notifications, avoid late-night scrolling, and use social apps only during
fixed time windows.
"""

    REPORT_PATH.write_text(textwrap.dedent(report).strip(), encoding="utf-8")

    return REPORT_PATH


def generate_interactive_dashboard():
    """Create a standalone dashboard that asks the user about their frustration."""
    dashboard = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Personal Frustration Data Science Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1d2630;
      --muted: #667085;
      --line: #d7dde5;
      --paper: #f7f8fb;
      --panel: #ffffff;
      --teal: #147d7e;
      --coral: #d85c4a;
      --blue: #386fa4;
      --gold: #b7791f;
      --green: #2f855a;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--paper);
    }

    header {
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 22px 28px 18px;
    }

    header h1 {
      margin: 0 0 6px;
      font-size: clamp(24px, 3vw, 34px);
      letter-spacing: 0;
    }

    header p {
      margin: 0;
      color: var(--muted);
      max-width: 920px;
      line-height: 1.45;
    }

    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 22px;
    }

    .top-grid {
      display: grid;
      grid-template-columns: minmax(260px, 0.9fr) minmax(320px, 1.1fr);
      gap: 18px;
      align-items: start;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
    }

    .panel h2 {
      margin: 0 0 12px;
      font-size: 19px;
    }

    label {
      display: block;
      margin: 13px 0 6px;
      color: #344054;
      font-weight: 700;
      font-size: 14px;
    }

    select,
    input[type="number"],
    input[type="range"] {
      width: 100%;
    }

    select,
    input[type="number"] {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 11px;
      background: white;
      color: var(--ink);
      font-size: 15px;
    }

    input[type="range"] {
      accent-color: var(--teal);
    }

    .field-row {
      display: grid;
      grid-template-columns: 1fr 86px;
      gap: 12px;
      align-items: end;
      padding: 8px 0;
      border-bottom: 1px solid #edf0f4;
    }

    .field-row:last-child {
      border-bottom: 0;
    }

    .value-pill {
      display: inline-block;
      min-width: 58px;
      padding: 4px 8px;
      border-radius: 999px;
      background: #eef6f6;
      color: var(--teal);
      text-align: center;
      font-weight: 700;
      font-size: 13px;
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }

    button {
      border: 1px solid var(--teal);
      border-radius: 6px;
      padding: 10px 13px;
      background: var(--teal);
      color: white;
      font-weight: 700;
      cursor: pointer;
    }

    button.secondary {
      background: white;
      color: var(--teal);
    }

    .score-card {
      display: grid;
      grid-template-columns: 140px 1fr;
      gap: 16px;
      align-items: center;
    }

    .gauge {
      width: 132px;
      height: 132px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      background: conic-gradient(var(--teal) 0deg, #e6eaef 0deg);
      position: relative;
    }

    .gauge::after {
      content: "";
      position: absolute;
      width: 96px;
      height: 96px;
      border-radius: 50%;
      background: white;
    }

    .gauge span {
      position: relative;
      z-index: 1;
      font-size: 25px;
      font-weight: 800;
    }

    .tag {
      display: inline-block;
      padding: 5px 9px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 13px;
      background: #eef2ff;
      color: var(--blue);
    }

    .result-title {
      font-size: 22px;
      font-weight: 800;
      margin: 4px 0 8px;
    }

    .muted {
      color: var(--muted);
      line-height: 1.45;
    }

    .list-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(240px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }

    ul {
      margin: 8px 0 0;
      padding-left: 20px;
      line-height: 1.5;
    }

    .cards {
      display: grid;
      grid-template-columns: repeat(3, minmax(210px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }

    .issue-card {
      background: white;
      border: 1px solid var(--line);
      border-left: 5px solid var(--blue);
      border-radius: 8px;
      padding: 15px;
      cursor: pointer;
      min-height: 132px;
    }

    .issue-card.active {
      border-left-color: var(--coral);
      box-shadow: 0 0 0 2px rgba(216, 92, 74, 0.15);
    }

    .issue-card h3 {
      margin: 0 0 6px;
      font-size: 16px;
    }

    .issue-card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.4;
      font-size: 14px;
    }

    canvas {
      width: 100%;
      min-height: 220px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
    }

    .footer-note {
      margin: 18px 0 0;
      color: var(--muted);
      font-size: 13px;
    }

    @media (max-width: 860px) {
      main {
        padding: 14px;
      }

      .top-grid,
      .list-grid,
      .cards,
      .score-card {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Personal Frustration Data Science Dashboard</h1>
    <p>
      Choose the daily-life issue that frustrates you most, enter your current
      habits, and get a simple prediction with practical recommendations.
    </p>
  </header>

  <main>
    <section class="top-grid">
      <div class="panel">
        <h2>What frustrates you most?</h2>
        <label for="issueSelect">Select your problem area</label>
        <select id="issueSelect"></select>
        <div id="inputFields"></div>
        <div class="actions">
          <button id="predictButton" type="button">Predict and Recommend</button>
          <button id="sampleButton" class="secondary" type="button">Use Sample Values</button>
          <button id="csvButton" class="secondary" type="button">Download CSV Template</button>
        </div>
      </div>

      <div class="panel">
        <h2>Prediction</h2>
        <div class="score-card">
          <div class="gauge" id="gauge"><span id="scoreText">0</span></div>
          <div>
            <span class="tag" id="levelText">Waiting</span>
            <div class="result-title" id="resultTitle">Enter your details</div>
            <p class="muted" id="resultSummary">
              The dashboard will estimate your risk or performance level and show the strongest improvement areas.
            </p>
          </div>
        </div>
        <canvas id="factorChart" width="700" height="260" aria-label="Factor impact chart"></canvas>
      </div>
    </section>

    <section class="list-grid">
      <div class="panel">
        <h2>Pros</h2>
        <ul id="prosList"></ul>
      </div>
      <div class="panel">
        <h2>Cons / Impacts</h2>
        <ul id="consList"></ul>
      </div>
    </section>

    <section class="panel" style="margin-top: 18px;">
      <h2>Recommendations</h2>
      <ul id="recommendationList"></ul>
      <p class="footer-note">
        Tip for the assignment: replace sample values with your own 7-14 day tracking data, then compare before-and-after habits.
      </p>
    </section>

    <section class="cards" id="issueCards"></section>
  </main>

  <script>
    const issues = {
      productivity: {
        title: "Daily Productivity Predictor",
        short: "Why are some days productive and others wasted?",
        unit: "/10",
        mode: "score",
        fields: [
          ["sleepHours", "Sleep hours", 3, 10, 0.1, 6.2],
          ["wakeUpHour", "Wake-up time, 24h", 4, 12, 0.1, 8.4],
          ["screenTime", "Total screen time, hours", 1, 12, 0.1, 7],
          ["socialMedia", "Social media minutes", 0, 360, 5, 125],
          ["coffee", "Coffee cups", 0, 6, 1, 2],
          ["exercise", "Exercise, 0 no / 1 yes", 0, 1, 1, 0],
          ["mood", "Mood score", 1, 10, 1, 5],
          ["focusHours", "Focused work/study hours", 0, 9, 0.1, 2.5]
        ],
        pros: [
          "Turns vague bad days into measurable habit patterns.",
          "Shows which variables are most worth changing first.",
          "Easy to collect with a 7-14 day spreadsheet."
        ],
        cons: [
          "Short tracking periods can be noisy.",
          "Mood and productivity scores are subjective.",
          "Unexpected events can affect results."
        ],
        calculate(v) {
          let score = 2.1 + 0.7 * v.sleepHours - 0.35 * Math.max(6 - v.sleepHours, 0)
            - 0.25 * Math.max(v.wakeUpHour - 8, 0) - 0.22 * v.screenTime
            - 0.012 * v.socialMedia + 0.2 * v.coffee + 0.8 * v.exercise
            + 0.32 * v.mood + 0.48 * v.focusHours;
          score = clamp(score, 1, 10);
          const recs = [];
          if (v.sleepHours < 6) recs.push("Sleep is below 6 hours; aim for 7 hours before trying complex productivity hacks.");
          if (v.socialMedia > 90) recs.push("Keep social media under 90 minutes and avoid it before study/work.");
          if (v.exercise === 0) recs.push("Add a short walk or workout; exercise improves both mood and focus.");
          if (v.focusHours < 3) recs.push("Schedule two focused work blocks instead of relying on all-day motivation.");
          if (v.screenTime > 6) recs.push("Create two phone-free periods during your highest-energy hours.");
          return result(score, "Predicted productivity score", recs, [
            ["Sleep", v.sleepHours / 10],
            ["Focused work", v.focusHours / 9],
            ["Mood", v.mood / 10],
            ["Low social media", 1 - v.socialMedia / 360],
            ["Exercise", v.exercise]
          ]);
        }
      },
      phone: {
        title: "Phone Usage vs Study/Work Performance",
        short: "Is my phone actually ruining my focus?",
        unit: "%",
        mode: "score",
        fields: [
          ["screenTime", "Total screen time, hours", 1, 12, 0.1, 7.5],
          ["instagram", "Instagram minutes", 0, 240, 5, 95],
          ["youtube", "YouTube minutes", 0, 240, 5, 80],
          ["whatsapp", "WhatsApp minutes", 0, 180, 5, 55],
          ["studyHours", "Study/work hours", 0, 10, 0.1, 3],
          ["taskCompletion", "Task completion percent", 0, 100, 5, 55],
          ["distractions", "Distraction count", 0, 80, 1, 35]
        ],
        pros: [
          "Makes phone distraction visible instead of emotional.",
          "App-wise tracking gives specific behavior changes.",
          "Very relatable for students and workers."
        ],
        cons: [
          "Some phone time can still be useful.",
          "Performance also depends on task difficulty.",
          "Distraction counts require honest tracking."
        ],
        calculate(v) {
          let score = 55 + 5 * v.studyHours + 0.3 * v.taskCompletion
            - 2.5 * v.screenTime - 0.08 * v.instagram - 0.04 * v.youtube
            - 0.02 * v.whatsapp - 0.7 * v.distractions;
          score = clamp(score, 0, 100);
          const recs = [];
          if (v.instagram > 60) recs.push("Move Instagram to a fixed 20-minute evening window.");
          if (v.distractions > 20) recs.push("Keep the phone outside arm's reach during study/work blocks.");
          if (v.screenTime > 6) recs.push("Set app timers and remove non-essential apps from the home screen.");
          if (v.taskCompletion < 70) recs.push("Choose three priority tasks before opening any social app.");
          return result(score, "Predicted performance", recs, [
            ["Study hours", v.studyHours / 10],
            ["Task completion", v.taskCompletion / 100],
            ["Low distractions", 1 - v.distractions / 80],
            ["Low Instagram", 1 - v.instagram / 240],
            ["Low screen time", 1 - v.screenTime / 12]
          ]);
        }
      },
      expense: {
        title: "Expense Prediction / Money Leak Detection",
        short: "Where does my money disappear?",
        unit: "%",
        mode: "risk",
        fields: [
          ["dailySpend", "Daily spending", 0, 5000, 50, 850],
          ["foodSpend", "Food/snacks spending", 0, 2000, 50, 350],
          ["shoppingSpend", "Shopping spending", 0, 3000, 50, 700],
          ["onlinePayments", "Online purchases count", 0, 20, 1, 6],
          ["impulsePercent", "Impulse purchases percent", 0, 100, 5, 55],
          ["needPercent", "Need-based purchases percent", 0, 100, 5, 45]
        ],
        pros: [
          "Finds invisible money leaks quickly.",
          "Works well with simple daily tracking.",
          "Helps separate need spending from impulse spending."
        ],
        cons: [
          "Requires honest category tagging.",
          "One-time purchases can distort a small dataset.",
          "Income level changes what counts as risky."
        ],
        calculate(v) {
          let risk = 20 + v.dailySpend / 60 + v.shoppingSpend / 70
            + 2.5 * v.onlinePayments + 0.5 * v.impulsePercent
            - 0.25 * v.needPercent;
          risk = clamp(risk, 0, 100);
          const recs = [];
          if (v.impulsePercent > 40) recs.push("Use a 24-hour rule before non-essential purchases.");
          if (v.onlinePayments > 5) recs.push("Reduce one-click payments and review cart items at night.");
          if (v.foodSpend > 300) recs.push("Set a weekly food/snack budget and track cashless payments.");
          if (v.shoppingSpend > 500) recs.push("Create a separate shopping allowance and stop when it is used.");
          return result(risk, "Predicted money-leak risk", recs, [
            ["Daily spend", v.dailySpend / 5000],
            ["Impulse buying", v.impulsePercent / 100],
            ["Shopping spend", v.shoppingSpend / 3000],
            ["Online buys", v.onlinePayments / 20],
            ["Need based", v.needPercent / 100]
          ]);
        }
      },
      sleep: {
        title: "Sleep Quality Analyzer",
        short: "Why do I feel tired even after sleeping?",
        unit: "/10",
        mode: "score",
        fields: [
          ["bedtime", "Bedtime, 24h", 20, 27, 0.1, 24.2],
          ["sleepDuration", "Sleep duration hours", 3, 10, 0.1, 6.1],
          ["phoneBeforeBed", "Phone before bed minutes", 0, 180, 5, 75],
          ["caffeine", "Caffeine cups", 0, 6, 1, 3],
          ["stress", "Stress level", 1, 10, 1, 7],
          ["energy", "Current energy score", 1, 10, 1, 4]
        ],
        pros: [
          "Connects sleep habits with next-day energy.",
          "Useful even with a short one-week log.",
          "Gives clear bedtime and phone-use experiments."
        ],
        cons: [
          "Sleep quality can be affected by health issues.",
          "Self-reported energy is subjective.",
          "Caffeine timing matters, not only count."
        ],
        calculate(v) {
          let quality = 2 + 0.85 * v.sleepDuration - 0.5 * Math.max(v.bedtime - 23, 0)
            - 0.018 * v.phoneBeforeBed - 0.25 * v.caffeine
            - 0.45 * v.stress + 0.2 * v.energy;
          quality = clamp(quality, 1, 10);
          const recs = [];
          if (v.sleepDuration < 7) recs.push("Move bedtime earlier until sleep reaches about 7 hours.");
          if (v.phoneBeforeBed > 45) recs.push("Stop phone use 45 minutes before sleep.");
          if (v.stress > 6) recs.push("Write tomorrow's first task before bed to reduce mental load.");
          if (v.caffeine > 2) recs.push("Avoid caffeine after mid-afternoon and reduce total cups.");
          return result(quality, "Predicted sleep quality", recs, [
            ["Sleep duration", v.sleepDuration / 10],
            ["Early bedtime", 1 - Math.max(v.bedtime - 20, 0) / 7],
            ["Low phone use", 1 - v.phoneBeforeBed / 180],
            ["Low stress", 1 - v.stress / 10],
            ["Energy", v.energy / 10]
          ]);
        }
      },
      commute: {
        title: "Commute Delay Predictor",
        short: "When should I leave to avoid being late?",
        unit: "%",
        mode: "risk",
        fields: [
          ["departure", "Departure time, 24h", 6, 11, 0.1, 8.5],
          ["distance", "Distance km", 1, 40, 1, 12],
          ["traffic", "Traffic level", 1, 10, 1, 8],
          ["rain", "Rain, 0 no / 1 yes", 0, 1, 1, 0],
          ["dayType", "Weekday, 1 yes / 0 weekend", 0, 1, 1, 1],
          ["buffer", "Extra buffer minutes", 0, 60, 5, 10]
        ],
        pros: [
          "Turns lateness into a planning problem.",
          "Easy to collect with maps travel duration.",
          "Useful for school, work, and appointments."
        ],
        cons: [
          "Traffic can change suddenly.",
          "Weather and road closures need fresh data.",
          "A simple model gives guidance, not certainty."
        ],
        calculate(v) {
          let risk = 20 + 4 * v.distance + 6 * v.traffic + 18 * v.rain
            + 12 * v.dayType + 9 * Math.max(8.5 - v.departure, 0)
            - 1.4 * v.buffer;
          risk = clamp(risk, 0, 100);
          const recs = [];
          if (v.traffic > 6) recs.push("Leave 15-20 minutes earlier during peak traffic.");
          if (v.buffer < 15) recs.push("Add at least 15 minutes of buffer for important arrivals.");
          if (v.rain === 1) recs.push("Increase buffer on rainy days because delays are less predictable.");
          if (v.departure >= 8 && v.departure <= 9.5) recs.push("Test leaving before peak time for three days.");
          return result(risk, "Predicted delay risk", recs, [
            ["Traffic", v.traffic / 10],
            ["Distance", v.distance / 40],
            ["Rain", v.rain],
            ["Weekday", v.dayType],
            ["Buffer", v.buffer / 60]
          ]);
        }
      },
      fitness: {
        title: "Personal Fitness Progress Predictor",
        short: "Am I actually improving?",
        unit: "%",
        mode: "score",
        fields: [
          ["workoutMinutes", "Workout minutes", 0, 120, 5, 35],
          ["weeklyWorkouts", "Workouts per week", 0, 7, 1, 3],
          ["sleepHours", "Sleep hours", 3, 10, 0.1, 6.5],
          ["protein", "Protein grams", 20, 180, 5, 75],
          ["calories", "Calories consistency score", 1, 10, 1, 6],
          ["performance", "Current performance score", 1, 10, 1, 5]
        ],
        pros: [
          "Shows whether habits support progress.",
          "Combines training, sleep, and nutrition.",
          "Useful for weekly review."
        ],
        cons: [
          "Progress can be slow and non-linear.",
          "Body weight alone is not enough.",
          "Injury, form, and recovery also matter."
        ],
        calculate(v) {
          let progress = 18 + 0.35 * v.workoutMinutes + 5.5 * v.weeklyWorkouts
            + 4 * v.sleepHours + 0.12 * v.protein + 3 * v.calories
            + 3.5 * v.performance;
          progress = clamp(progress, 0, 100);
          const recs = [];
          if (v.weeklyWorkouts < 3) recs.push("Aim for at least three consistent workouts per week.");
          if (v.sleepHours < 7) recs.push("Improve sleep because recovery drives progress.");
          if (v.protein < 80) recs.push("Increase protein gradually based on your body and diet needs.");
          if (v.calories < 6) recs.push("Keep meals more consistent before judging workout results.");
          return result(progress, "Predicted fitness progress readiness", recs, [
            ["Workout time", v.workoutMinutes / 120],
            ["Weekly workouts", v.weeklyWorkouts / 7],
            ["Sleep", v.sleepHours / 10],
            ["Protein", v.protein / 180],
            ["Consistency", v.calories / 10]
          ]);
        }
      },
      social: {
        title: "Social Media Overuse Risk",
        short: "Unnecessary feed wastes time and causes procrastination.",
        unit: "%",
        mode: "risk",
        fields: [
          ["dailyMinutes", "Daily social media minutes", 0, 480, 5, 210],
          ["unnecessaryFeed", "Unnecessary feed percent", 0, 100, 5, 65],
          ["shortVideos", "Short-video minutes", 0, 300, 5, 95],
          ["notifications", "Notifications per day", 0, 180, 5, 70],
          ["lateNight", "Late-night usage minutes", 0, 180, 5, 55],
          ["workHours", "Study/work hours", 0, 12, 0.1, 4.5],
          ["selfControl", "Self-control score", 1, 10, 1, 4],
          ["purposefulUse", "Purposeful use percent", 0, 100, 5, 25]
        ],
        pros: [
          "Useful for learning, networking, and communication.",
          "Can provide planned entertainment and creative ideas.",
          "Helps creators and students share work."
        ],
        cons: [
          "Unnecessary feed increases mindless scrolling.",
          "Notifications interrupt deep work.",
          "Late-night use can reduce sleep and motivation."
        ],
        calculate(v) {
          let raw = -3.2 + 0.01 * v.dailyMinutes + 0.03 * v.unnecessaryFeed
            + 0.008 * v.shortVideos + 0.012 * v.notifications
            + 0.018 * v.lateNight - 0.17 * v.workHours
            - 0.26 * v.selfControl - 0.012 * v.purposefulUse;
          let risk = 100 / (1 + Math.exp(-raw));
          risk = clamp(risk, 0, 100);
          const recs = [];
          if (v.dailyMinutes > 150) recs.push("Set a daily social app timer near 120 minutes.");
          if (v.unnecessaryFeed > 45) recs.push("Unfollow low-value pages and open feeds only after planned work.");
          if (v.notifications > 35) recs.push("Turn off non-essential notifications.");
          if (v.lateNight > 25) recs.push("Keep the phone away from bed and stop scrolling before sleep.");
          if (v.selfControl <= 5) recs.push("Use app blockers during deep-work blocks.");
          return result(risk, "Predicted procrastination risk", recs, [
            ["Daily minutes", v.dailyMinutes / 480],
            ["Useless feed", v.unnecessaryFeed / 100],
            ["Short videos", v.shortVideos / 300],
            ["Notifications", v.notifications / 180],
            ["Purposeful use", v.purposefulUse / 100]
          ]);
        }
      }
    };

    const issueSelect = document.getElementById("issueSelect");
    const inputFields = document.getElementById("inputFields");
    const prosList = document.getElementById("prosList");
    const consList = document.getElementById("consList");
    const recommendationList = document.getElementById("recommendationList");
    const issueCards = document.getElementById("issueCards");
    const scoreText = document.getElementById("scoreText");
    const levelText = document.getElementById("levelText");
    const resultTitle = document.getElementById("resultTitle");
    const resultSummary = document.getElementById("resultSummary");
    const gauge = document.getElementById("gauge");
    const chart = document.getElementById("factorChart");

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function result(value, title, recommendations, factors) {
      return { value, title, recommendations, factors };
    }

    function levelFor(issue, value) {
      if (issue.mode === "risk") {
        if (value >= 70) return ["High Risk", "#d85c4a"];
        if (value >= 40) return ["Moderate Risk", "#b7791f"];
        return ["Low Risk", "#2f855a"];
      }
      const max = issue.unit === "/10" ? 10 : 100;
      const scaled = value / max;
      if (scaled >= 0.75) return ["Strong", "#2f855a"];
      if (scaled >= 0.55) return ["Moderate", "#b7791f"];
      return ["Needs Attention", "#d85c4a"];
    }

    function formatValue(field, value) {
      const step = Number(field[4]);
      if (step < 1) return Number(value).toFixed(1);
      return String(Math.round(Number(value)));
    }

    function setList(element, items) {
      element.innerHTML = "";
      items.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = item;
        element.appendChild(li);
      });
    }

    function renderIssueOptions() {
      Object.entries(issues).forEach(([key, issue]) => {
        const option = document.createElement("option");
        option.value = key;
        option.textContent = issue.title;
        issueSelect.appendChild(option);

        const card = document.createElement("article");
        card.className = "issue-card";
        card.dataset.issue = key;
        card.innerHTML = `<h3>${issue.title}</h3><p>${issue.short}</p>`;
        card.addEventListener("click", () => {
          issueSelect.value = key;
          renderIssue();
        });
        issueCards.appendChild(card);
      });
    }

    function renderIssue() {
      const key = issueSelect.value;
      const issue = issues[key];
      inputFields.innerHTML = "";
      issue.fields.forEach((field) => {
        const [id, label, min, max, step, value] = field;
        const row = document.createElement("div");
        row.className = "field-row";
        row.innerHTML = `
          <div>
            <label for="${id}">${label} <span class="value-pill" id="${id}Value">${value}</span></label>
            <input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${value}">
          </div>
          <input id="${id}Number" type="number" min="${min}" max="${max}" step="${step}" value="${value}">
        `;
        inputFields.appendChild(row);

        const range = document.getElementById(id);
        const number = document.getElementById(`${id}Number`);
        const pill = document.getElementById(`${id}Value`);
        const sync = (source, target) => {
          target.value = source.value;
          pill.textContent = formatValue(field, source.value);
          predict();
        };
        range.addEventListener("input", () => sync(range, number));
        number.addEventListener("input", () => sync(number, range));
      });

      setList(prosList, issue.pros);
      setList(consList, issue.cons);
      document.querySelectorAll(".issue-card").forEach((card) => {
        card.classList.toggle("active", card.dataset.issue === key);
      });
      predict();
    }

    function valuesForCurrentIssue() {
      const issue = issues[issueSelect.value];
      const values = {};
      issue.fields.forEach((field) => {
        values[field[0]] = Number(document.getElementById(field[0]).value);
      });
      return values;
    }

    function predict() {
      const issue = issues[issueSelect.value];
      const output = issue.calculate(valuesForCurrentIssue());
      const max = issue.unit === "/10" ? 10 : 100;
      const percent = clamp(output.value / max, 0, 1);
      const [level, color] = levelFor(issue, output.value);

      scoreText.textContent = `${output.value.toFixed(issue.unit === "/10" ? 1 : 0)}${issue.unit}`;
      gauge.style.background = `conic-gradient(${color} ${percent * 360}deg, #e6eaef 0deg)`;
      levelText.textContent = level;
      levelText.style.color = color;
      resultTitle.textContent = output.title;
      resultSummary.textContent = `${issue.short} This estimate is based on the values you entered.`;

      const recs = output.recommendations.length
        ? output.recommendations
        : ["Your current pattern looks balanced. Keep tracking for one week to confirm it."];
      setList(recommendationList, recs);
      drawChart(output.factors, issue.mode);
    }

    function drawChart(factors, mode) {
      const ctx = chart.getContext("2d");
      const width = chart.width;
      const height = chart.height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);

      const margin = 42;
      const barHeight = 26;
      const gap = 17;
      const maxBarWidth = width - 220;
      ctx.font = "14px Arial";
      ctx.textBaseline = "middle";

      factors.forEach((factor, index) => {
        const y = margin + index * (barHeight + gap);
        const label = factor[0];
        const value = clamp(factor[1], 0, 1);
        const color = mode === "risk" ? "#d85c4a" : "#147d7e";

        ctx.fillStyle = "#344054";
        ctx.fillText(label, 18, y + barHeight / 2);
        ctx.fillStyle = "#edf0f4";
        ctx.fillRect(170, y, maxBarWidth, barHeight);
        ctx.fillStyle = color;
        ctx.fillRect(170, y, maxBarWidth * value, barHeight);
        ctx.fillStyle = "#1d2630";
        ctx.fillText(`${Math.round(value * 100)}%`, 182 + maxBarWidth, y + barHeight / 2);
      });
    }

    function useSampleValues() {
      const issue = issues[issueSelect.value];
      issue.fields.forEach((field) => {
        const id = field[0];
        const value = field[5];
        document.getElementById(id).value = value;
        document.getElementById(`${id}Number`).value = value;
        document.getElementById(`${id}Value`).textContent = formatValue(field, value);
      });
      predict();
    }

    function downloadCsvTemplate() {
      const issue = issues[issueSelect.value];
      const headers = issue.fields.map((field) => field[1].replaceAll(",", ""));
      const csv = `${headers.join(",")}\n${issue.fields.map((field) => field[5]).join(",")}\n`;
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${issue.title.toLowerCase().replaceAll(" ", "_")}_template.csv`;
      link.click();
      URL.revokeObjectURL(url);
    }

    renderIssueOptions();
    issueSelect.value = "productivity";
    renderIssue();

    issueSelect.addEventListener("change", renderIssue);
    document.getElementById("predictButton").addEventListener("click", predict);
    document.getElementById("sampleButton").addEventListener("click", useSampleValues);
    document.getElementById("csvButton").addEventListener("click", downloadCsvTemplate);
  </script>
</body>
</html>
"""

    DASHBOARD_PATH.write_text(textwrap.dedent(dashboard).strip(), encoding="utf-8")
    return DASHBOARD_PATH


def main():
    REPORT_DIR.mkdir(exist_ok=True)

    productivity_data = create_productivity_dataset()
    productivity_data.to_csv(PRODUCTIVITY_DATA_PATH, index=False)
    (
        productivity_linear_model,
        productivity_tree_model,
        scored_productivity_data,
        productivity_features,
    ) = train_productivity_models(productivity_data)
    productivity_metrics = {
        "linear": regression_metrics(
            scored_productivity_data["productivity_score"],
            scored_productivity_data["linear_prediction"],
        ),
        "tree": regression_metrics(
            scored_productivity_data["productivity_score"],
            scored_productivity_data["tree_prediction"],
        ),
    }
    make_productivity_visualizations(scored_productivity_data)
    productivity_report_path = write_productivity_report(
        scored_productivity_data,
        productivity_linear_model,
        productivity_tree_model,
        productivity_metrics,
        productivity_features,
    )

    data = create_synthetic_dataset()
    data.to_csv(DATA_PATH, index=False)

    model, scored_data, feature_columns = train_logistic_model(data)
    metrics = calculate_metrics(scored_data)
    make_visualizations(scored_data)
    report_path = write_report(scored_data, model, metrics, feature_columns)
    dashboard_path = generate_interactive_dashboard()

    example_profile = {
        "daily_minutes": 210,
        "unnecessary_feed_percent": 65,
        "short_video_minutes": 95,
        "notifications_per_day": 70,
        "late_night_usage_minutes": 55,
        "study_or_work_hours": 4.5,
        "self_control_score": 4,
        "purposeful_use_percent": 25,
    }
    probability, level = predict_personal_risk(
        model,
        feature_columns,
        example_profile,
    )

    print("\nSOCIAL MEDIA OVERUSE PREDICTION PROJECT")
    print("=" * 48)
    print(f"Dataset saved to: {DATA_PATH}")
    print(f"Report saved to: {report_path}")
    print(f"Charts saved in: {REPORT_DIR}")
    print("\nModel performance")
    print(f"Accuracy : {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall   : {metrics['recall']:.2%}")
    print("\nExample prediction")
    print(f"Predicted procrastination risk: {probability:.2%}")
    print(f"Risk level: {level}")
    print("\nRecommended actions")
    for index, recommendation in enumerate(
        build_recommendations(example_profile, probability),
        start=1,
    ):
        print(f"{index}. {recommendation}")

    productivity_profile = {
        "sleep_hours": 5.8,
        "wake_up_hour": 8.7,
        "screen_time_hours": 7.2,
        "social_media_minutes": 130,
        "coffee_cups": 2,
        "exercise": 0,
        "mood_score": 5,
        "focused_work_hours": 2.4,
    }
    productivity_prediction, productivity_level_name = predict_productivity(
        productivity_linear_model,
        productivity_tree_model,
        productivity_features,
        productivity_profile,
    )

    print("\nDAILY PRODUCTIVITY PREDICTION PROJECT")
    print("=" * 48)
    print(f"Dataset saved to: {PRODUCTIVITY_DATA_PATH}")
    print(f"Report saved to: {productivity_report_path}")
    print(f"Dashboard saved to: {dashboard_path}")
    print("\nLinear Regression performance")
    print(f"MAE      : {productivity_metrics['linear']['mae']:.2f}")
    print(f"RMSE     : {productivity_metrics['linear']['rmse']:.2f}")
    print(f"R-squared: {productivity_metrics['linear']['r_squared']:.2f}")
    print("\nDecision Tree performance")
    print(f"MAE      : {productivity_metrics['tree']['mae']:.2f}")
    print(f"RMSE     : {productivity_metrics['tree']['rmse']:.2f}")
    print(f"R-squared: {productivity_metrics['tree']['r_squared']:.2f}")
    print("\nExample productivity prediction")
    print(f"Predicted productivity score: {productivity_prediction:.1f}/10")
    print(f"Productivity level: {productivity_level_name}")
    print("\nProductivity recommendations")
    for index, recommendation in enumerate(
        build_productivity_recommendations(
            productivity_profile,
            productivity_prediction,
        ),
        start=1,
    ):
        print(f"{index}. {recommendation}")


if __name__ == "__main__":
    main()
