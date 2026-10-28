
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

st.set_page_config(
    page_title="Eco-Health Risk Predictor",
    layout="wide"
)

st.title("Eco-Health Risk Predictor Dashboard")

st.markdown("""
This dashboard uses environmental measurements to predict
health-impact classes with a Random Forest classifier.

Predictions are intended for educational and analytical purposes.
They are not validated medical or public-health assessments.
""")

st.divider()

DATA_PATH = "air_quality_health_impact_data.csv"


def normalize_column_name(name):
    return "".join(
        character.lower()
        for character in str(name)
        if character.isalnum()
    )


@st.cache_data
def load_data(file_path, modified_time):
    data = pd.read_csv(file_path)
    data.columns = data.columns.astype(str).str.strip()
    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def identify_target(columns):
    normalized = {
        column: normalize_column_name(column)
        for column in columns
    }

    priorities = [
        "healthimpactclass",
        "healthriskclass",
        "healthrisk",
    ]

    for target_name in priorities:
        for column, normalized_name in normalized.items():
            if normalized_name == target_name:
                return column

    return None


def identify_features(data, target):
    excluded = []

    leakage_columns = {
        "healthimpactscore",
        "respiratorycases",
        "cardiovascularcases",
        "hospitaladmissions",
    }

    identifier_columns = {"id", "recordid"}

    for column in data.columns:
        if column == target:
            continue

        normalized = normalize_column_name(column)

        if normalized in identifier_columns:
            excluded.append(column)
        elif normalized in leakage_columns:
            excluded.append(column)

    features = data.drop(
        columns=[target] + excluded
    ).copy()

    all_missing = features.columns[
        features.isna().all()
    ].tolist()

    features = features.drop(columns=all_missing)
    excluded.extend(all_missing)

    return features, excluded


def get_class_labels(target, y):
    values = sorted(
        y.unique().tolist(),
        key=lambda value: (
            0, value
        ) if isinstance(value, (int, float, np.integer, np.floating))
        else (1, str(value))
    )

    target_name = normalize_column_name(target)

    if (
        target_name == "healthimpactclass"
        and set(values) == {0, 1, 2, 3, 4}
    ):
        labels = {
            0: "Very High Risk",
            1: "High Risk",
            2: "Moderate Risk",
            3: "Low Risk",
            4: "Very Low Risk",
        }

    elif (
        target_name in {"healthrisk", "healthriskclass"}
        and set(values) == {0, 1, 2}
    ):
        labels = {
            0: "Low Risk",
            1: "Moderate Risk",
            2: "High Risk",
        }

    else:
        labels = {
            value: str(value)
            for value in values
        }

        warnings.warn(
            "Unrecognised target encoding. "
            "Actual target values are displayed as labels."
        )

    return values, labels


@st.cache_resource
def train_model(data, target, feature_columns):
    X = data[list(feature_columns)].copy()
    y = data[target].copy()

    if y.value_counts().min() < 2:
        raise ValueError(
            "At least one target class has fewer than two records. "
            "A stratified split cannot be performed reliably."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_features = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_features = [
        column
        for column in X_train.columns
        if column not in numeric_features
    ]

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return model, X_test, y_test, predictions


if not os.path.exists(DATA_PATH):
    st.error(
        f"Dataset not found: {DATA_PATH}. "
        "Place the CSV file beside app.py."
    )
    st.stop()

try:
    modified_time = os.path.getmtime(DATA_PATH)
    df = load_data(DATA_PATH, modified_time)

    if df.empty:
        st.error("The dataset contains no records.")
        st.stop()

    target_column = identify_target(df.columns)

    if target_column is None:
        st.error(
            "Could not identify the target column. "
            "Expected HealthImpactClass, HealthRiskClass, or HealthRisk."
        )
        st.write("Available columns:", df.columns.tolist())
        st.stop()

    df = df.dropna(subset=[target_column]).copy()

    X, excluded_columns = identify_features(
        df,
        target_column
    )

    if X.empty:
        st.error("No usable predictor columns remain.")
        st.stop()

    if df[target_column].nunique() < 2:
        st.error("The target must contain at least two classes.")
        st.stop()

    class_order, class_labels = get_class_labels(
        target_column,
        df[target_column]
    )

    model, X_test, y_test, y_pred = train_model(
        df,
        target_column,
        tuple(X.columns)
    )

except (ValueError, KeyError) as error:
    st.error(f"Unable to prepare the model: {error}")
    st.stop()


st.subheader("Dataset Overview")

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Total Records",
    f"{len(df):,}"
)

metric2.metric(
    "Predictor Features",
    f"{X.shape[1]:,}"
)

metric3.metric(
    "Target Classes",
    f"{df[target_column].nunique():,}"
)

st.write("Target column:", f"`{target_column}`")

with st.expander("Dataset preview and excluded columns"):
    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.write("Excluded columns:", excluded_columns)
    st.write("Features used:", X.columns.tolist())


accuracy = accuracy_score(y_test, y_pred)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    labels=class_order,
    average="macro",
    zero_division=0
)

st.divider()
st.subheader("Model Evaluation")

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Test Accuracy",
    f"{accuracy:.1%}"
)

metric2.metric(
    "Balanced Accuracy",
    f"{balanced_accuracy:.1%}"
)

metric3.metric(
    "Macro F1-Score",
    f"{macro_f1:.3f}"
)

left_column, right_column = st.columns(
    [1.2, 1]
)

with left_column:
    st.subheader("Confusion Matrix")

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=class_order
    )

    class_names = [
        class_labels[value]
        for value in class_order
    ]

    fig, ax = plt.subplots(
        figsize=(
            max(7, len(class_names) * 1.4),
            max(5, len(class_names) * 1.1)
        )
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("Random Forest Classification Results")

    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        labels=class_order,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

with right_column:
    st.subheader("Interactive Prediction Engine")

    st.write(
        "Enter environmental measurements to generate "
        "a prediction using the trained model."
    )

    user_inputs = {}

    with st.form("prediction_form"):

        for column in X.columns:
            series = df[column]

            if pd.api.types.is_numeric_dtype(series):
                values = pd.to_numeric(
                    series,
                    errors="coerce"
                ).dropna()

                if values.empty:
                    continue

                minimum = float(values.min())
                maximum = float(values.max())
                default = float(values.median())

                default = min(
                    max(default, minimum),
                    maximum
                )

                integer_feature = (
                    pd.api.types.is_integer_dtype(series.dtype)
                    or (
                        np.allclose(
                            values.to_numpy(),
                            np.round(values.to_numpy())
                        )
                    )
                )

                if minimum == maximum:
                    user_inputs[column] = minimum

                elif integer_feature:
                    user_inputs[column] = st.slider(
                        column,
                        min_value=int(minimum),
                        max_value=int(maximum),
                        value=int(round(default)),
                        step=1
                    )

                else:
                    user_inputs[column] = st.slider(
                        column,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=(maximum - minimum) / 100
                    )

            else:
                options = series.dropna().unique().tolist()

                if not options:
                    continue

                mode_values = series.mode(dropna=True)
                default = (
                    mode_values.iloc[0]
                    if not mode_values.empty
                    else options[0]
                )

                default_index = options.index(default)

                user_inputs[column] = st.selectbox(
                    column,
                    options=options,
                    index=default_index
                )

        submitted = st.form_submit_button(
            "Predict Health-Impact Class",
            use_container_width=True
        )

    if submitted:
        user_df = pd.DataFrame(
            [user_inputs],
            columns=X.columns
        )

        prediction = model.predict(user_df)[0]
        predicted_label = class_labels[prediction]

        st.success(
            f"Predicted class: {predicted_label}"
        )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(user_df)[0]

            probability_df = pd.DataFrame({
                "Class": [
                    class_labels[value]
                    for value in model.classes_
                ],
                "Estimated Probability": probabilities,
            }).sort_values(
                "Estimated Probability",
                ascending=False
            )

            st.write("Model-estimated class probabilities:")
            st.dataframe(
                probability_df,
                hide_index=True,
                use_container_width=True
            )

        st.caption(
            "These predictions and probabilities reflect the fitted "
            "model, not a validated clinical risk assessment."
        )


st.divider()
st.subheader("Feature Importance")

try:
    feature_names = (
        model.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importance_values = (
        model.named_steps["classifier"]
        .feature_importances_
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values,
    }).sort_values(
        "Importance",
        ascending=False
    ).head(15)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        importance_df["Feature"][::-1],
        importance_df["Importance"][::-1]
    )

    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top 15 Predictive Features")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

except (ValueError, AttributeError):
    st.info("Feature importance is unavailable for this model.")

st.caption(
    "Feature importance describes model reliance, not causation. "
    "Independent real-world validation is required before any "
    "public-health application."
)
