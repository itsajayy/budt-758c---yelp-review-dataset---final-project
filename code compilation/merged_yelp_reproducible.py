# Merged Yelp highly-useful review prediction pipeline

# Generated from compiled code.ipynb cells 2-50 with preservation-focused diagnostics.



# %% [source cell 2]

# Yelp Top-Useful Prediction
# Super Advanced 85+ Version
#
# Includes:
# 1. External data source: US holiday calendar
# 2. 10+ feature insight tables/charts
# 3. 6+ model comparison
# 4. Cross-validation
# 5. Learning curve
# 6. Tuning curve
# 7. Model zoo: many LGB/XGB/MLP variants
# 8. Advanced top-K random blend
# 9. Final submission under FPR <= 0.097

import os
import gc
import ast
import json
import re
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import lightgbm as lgb
import xgboost as xgb
import holidays


# %% [source cell 4]

# 1. Settings

RANDOM_STATE = 42
N_JOBS = -1

# import dataset
# Original notebook path is preserved as the fallback. Set YELP_DATA_DIR to run
# the same pipeline from a different machine without changing model logic.
ORIGINAL_DATA_DIR = r"C:\Users\laoti\Desktop\Grad School\BUDT758T\Data&Sample"
DATA_DIR = os.environ.get("YELP_DATA_DIR", ORIGINAL_DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, "review_train_x.csv")):
    cwd_data_candidate = os.getcwd()
    if os.path.exists(os.path.join(cwd_data_candidate, "review_train_x.csv")):
        DATA_DIR = cwd_data_candidate
OUTPUT_DIR = DATA_DIR

FPR_LIMIT = 0.10
FPR_LIMIT_FINAL = 0.097

# The result from last submission
CURRENT_BEST_TPR = 0.7356718721389521
CURRENT_BEST_FPR = 0.09698189570279951

# Values used in the written final report / leaderboard comparison.
REPORT_FINAL_TPR = 0.735227
FIRST_PLACE_TPR = 0.7364

# change user.useful to False because it is leakage
USE_USER_USEFUL_FEATURES = True

# To run the blocks required for the report:: 6 models / CV / learning curve / tuning curve
RUN_REPORT_BLOCKS = True

# Whether to run more models
RUN_EXTRA_LGB = True
RUN_EXTRA_XGB = True
RUN_EXTRA_MLP = True

# For strong computer
MODEL_COMPARE_SAMPLE_SIZE = 200000
CURVE_SAMPLE_SIZE = 400000

# Advanced blend settings
TOP_K_FOR_BLEND = 12
BLEND_SEARCH_SAMPLE_SIZE = 200000
RANDOM_BLEND_TRIALS = 30000
RANDOM_BLEND_TOP_EVAL = 300

# MLP seeds
MLP_SEEDS = [99, 42, 21, 1, 123]

# Whether to use GPU for XGBoost
# Default is False to prevent environment-related errors
# If XGBoost supports CUDA, you can change it to True
USE_XGB_GPU = False

print("Settings loaded.")
print("DATA_DIR:", DATA_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)
print(
    "Conflict note: later notebook cells and separate RF/tree/HGB scripts restart "
    "feature selection, splitting, or threshold logic. They are intentionally not "
    "run in this final blend pipeline."
)


def print_merge_checks(df, label):
    """Print row/column/missing diagnostics without mutating the dataframe."""
    missing_total = int(df.isna().sum().sum())
    print(f"{label}: rows={df.shape[0]}, columns={df.shape[1]}, missing_values={missing_total}")
    top_missing = df.isna().sum().sort_values(ascending=False).head(10)
    top_missing = top_missing[top_missing > 0]
    if len(top_missing) > 0:
        print(f"{label} top missing columns:")
        print(top_missing)


def merge_with_checks(left, right, *, on, how, label):
    """Run the original pandas merge and print diagnostics immediately after."""
    before_rows = len(left)
    before_cols = left.shape[1]
    if right.duplicated(subset=on).any():
        dup_count = int(right.duplicated(subset=on).sum())
        print(f"WARNING: right table for {label} has {dup_count} duplicate key rows on {on}.")
    out = left.merge(right, on=on, how=how)
    print(
        f"{label}: before=({before_rows}, {before_cols}), "
        f"after={out.shape}, row_delta={len(out) - before_rows}"
    )
    print_merge_checks(out, label)
    return out


def warn_duplicate_columns(df, label):
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"WARNING: duplicate columns in {label} before de-duplication: {duplicate_cols}")
    else:
        print(f"No duplicate columns detected in {label}.")


# %% [source cell 6]

# 2. Load Data

def read_csv_from_data_dir(filename):
    path = os.path.join(DATA_DIR, filename)
    print("Reading:", path)
    return pd.read_csv(path)

# Read csv files
train_x = read_csv_from_data_dir("review_train_x.csv")
train_y = read_csv_from_data_dir("review_train_y.csv")
test_x = read_csv_from_data_dir("review_test_x.csv")
business = read_csv_from_data_dir("business.csv")
user = read_csv_from_data_dir("user.csv")
tip = read_csv_from_data_dir("tip.csv")

if train_y.shape[1] == 1:
    y = train_y.iloc[:, 0].astype(int).reset_index(drop=True)
else:
    y = train_y["top_useful"].astype(int).reset_index(drop=True)

# Find distribution and shapes
print("train_x:", train_x.shape)
print("train_y:", train_y.shape)
print("test_x:", test_x.shape)
print("business:", business.shape)
print("user:", user.shape)
print("tip:", tip.shape)

print("\nLabel distribution:")
print(y.value_counts())
print("Positive rate:", y.mean())


# %% [source cell 8]

# ============================================================
# 3. Evaluation Functions
# ============================================================

# Find the best TPR under a specific FPR limit
def best_tpr_under_fpr(y_true, y_score, fpr_limit=0.10):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid_idx = np.where(fpr <= fpr_limit)[0]

    if len(valid_idx) == 0:
        return {
            "auc": roc_auc_score(y_true, y_score),
            "fpr": np.nan,
            "tpr": np.nan,
            "threshold": np.nan,
            "accuracy": np.nan,
            "tn": np.nan,
            "fp": np.nan,
            "fn": np.nan,
            "tp": np.nan
        }

    best_idx = valid_idx[np.argmax(tpr[valid_idx])]
    best_threshold = thresholds[best_idx]

    y_pred = (y_score >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "auc": roc_auc_score(y_true, y_score),
        "fpr": fp / (fp + tn),
        "tpr": tp / (tp + fn),
        "threshold": best_threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


def print_metrics(name, metrics):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    for k, v in metrics.items():
        print(f"{k}: {v}")


def get_model_score(model, X_data):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_data)[:, 1]
    else:
        return model.decision_function(X_data)


# %% [source cell 10]

# ============================================================
# 4. External Holiday Features
# ============================================================

def add_us_holiday_features(df):
    df = df.copy()

    if "date" not in df.columns:
        df["is_us_holiday"] = 0.0
        df["days_to_nearest_holiday"] = 999.0
        df["days_since_previous_holiday"] = 999.0
        df["days_until_next_holiday"] = 999.0
        df["is_holiday_window_3"] = 0.0
        df["is_holiday_window_7"] = 0.0
        df["is_before_holiday_3"] = 0.0
        df["is_after_holiday_3"] = 0.0
        return df

    date_series = pd.to_datetime(df["date"], errors="coerce")
    years = date_series.dt.year.dropna().astype(int)

    if len(years) == 0:
        min_year, max_year = 2004, 2023
    else:
        min_year = int(years.min()) - 1
        max_year = int(years.max()) + 1

    us_holidays = holidays.US(years=range(min_year, max_year + 1))
    holiday_dates = sorted(list(us_holidays.keys()))

    holiday_ord = np.array(
        pd.to_datetime(holiday_dates).values.astype("datetime64[D]").astype("int64")
    )

    date_day = date_series.dt.floor("D")
    valid_mask = date_series.notna().values
    date_ord_all = date_day.values.astype("datetime64[D]").astype("int64")

    is_holiday = np.zeros(len(df), dtype="float32")
    nearest_days = np.full(len(df), 999.0, dtype="float32")
    since_prev = np.full(len(df), 999.0, dtype="float32")
    until_next = np.full(len(df), 999.0, dtype="float32")

    valid_dates = date_ord_all[valid_mask]

    idx = np.searchsorted(holiday_ord, valid_dates)

    has_prev = idx > 0
    has_next = idx < len(holiday_ord)

    prev_days = np.full(len(valid_dates), 999.0)
    next_days = np.full(len(valid_dates), 999.0)

    prev_days[has_prev] = valid_dates[has_prev] - holiday_ord[idx[has_prev] - 1]
    next_days[has_next] = holiday_ord[idx[has_next]] - valid_dates[has_next]

    exact_holiday = np.isin(valid_dates, holiday_ord)
    prev_days[exact_holiday] = 0
    next_days[exact_holiday] = 0

    nearest = np.minimum(prev_days, next_days)

    is_holiday[valid_mask] = exact_holiday.astype("float32")
    nearest_days[valid_mask] = nearest.astype("float32")
    since_prev[valid_mask] = prev_days.astype("float32")
    until_next[valid_mask] = next_days.astype("float32")

    df["is_us_holiday"] = is_holiday
    df["days_to_nearest_holiday"] = nearest_days
    df["days_since_previous_holiday"] = since_prev
    df["days_until_next_holiday"] = until_next

    df["is_holiday_window_3"] = (df["days_to_nearest_holiday"] <= 3).astype("float32")
    df["is_holiday_window_7"] = (df["days_to_nearest_holiday"] <= 7).astype("float32")
    df["is_before_holiday_3"] = (df["days_until_next_holiday"] <= 3).astype("float32")
    df["is_after_holiday_3"] = (df["days_since_previous_holiday"] <= 3).astype("float32")

    return df


# %% [source cell 12]

# ============================================================
# 5. Review-Level Features
# ============================================================

def add_review_features(df):
    df = df.copy()

    if "text" not in df.columns:
        df["text"] = ""

    df["text"] = df["text"].fillna("").astype(str)

    df["text_len"] = df["text"].str.len().astype("float32")
    df["word_count"] = df["text"].str.split().str.len().astype("float32")
    df["avg_word_len"] = df["text_len"] / (df["word_count"] + 1)

    df["sentence_count"] = df["text"].str.count(r"[.!?]+").astype("float32")
    df["exclaim_count"] = df["text"].str.count("!").astype("float32")
    df["question_count"] = df["text"].str.count(r"\?").astype("float32")
    df["comma_count"] = df["text"].str.count(",").astype("float32")
    df["period_count"] = df["text"].str.count(r"\.").astype("float32")
    df["digit_count"] = df["text"].str.count(r"\d").astype("float32")

    df["upper_count"] = df["text"].str.count(r"[A-Z]").astype("float32")
    df["upper_ratio"] = df["upper_count"] / (df["text_len"] + 1)
    df["upper_per_word"] = df["upper_count"] / (df["word_count"] + 1)

    df["log_text_len"] = np.log1p(df["text_len"].clip(lower=0))
    df["log_word_count"] = np.log1p(df["word_count"].clip(lower=0))

    df["punctuation_total"] = (
            df["exclaim_count"] +
            df["question_count"] +
            df["comma_count"] +
            df["period_count"]
    )

    df["period_per_word"] = df["period_count"] / (df["word_count"] + 1)
    df["punctuation_per_word"] = df["punctuation_total"] / (df["word_count"] + 1)

    positive_words = [
        "good", "great", "excellent", "amazing", "awesome", "best",
        "love", "loved", "perfect", "delicious", "friendly", "nice",
        "wonderful", "fantastic", "recommend", "favorite", "fresh",
        "clean", "fast", "helpful", "enjoyed"
    ]

    negative_words = [
        "bad", "terrible", "awful", "worst", "hate", "hated",
        "poor", "rude", "slow", "dirty", "disappointed",
        "horrible", "never", "overpriced", "cold", "bland",
        "wait", "waiting", "wrong", "expensive"
    ]

    text_lower = df["text"].str.lower()

    df["positive_word_count"] = 0
    for w in positive_words:
        df["positive_word_count"] += text_lower.str.count(r"\b" + w + r"\b")

    df["negative_word_count"] = 0
    for w in negative_words:
        df["negative_word_count"] += text_lower.str.count(r"\b" + w + r"\b")

    df["positive_word_count"] = df["positive_word_count"].astype("float32")
    df["negative_word_count"] = df["negative_word_count"].astype("float32")

    df["sentiment_balance"] = df["positive_word_count"] - df["negative_word_count"]
    df["positive_word_ratio"] = df["positive_word_count"] / (df["word_count"] + 1)
    df["negative_word_ratio"] = df["negative_word_count"] / (df["word_count"] + 1)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["review_year"] = df["date"].dt.year.astype("float32")
        df["review_month"] = df["date"].dt.month.astype("float32")
        df["review_dayofweek"] = df["date"].dt.dayofweek.astype("float32")
        df["review_hour"] = df["date"].dt.hour.astype("float32")
        df["is_weekend"] = df["review_dayofweek"].isin([5, 6]).astype("float32")
    else:
        df["review_year"] = np.nan
        df["review_month"] = np.nan
        df["review_dayofweek"] = np.nan
        df["review_hour"] = np.nan
        df["is_weekend"] = np.nan

    if "stars" in df.columns:
        df["review_stars"] = pd.to_numeric(df["stars"], errors="coerce").astype("float32")
    else:
        df["review_stars"] = np.nan

    df["is_extreme_star"] = ((df["review_stars"] <= 2) | (df["review_stars"] >= 5)).astype("float32")
    df["is_positive_star"] = (df["review_stars"] >= 4).astype("float32")
    df["is_negative_star"] = (df["review_stars"] <= 2).astype("float32")

    df = add_us_holiday_features(df)

    return df


# %% [source cell 14]

# ============================================================
# 6. Business Features
# ============================================================

def parse_attr_dict(val):
    if isinstance(val, dict):
        return val

    if pd.isna(val):
        return {}

    s = str(val)

    try:
        return json.loads(
            s.replace("'", '"')
            .replace("True", "true")
            .replace("False", "false")
            .replace("None", "null")
        )
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return {}


def get_numeric_col(df, col, default=np.nan):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    else:
        return pd.Series([default] * len(df), index=df.index)


def attr_to_number(attr_dict, key):
    val = attr_dict.get(key, np.nan)

    if isinstance(val, bool):
        return float(val)

    if pd.isna(val):
        return np.nan

    val_str = str(val).strip().strip("'\"").lower()

    if val_str in ["true", "1", "yes"]:
        return 1.0
    if val_str in ["false", "0", "no"]:
        return 0.0
    if val_str in ["none", "nan", "null", ""]:
        return np.nan

    try:
        return float(val_str)
    except Exception:
        return np.nan


def map_attr_value(val, mapping):
    if pd.isna(val):
        return np.nan

    val_str = str(val).strip().strip("'\"").lower()
    return mapping.get(val_str, np.nan)


def clean_business_features(business):
    b = business.copy()
    b = b.drop_duplicates(subset=["business_id"], keep="first")

    b["business_stars"] = get_numeric_col(b, "stars").astype("float32")
    b["business_review_count"] = get_numeric_col(b, "review_count").astype("float32")
    b["is_open"] = get_numeric_col(b, "is_open", default=0).fillna(0).astype("float32")
    b["latitude"] = get_numeric_col(b, "latitude").astype("float32")
    b["longitude"] = get_numeric_col(b, "longitude").astype("float32")

    b["log_business_review_count"] = np.log1p(b["business_review_count"].clip(lower=0))

    if "categories" in b.columns:
        b["categories"] = b["categories"].fillna("").astype(str).str.lower()
    else:
        b["categories"] = ""

    b["category_count"] = b["categories"].apply(
        lambda x: 0 if x == "" else len([c for c in x.split(",") if c.strip()])
    ).astype("float32")

    category_keywords = {
        "cat_restaurant": "restaurant",
        "cat_food": "food",
        "cat_bar": "bar",
        "cat_nightlife": "nightlife",
        "cat_coffee": "coffee",
        "cat_shopping": "shopping",
        "cat_beauty": "beauty",
        "cat_hotel": "hotel",
        "cat_auto": "auto",
        "cat_health": "health",
        "cat_mexican": "mexican",
        "cat_chinese": "chinese",
        "cat_japanese": "japanese",
        "cat_american": "american",
        "cat_pizza": "pizza",
        "cat_burgers": "burgers",
        "cat_seafood": "seafood",
        "cat_breakfast": "breakfast",
        "cat_fastfood": "fast food",
        "cat_italian": "italian",
        "cat_sushi": "sushi",
        "cat_sandwich": "sandwich",
        "cat_event": "event planning",
        "cat_service": "services"
    }

    for new_col, keyword in category_keywords.items():
        b[new_col] = b["categories"].str.contains(keyword, regex=False).astype("float32")

    if "attributes" in b.columns:
        attrs = b["attributes"].apply(parse_attr_dict)
    else:
        attrs = pd.Series([{} for _ in range(len(b))], index=b.index)

    simple_attr_keys = [
        "BusinessAcceptsCreditCards",
        "BikeParking",
        "ByAppointmentOnly",
        "RestaurantsReservations",
        "RestaurantsGoodForGroups",
        "RestaurantsTakeOut",
        "GoodForKids",
        "HasTV",
        "Caters",
        "HappyHour",
        "OutdoorSeating",
        "RestaurantsDelivery",
        "RestaurantsTableService",
        "WheelchairAccessible",
        "DogsAllowed"
    ]

    for key in simple_attr_keys:
        b["attr_" + key] = attrs.apply(lambda d: attr_to_number(d, key)).astype("float32")

    attire_map = {"casual": 1.0, "dressy": 2.0, "formal": 3.0}
    alcohol_map = {"none": 0.0, "beer_and_wine": 1.0, "full_bar": 2.0}
    noise_map = {"quiet": 1.0, "average": 2.0, "loud": 3.0, "very_loud": 4.0}
    wifi_map = {"no": 0.0, "free": 1.0, "paid": 2.0}

    price_raw = attrs.apply(lambda d: d.get("RestaurantsPriceRange2", np.nan))
    b["attr_RestaurantsPriceRange2"] = pd.to_numeric(price_raw, errors="coerce").astype("float32")

    b["attr_RestaurantsAttire"] = attrs.apply(
        lambda d: map_attr_value(d.get("RestaurantsAttire", np.nan), attire_map)
    ).astype("float32")

    b["attr_Alcohol"] = attrs.apply(
        lambda d: map_attr_value(d.get("Alcohol", np.nan), alcohol_map)
    ).astype("float32")

    b["attr_NoiseLevel"] = attrs.apply(
        lambda d: map_attr_value(d.get("NoiseLevel", np.nan), noise_map)
    ).astype("float32")

    b["attr_WiFi"] = attrs.apply(
        lambda d: map_attr_value(d.get("WiFi", np.nan), wifi_map)
    ).astype("float32")

    def parse_hours(val):
        if pd.isna(val):
            return 0, 0.0

        try:
            h = ast.literal_eval(str(val))
            if not isinstance(h, dict):
                return 0, 0.0

            days_open = len(h)
            total_hours = 0.0

            for _, timerange in h.items():
                parts = str(timerange).split("-")
                if len(parts) != 2:
                    continue

                s1 = parts[0].split(":")
                s2 = parts[1].split(":")

                start = float(s1[0]) + float(s1[1]) / 60
                end = float(s2[0]) + float(s2[1]) / 60

                diff = end - start
                if diff < 0:
                    diff += 24

                total_hours += diff

            avg_hours = total_hours / days_open if days_open > 0 else 0.0
            return days_open, avg_hours

        except Exception:
            return 0, 0.0

    if "hours" in b.columns:
        hours_parsed = b["hours"].apply(parse_hours)
        b["days_open"] = hours_parsed.apply(lambda x: x[0]).astype("float32")
        b["avg_hours_per_day"] = hours_parsed.apply(lambda x: x[1]).astype("float32")
    else:
        b["days_open"] = 0.0
        b["avg_hours_per_day"] = 0.0

    keep_cols = [
        "business_id",
        "business_stars",
        "business_review_count",
        "log_business_review_count",
        "is_open",
        "latitude",
        "longitude",
        "category_count",
        "days_open",
        "avg_hours_per_day"
    ]

    keep_cols += list(category_keywords.keys())
    keep_cols += ["attr_" + key for key in simple_attr_keys]
    keep_cols += [
        "attr_RestaurantsPriceRange2",
        "attr_RestaurantsAttire",
        "attr_Alcohol",
        "attr_NoiseLevel",
        "attr_WiFi"
    ]

    keep_cols = [c for c in keep_cols if c in b.columns]

    return b[keep_cols]


# %% [source cell 16]

# ============================================================
# 7. User Features
# ============================================================

def clean_user_features(user):
    u = user.copy()
    u = u.drop_duplicates(subset=["user_id"], keep="first")

    numeric_cols = ["review_count", "funny", "cool", "fans", "average_stars"]

    if USE_USER_USEFUL_FEATURES:
        numeric_cols.append("useful")

    for col in numeric_cols:
        if col in u.columns:
            u["user_" + col] = pd.to_numeric(u[col], errors="coerce").astype("float32")
        else:
            u["user_" + col] = np.nan

    if "elite" in u.columns:
        u["elite"] = u["elite"].fillna("").astype(str)
        u["elite"] = u["elite"].str.replace("20,20", "2020", regex=False)

        u["user_elite_count"] = u["elite"].apply(
            lambda x: 0 if x.strip() == "" or x.lower() in ["nan", "none"]
            else len([i for i in x.split(",") if i.strip()])
        ).astype("float32")
    else:
        u["user_elite_count"] = 0.0

    if "friends" in u.columns:
        u["friends"] = u["friends"].fillna("").astype(str)
        u["user_friend_count"] = u["friends"].apply(
            lambda x: 0 if x.strip() == "" or x.lower() in ["none", "nan"]
            else len([i for i in x.split(",") if i.strip()])
        ).astype("float32")
    else:
        u["user_friend_count"] = 0.0

    if "yelping_since" in u.columns:
        u["yelping_since"] = pd.to_datetime(u["yelping_since"], errors="coerce")
        u["yelping_since_year"] = u["yelping_since"].dt.year.astype("float32")
    else:
        u["yelping_since_year"] = np.nan

    keep_cols = ["user_id"]
    keep_cols += ["user_" + col for col in numeric_cols]
    keep_cols += [
        "user_elite_count",
        "user_friend_count",
        "yelping_since_year"
    ]

    keep_cols = [c for c in keep_cols if c in u.columns]

    return u[keep_cols]


# %% [source cell 18]

# ============================================================
# 8. Tip Aggregation
# ============================================================

def aggregate_tip_features(tip):
    t = tip.copy()

    if "text" not in t.columns:
        t["text"] = ""

    t["text"] = t["text"].fillna("").astype(str)
    t["tip_text_len"] = t["text"].str.len().astype("float32")
    t["tip_word_count"] = t["text"].str.split().str.len().astype("float32")

    if "compliment_count" in t.columns:
        t["compliment_count"] = pd.to_numeric(
            t["compliment_count"], errors="coerce"
        ).fillna(0).astype("float32")
    else:
        t["compliment_count"] = 0.0

    tip_business = t.groupby("business_id", as_index=False).agg(
        business_tip_count=("text", "count"),
        business_tip_avg_len=("tip_text_len", "mean"),
        business_tip_avg_words=("tip_word_count", "mean"),
        business_tip_compliments=("compliment_count", "sum")
    )

    tip_user = t.groupby("user_id", as_index=False).agg(
        user_tip_count=("text", "count"),
        user_tip_avg_len=("tip_text_len", "mean"),
        user_tip_avg_words=("tip_word_count", "mean"),
        user_tip_compliments=("compliment_count", "sum")
    )

    for df in [tip_business, tip_user]:
        for c in df.columns:
            if c not in ["business_id", "user_id"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    return tip_business, tip_user


# %% [source cell 20]

# ============================================================
# 9. Build Feature Tables
# ============================================================

t0 = time.time()

business_clean = clean_business_features(business)
user_clean = clean_user_features(user)
tip_business, tip_user = aggregate_tip_features(tip)

print("business_clean:", business_clean.shape)
print("user_clean:", user_clean.shape)
print("tip_business:", tip_business.shape)
print("tip_user:", tip_user.shape)

train_fe = add_review_features(train_x)
test_fe = add_review_features(test_x)

print("train_fe before merge:", train_fe.shape)
print("test_fe before merge:", test_fe.shape)

train_fe = merge_with_checks(train_fe, business_clean, on="business_id", how="left", label="train + business")
test_fe = merge_with_checks(test_fe, business_clean, on="business_id", how="left", label="test + business")

train_fe = merge_with_checks(train_fe, user_clean, on="user_id", how="left", label="train + user")
test_fe = merge_with_checks(test_fe, user_clean, on="user_id", how="left", label="test + user")

train_fe = merge_with_checks(train_fe, tip_business, on="business_id", how="left", label="train + business tips")
test_fe = merge_with_checks(test_fe, tip_business, on="business_id", how="left", label="test + business tips")

train_fe = merge_with_checks(train_fe, tip_user, on="user_id", how="left", label="train + user tips")
test_fe = merge_with_checks(test_fe, tip_user, on="user_id", how="left", label="test + user tips")

print("train_fe after merge:", train_fe.shape)
print("test_fe after merge:", test_fe.shape)
print("Feature table time:", (time.time() - t0) / 60, "minutes")


# %% [source cell 22]

# ============================================================
# 10. Interaction, Ratio, Log Features
# ============================================================

for df in [train_fe, test_fe]:

    df["star_diff_from_business"] = df["review_stars"] - df["business_stars"]
    df["user_star_diff"] = df["review_stars"] - df["user_average_stars"]

    df["abs_star_diff_from_business"] = df["star_diff_from_business"].abs()
    df["abs_user_star_diff"] = df["user_star_diff"].abs()

    df["log_user_review_count"] = np.log1p(df["user_review_count"].fillna(0).clip(lower=0))
    df["log_user_fans"] = np.log1p(df["user_fans"].fillna(0).clip(lower=0))
    df["log_user_friend_count"] = np.log1p(df["user_friend_count"].fillna(0).clip(lower=0))
    df["log_user_elite_count"] = np.log1p(df["user_elite_count"].fillna(0).clip(lower=0))

    if USE_USER_USEFUL_FEATURES and "user_useful" in df.columns:
        df["log_user_useful"] = np.log1p(df["user_useful"].fillna(0).clip(lower=0))
    else:
        df["user_useful"] = 0.0
        df["log_user_useful"] = 0.0

    if "user_funny" not in df.columns:
        df["user_funny"] = 0.0

    if "user_cool" not in df.columns:
        df["user_cool"] = 0.0

    df["log_user_funny"] = np.log1p(df["user_funny"].fillna(0).clip(lower=0))
    df["log_user_cool"] = np.log1p(df["user_cool"].fillna(0).clip(lower=0))

    df["user_total_feedback"] = df["user_useful"] + df["user_funny"] + df["user_cool"]
    df["log_user_total_feedback"] = np.log1p(df["user_total_feedback"].fillna(0).clip(lower=0))

    df["user_feedback_per_review"] = df["user_total_feedback"] / (df["user_review_count"] + 1)
    df["user_useful_per_review"] = df["user_useful"] / (df["user_review_count"] + 1)
    df["user_funny_per_review"] = df["user_funny"] / (df["user_review_count"] + 1)
    df["user_cool_per_review"] = df["user_cool"] / (df["user_review_count"] + 1)
    df["user_fans_per_review"] = df["user_fans"] / (df["user_review_count"] + 1)

    df["user_useful_share"] = df["user_useful"] / (df["user_total_feedback"] + 1)
    df["user_funny_share"] = df["user_funny"] / (df["user_total_feedback"] + 1)
    df["user_cool_share"] = df["user_cool"] / (df["user_total_feedback"] + 1)

    df["review_len_x_user_review_count"] = df["log_text_len"] * df["log_user_review_count"]
    df["review_len_x_user_useful"] = df["log_text_len"] * df["user_useful_per_review"]
    df["log_text_x_log_user_useful"] = df["log_text_len"] * df["log_user_useful"]
    df["log_text_x_user_feedback_per_review"] = df["log_text_len"] * df["user_feedback_per_review"]
    df["log_word_x_log_user_useful"] = df["log_word_count"] * df["log_user_useful"]

    df["business_popularity_x_star_diff"] = df["log_business_review_count"] * df["star_diff_from_business"]
    df["business_popularity_x_text_len"] = df["log_business_review_count"] * df["log_text_len"]
    df["log_text_x_log_business_reviews"] = df["log_text_len"] * df["log_business_review_count"]
    df["log_word_x_log_business_reviews"] = df["log_word_count"] * df["log_business_review_count"]

    df["negative_long_review"] = df["is_negative_star"] * df["log_text_len"]
    df["positive_long_review"] = df["is_positive_star"] * df["log_text_len"]
    df["extreme_long_review"] = df["is_extreme_star"] * df["log_text_len"]

    df["log_business_tip_count"] = np.log1p(df["business_tip_count"].fillna(0).clip(lower=0))
    df["log_user_tip_count"] = np.log1p(df["user_tip_count"].fillna(0).clip(lower=0))

    df["business_tip_x_text_len"] = df["log_business_tip_count"] * df["log_text_len"]
    df["user_tip_x_text_len"] = df["log_user_tip_count"] * df["log_text_len"]

warn_duplicate_columns(train_fe, "train_fe after interaction features")
warn_duplicate_columns(test_fe, "test_fe after interaction features")
train_fe = train_fe.loc[:, ~train_fe.columns.duplicated()]
test_fe = test_fe.loc[:, ~test_fe.columns.duplicated()]

print("train_fe after interaction:", train_fe.shape)
print("test_fe after interaction:", test_fe.shape)

gc.collect()


# %% [source cell 24]

# 11. Out-of-Fold Target Encoding

def add_oof_target_encoding(
        train_df,
        test_df,
        y,
        cols,
        n_splits=5,
        smoothing=30,
        random_state=42
):
    train_df = train_df.copy()
    test_df = test_df.copy()
    y = pd.Series(y).reset_index(drop=True)

    global_mean = y.mean()

    if isinstance(cols, str):
        cols = [cols]

    enc_name = "te_" + "_".join(cols)

    key_train = train_df[cols].astype(str).agg("_".join, axis=1)
    key_test = test_df[cols].astype(str).agg("_".join, axis=1)

    oof_values = np.zeros(len(train_df), dtype="float32")

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    for tr_idx, va_idx in skf.split(train_df, y):
        fold_key_train = key_train.iloc[tr_idx]
        fold_y_train = y.iloc[tr_idx]

        stats = pd.DataFrame({
            "key": fold_key_train.values,
            "target": fold_y_train.values
        }).groupby("key")["target"].agg(["mean", "count"])

        stats["smooth"] = (
                                  stats["mean"] * stats["count"] + global_mean * smoothing
                          ) / (stats["count"] + smoothing)

        val_keys = key_train.iloc[va_idx]
        oof_values[va_idx] = val_keys.map(stats["smooth"]).fillna(global_mean).astype("float32")

    full_stats = pd.DataFrame({
        "key": key_train.values,
        "target": y.values
    }).groupby("key")["target"].agg(["mean", "count"])

    full_stats["smooth"] = (
                                   full_stats["mean"] * full_stats["count"] + global_mean * smoothing
                           ) / (full_stats["count"] + smoothing)

    train_df[enc_name] = oof_values
    test_df[enc_name] = key_test.map(full_stats["smooth"]).fillna(global_mean).astype("float32")

    print("Added target encoding:", enc_name)

    return train_df, test_df


y_series = pd.Series(y).reset_index(drop=True).astype(int)
train_fe = train_fe.reset_index(drop=True)
test_fe = test_fe.reset_index(drop=True)

single_te_cols = [
    "user_id",
    "business_id",
    "review_year",
    "review_stars"
]

for col in single_te_cols:
    if col in train_fe.columns and col in test_fe.columns:
        train_fe, test_fe = add_oof_target_encoding(
            train_fe,
            test_fe,
            y_series,
            cols=[col],
            n_splits=5,
            smoothing=30,
            random_state=RANDOM_STATE
        )

combo_te_cols = [
    ["business_id", "review_stars"],
    ["user_id", "review_stars"],
    ["business_id", "review_year"],
    ["user_id", "review_year"],
    ["review_year", "review_stars"]
]

for cols in combo_te_cols:
    if all(c in train_fe.columns for c in cols) and all(c in test_fe.columns for c in cols):
        train_fe, test_fe = add_oof_target_encoding(
            train_fe,
            test_fe,
            y_series,
            cols=cols,
            n_splits=5,
            smoothing=50,
            random_state=RANDOM_STATE
        )

for df in [train_fe, test_fe]:

    if "te_user_id" in df.columns:
        df["te_user_x_log_text_len"] = df["te_user_id"] * df["log_text_len"]

    if "te_business_id" in df.columns:
        df["te_business_x_log_text_len"] = df["te_business_id"] * df["log_text_len"]

    if "te_user_id" in df.columns and "te_business_id" in df.columns:
        df["te_user_minus_business"] = df["te_user_id"] - df["te_business_id"]
        df["te_user_x_business"] = df["te_user_id"] * df["te_business_id"]

    if "te_business_id_review_stars" in df.columns:
        df["te_business_star_x_log_text"] = df["te_business_id_review_stars"] * df["log_text_len"]

    if "te_user_id_review_stars" in df.columns:
        df["te_user_star_x_log_text"] = df["te_user_id_review_stars"] * df["log_text_len"]

te_cols = [c for c in train_fe.columns if c.startswith("te_")]

print("Number of TE columns:", len(te_cols))
print("train_fe after TE:", train_fe.shape)
print("test_fe after TE:", test_fe.shape)

gc.collect()


# %% [source cell 26]

# 12. Select Numeric Features

drop_cols = [
    "review_id",
    "user_id",
    "business_id",
    "text",
    "date",
    "stars"
]

feature_cols = [c for c in train_fe.columns if c not in drop_cols]

X = train_fe[feature_cols].select_dtypes(include=["number", "bool"]).copy()
X_test_final = test_fe[X.columns].copy()

X = X.replace([np.inf, -np.inf], np.nan)
X_test_final = X_test_final.replace([np.inf, -np.inf], np.nan)

X = X.astype("float32")
X_test_final = X_test_final.astype("float32")

print("Number of features:", X.shape[1])
print("X:", X.shape)
print("X_test_final:", X_test_final.shape)

feature_list_path = os.path.join(OUTPUT_DIR, "super_final_feature_list.csv")
pd.DataFrame({"feature": X.columns}).to_csv(feature_list_path, index=False)
print("Saved feature list:", feature_list_path)

feature_list_json_path = os.path.join(OUTPUT_DIR, "super_final_feature_list.json")
with open(feature_list_json_path, "w", encoding="utf-8") as f:
    json.dump(list(X.columns), f, indent=2)
print("Saved JSON feature list:", feature_list_json_path)


# %% [source cell 28]

# 13. Train / Validation Split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

if "review_id" in train_fe.columns:
    train_review_ids_all = train_fe["review_id"].reset_index(drop=True)
    train_split_review_ids = train_review_ids_all.loc[X_train.index].reset_index(drop=True)
    val_review_ids = train_review_ids_all.loc[X_val.index].reset_index(drop=True)
else:
    print("WARNING: review_id missing from train_fe; using original row index for validation alignment checks.")
    train_review_ids_all = pd.Series(np.arange(len(X)), name="review_id")
    train_split_review_ids = pd.Series(X_train.index.to_numpy(), name="review_id").reset_index(drop=True)
    val_review_ids = pd.Series(X_val.index.to_numpy(), name="review_id").reset_index(drop=True)

if "review_id" in test_fe.columns:
    test_review_ids = test_fe["review_id"].reset_index(drop=True)
else:
    print("WARNING: review_id missing from test_fe; using test row index for score diagnostics.")
    test_review_ids = pd.Series(np.arange(len(X_test_final)), name="review_id")

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("y_train positive rate:", y_train.mean())
print("y_val positive rate:", y_val.mean())
print("Validation review_id count:", len(val_review_ids))
print("Validation review_id unique count:", val_review_ids.nunique())

gc.collect()


# %% [source cell 34]

# ============================================================
# 16. Score Bank for Model Zoo
# ============================================================

n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
scale_pos_weight = n_neg / n_pos

print("Positive:", n_pos)
print("Negative:", n_neg)
print("scale_pos_weight:", scale_pos_weight)

score_bank_val = {}
score_bank_test = {}
score_bank_val_review_ids = {}
model_metrics = []


def add_model_scores(name, val_score, test_score):
    val_score = np.asarray(val_score).astype("float32")
    test_score = np.asarray(test_score).astype("float32")

    if len(val_score) != len(val_review_ids):
        raise ValueError(f"{name} validation score length does not match val_review_ids.")
    if len(test_score) != len(test_review_ids):
        raise ValueError(f"{name} test score length does not match test_review_ids.")

    score_bank_val[name] = val_score
    score_bank_test[name] = test_score
    score_bank_val_review_ids[name] = val_review_ids.copy()

    val_pred_path = os.path.join(OUTPUT_DIR, f"validation_predictions_{name}.csv")
    pd.DataFrame({
        "review_id": val_review_ids,
        "probability": val_score
    }).to_csv(val_pred_path, index=False)
    print("Saved validation probabilities:", val_pred_path)

    metrics = best_tpr_under_fpr(
        y_val,
        val_score,
        fpr_limit=FPR_LIMIT_FINAL
    )

    row = {
        "model": name,
        "auc": metrics["auc"],
        "fpr": metrics["fpr"],
        "tpr": metrics["tpr"],
        "threshold": metrics["threshold"]
    }

    model_metrics.append(row)

    print(
        f"{name:35s} | "
        f"AUC={metrics['auc']:.6f} | "
        f"FPR={metrics['fpr']:.6f} | "
        f"TPR={metrics['tpr']:.6f}"
    )

    return metrics


# %% [source cell 36]

# ============================================================
# 17. Train LightGBM Model Zoo
# ============================================================

def train_lgb_add(name, params, seed=42):
    print("\n" + "=" * 80)
    print("Training LightGBM:", name)
    print("=" * 80)

    params = params.copy()
    n_estimators = params.pop("n_estimators")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        **params
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100)
        ]
    )

    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test_final)[:, 1]

    metrics = add_model_scores(name, val_score, test_score)

    gc.collect()

    return model, metrics


lgb_configs = [
    {
        "name": "lgb_main_seed42",
        "seed": 42,
        "params": {
            "n_estimators": 1700,
            "learning_rate": 0.03,
            "num_leaves": 127,
            "min_child_samples": 100,
            "subsample": 0.85,
            "colsample_bytree": 0.85
        }
    },
    {
        "name": "lgb_main_seed2024",
        "seed": 2024,
        "params": {
            "n_estimators": 1700,
            "learning_rate": 0.03,
            "num_leaves": 127,
            "min_child_samples": 100,
            "subsample": 0.85,
            "colsample_bytree": 0.85
        }
    },
    {
        "name": "lgb_leaves63_lr003",
        "seed": 42,
        "params": {
            "n_estimators": 1700,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 80,
            "subsample": 0.85,
            "colsample_bytree": 0.85
        }
    },
    {
        "name": "lgb_leaves191_reg",
        "seed": 77,
        "params": {
            "n_estimators": 2200,
            "learning_rate": 0.025,
            "num_leaves": 191,
            "min_child_samples": 180,
            "subsample": 0.80,
            "colsample_bytree": 0.75,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0
        }
    },
    {
        "name": "lgb_leaves255_safe",
        "seed": 123,
        "params": {
            "n_estimators": 2400,
            "learning_rate": 0.02,
            "num_leaves": 255,
            "min_child_samples": 220,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "reg_alpha": 2.0,
            "reg_lambda": 8.0
        }
    },
    {
        "name": "lgb_small_leaves",
        "seed": 314,
        "params": {
            "n_estimators": 1600,
            "learning_rate": 0.035,
            "num_leaves": 31,
            "min_child_samples": 60,
            "subsample": 0.90,
            "colsample_bytree": 0.90
        }
    },
    {
        "name": "lgb_more_colsample",
        "seed": 888,
        "params": {
            "n_estimators": 1900,
            "learning_rate": 0.025,
            "num_leaves": 127,
            "min_child_samples": 120,
            "subsample": 0.90,
            "colsample_bytree": 0.95,
            "reg_lambda": 2.0
        }
    }
]

lgb_models = {}

if RUN_EXTRA_LGB:
    for cfg in lgb_configs:
        model, metrics = train_lgb_add(
            name=cfg["name"],
            params=cfg["params"],
            seed=cfg["seed"]
        )
        lgb_models[cfg["name"]] = model
else:
    model, metrics = train_lgb_add(
        name=lgb_configs[0]["name"],
        params=lgb_configs[0]["params"],
        seed=lgb_configs[0]["seed"]
    )
    lgb_models[lgb_configs[0]["name"]] = model


# %% [source cell 38]

# ============================================================
# 18. Train XGBoost Model Zoo
# ============================================================

def train_xgb_add(name, params, seed=42):
    print("\n" + "=" * 80)
    print("Training XGBoost:", name)
    print("=" * 80)

    params = params.copy()

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": -1
    }

    if USE_XGB_GPU:
        base_params["device"] = "cuda"

    model = xgb.XGBClassifier(
        **base_params,
        **params
    )

    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )
    except Exception as e:
        print("XGBoost GPU or parameter issue. Retrying on CPU.")
        print("Original error:", e)

        base_params.pop("device", None)

        model = xgb.XGBClassifier(
            **base_params,
            **params
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )

    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test_final)[:, 1]

    metrics = add_model_scores(name, val_score, test_score)

    gc.collect()

    return model, metrics


xgb_configs = [
    {
        "name": "xgb_depth8_main_seed42",
        "seed": 42,
        "params": {
            "n_estimators": 1300,
            "learning_rate": 0.03,
            "max_depth": 8,
            "min_child_weight": 100,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0
        }
    },
    {
        "name": "xgb_depth8_reg_seed2024",
        "seed": 2024,
        "params": {
            "n_estimators": 1500,
            "learning_rate": 0.025,
            "max_depth": 8,
            "min_child_weight": 120,
            "subsample": 0.90,
            "colsample_bytree": 0.80,
            "reg_alpha": 0.5,
            "reg_lambda": 3.0
        }
    },
    {
        "name": "xgb_depth6_safe",
        "seed": 77,
        "params": {
            "n_estimators": 1300,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 150,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0
        }
    },
    {
        "name": "xgb_depth5_more_reg",
        "seed": 123,
        "params": {
            "n_estimators": 1200,
            "learning_rate": 0.035,
            "max_depth": 5,
            "min_child_weight": 200,
            "subsample": 0.80,
            "colsample_bytree": 0.75,
            "reg_alpha": 2.0,
            "reg_lambda": 8.0
        }
    },
    {
        "name": "xgb_depth9_large",
        "seed": 314,
        "params": {
            "n_estimators": 1100,
            "learning_rate": 0.025,
            "max_depth": 9,
            "min_child_weight": 150,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0
        }
    }
]

xgb_models = {}

if RUN_EXTRA_XGB:
    for cfg in xgb_configs:
        model, metrics = train_xgb_add(
            name=cfg["name"],
            params=cfg["params"],
            seed=cfg["seed"]
        )
        xgb_models[cfg["name"]] = model
else:
    model, metrics = train_xgb_add(
        name=xgb_configs[0]["name"],
        params=xgb_configs[0]["params"],
        seed=xgb_configs[0]["seed"]
    )
    xgb_models[xgb_configs[0]["name"]] = model


# %% [source cell 40]

# ============================================================
# 19. Train MLP Ensemble and Individual Seeds
# ============================================================

mlp_val_scores = []
mlp_test_scores = []
mlp_single_results = []

if RUN_EXTRA_MLP:
    for seed in MLP_SEEDS:
        print("\n" + "=" * 80)
        print(f"Training MLP, random_state={seed}")
        print("=" * 80)

        mlp_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                alpha=0.001,
                learning_rate_init=0.0005,
                max_iter=140,
                early_stopping=True,
                validation_fraction=0.10,
                random_state=seed
            ))
        ])

        mlp_model.fit(X_train, y_train)

        val_score = mlp_model.predict_proba(X_val)[:, 1]
        test_score = mlp_model.predict_proba(X_test_final)[:, 1]

        mlp_val_scores.append(val_score.astype("float32"))
        mlp_test_scores.append(test_score.astype("float32"))

        metrics = add_model_scores(f"mlp_seed_{seed}", val_score, test_score)

        mlp_single_results.append({
            "seed": seed,
            "auc": metrics["auc"],
            "fpr": metrics["fpr"],
            "tpr": metrics["tpr"],
            "threshold": metrics["threshold"]
        })

        gc.collect()

    mlp_val_score = np.mean(mlp_val_scores, axis=0)
    mlp_test_score = np.mean(mlp_test_scores, axis=0)

    mlp_ensemble_metrics = add_model_scores(
        "mlp_ensemble_all_seeds",
        mlp_val_score,
        mlp_test_score
    )

    mlp_single_results_df = pd.DataFrame(mlp_single_results)
    mlp_single_path = os.path.join(OUTPUT_DIR, "super_mlp_single_seed_results.csv")
    mlp_single_results_df.to_csv(mlp_single_path, index=False)
    print("Saved:", mlp_single_path)


# %% [source cell 42]

# ============================================================
# 20. Model Zoo Summary
# ============================================================

model_metrics_df = pd.DataFrame(model_metrics)

model_metrics_df = model_metrics_df.sort_values(
    ["tpr", "auc"],
    ascending=False
).reset_index(drop=True)

model_zoo_path = os.path.join(OUTPUT_DIR, "super_model_zoo_results.csv")
model_metrics_df.to_csv(model_zoo_path, index=False)

print("\nModel zoo results:")
print(model_metrics_df)
print("Saved:", model_zoo_path)


def assert_validation_prediction_alignment(model_names):
    """Confirm model validation probabilities share the exact same review_id order."""
    reference_ids = val_review_ids.reset_index(drop=True).astype(str)
    for model_name in model_names:
        ids = score_bank_val_review_ids[model_name].reset_index(drop=True).astype(str)
        if not ids.equals(reference_ids):
            mismatch = np.where(ids.to_numpy() != reference_ids.to_numpy())[0]
            first_bad = int(mismatch[0]) if len(mismatch) else None
            raise ValueError(
                f"Validation predictions are not aligned for {model_name}; "
                f"first mismatch position={first_bad}"
            )
        if len(score_bank_val[model_name]) != len(reference_ids):
            raise ValueError(f"Validation probabilities for {model_name} have the wrong length.")
    print("Validation prediction alignment confirmed for models:", model_names)


# %% [source cell 44]

# ============================================================
# 21. Advanced Top-K Blend Search
# ============================================================

def evaluate_blend_candidate(y_true, score, fpr_limit):
    return best_tpr_under_fpr(y_true, score, fpr_limit=fpr_limit)


def advanced_topk_blend_search(
        model_metrics_df,
        score_bank_val,
        score_bank_test,
        y_val,
        top_k=12,
        search_sample_size=200000,
        random_trials=30000,
        top_eval=300,
        random_state=42,
        fpr_limit=0.097
):
    top_k = min(top_k, len(model_metrics_df))

    top_model_names = model_metrics_df.head(top_k)["model"].tolist()

    print("\nTop models used for blend:")
    for name in top_model_names:
        print(name)

    assert_validation_prediction_alignment(top_model_names)

    val_matrix = np.column_stack([score_bank_val[name] for name in top_model_names]).astype("float32")
    test_matrix = np.column_stack([score_bank_test[name] for name in top_model_names]).astype("float32")

    rng = np.random.default_rng(random_state)

    # sample validation rows for fast blend search
    if search_sample_size < len(y_val):
        sample_idx, _ = train_test_split(
            np.arange(len(y_val)),
            train_size=search_sample_size,
            random_state=random_state,
            stratify=y_val
        )
    else:
        sample_idx = np.arange(len(y_val))

    y_sub = pd.Series(y_val).reset_index(drop=True).iloc[sample_idx].values
    val_sub = val_matrix[sample_idx, :]

    candidates = []

    # include single models
    for i, name in enumerate(top_model_names):
        w = np.zeros(top_k)
        w[i] = 1.0

        score_sub = val_sub @ w
        metrics = evaluate_blend_candidate(y_sub, score_sub, fpr_limit)

        candidates.append({
            "stage": "single",
            "auc_sub": metrics["auc"],
            "fpr_sub": metrics["fpr"],
            "tpr_sub": metrics["tpr"],
            "threshold_sub": metrics["threshold"],
            **{f"w_{top_model_names[j]}": w[j] for j in range(top_k)}
        })

    # equal weight
    w_equal = np.ones(top_k) / top_k
    score_sub = val_sub @ w_equal
    metrics = evaluate_blend_candidate(y_sub, score_sub, fpr_limit)

    candidates.append({
        "stage": "equal",
        "auc_sub": metrics["auc"],
        "fpr_sub": metrics["fpr"],
        "tpr_sub": metrics["tpr"],
        "threshold_sub": metrics["threshold"],
        **{f"w_{top_model_names[j]}": w_equal[j] for j in range(top_k)}
    })

    # random Dirichlet blends
    alphas = [0.15, 0.25, 0.5, 0.8, 1.0, 2.0]

    trials_per_alpha = max(1, random_trials // len(alphas))

    for alpha in alphas:
        print("Random blend search alpha:", alpha)

        for _ in range(trials_per_alpha):
            w = rng.dirichlet(np.ones(top_k) * alpha)

            score_sub = val_sub @ w
            metrics = evaluate_blend_candidate(y_sub, score_sub, fpr_limit)

            candidates.append({
                "stage": f"dirichlet_{alpha}",
                "auc_sub": metrics["auc"],
                "fpr_sub": metrics["fpr"],
                "tpr_sub": metrics["tpr"],
                "threshold_sub": metrics["threshold"],
                **{f"w_{top_model_names[j]}": w[j] for j in range(top_k)}
            })

    candidates_df = pd.DataFrame(candidates)

    candidates_df = candidates_df.sort_values(
        ["tpr_sub", "auc_sub"],
        ascending=False
    ).reset_index(drop=True)

    # evaluate top candidates on full validation
    full_results = []

    top_candidates = candidates_df.head(top_eval)

    for idx, row in top_candidates.iterrows():
        w = np.array([row[f"w_{name}"] for name in top_model_names])
        score_full = val_matrix @ w

        metrics = evaluate_blend_candidate(y_val, score_full, fpr_limit)

        full_results.append({
            "rank_from_sub": idx,
            "stage": row["stage"],
            "auc": metrics["auc"],
            "fpr": metrics["fpr"],
            "tpr": metrics["tpr"],
            "threshold": metrics["threshold"],
            **{f"w_{top_model_names[j]}": w[j] for j in range(top_k)}
        })

    full_results_df = pd.DataFrame(full_results)

    full_results_df = full_results_df.sort_values(
        ["tpr", "auc"],
        ascending=False
    ).reset_index(drop=True)

    return top_model_names, val_matrix, test_matrix, candidates_df, full_results_df


top_model_names, val_matrix, test_matrix, blend_candidates_df, advanced_blend_df = advanced_topk_blend_search(
    model_metrics_df=model_metrics_df,
    score_bank_val=score_bank_val,
    score_bank_test=score_bank_test,
    y_val=y_val,
    top_k=TOP_K_FOR_BLEND,
    search_sample_size=BLEND_SEARCH_SAMPLE_SIZE,
    random_trials=RANDOM_BLEND_TRIALS,
    top_eval=RANDOM_BLEND_TOP_EVAL,
    random_state=RANDOM_STATE,
    fpr_limit=FPR_LIMIT_FINAL
)

blend_candidates_path = os.path.join(OUTPUT_DIR, "super_blend_candidates_sample_search.csv")
advanced_blend_path = os.path.join(OUTPUT_DIR, "super_advanced_blend_full_eval.csv")

blend_candidates_df.to_csv(blend_candidates_path, index=False)
advanced_blend_df.to_csv(advanced_blend_path, index=False)

print("\nTop 20 advanced blend full-validation results:")
print(advanced_blend_df.head(20))

print("Saved sample blend candidates:", blend_candidates_path)
print("Saved full eval blend results:", advanced_blend_path)


# %% [source cell 46]

# ============================================================
# 22. Local Blend Refinement Around Best Weights
# ============================================================

def local_refine_blend(
        advanced_blend_df,
        top_model_names,
        val_matrix,
        y_val,
        n_rounds=5000,
        noise_scale=0.06,
        random_state=42,
        fpr_limit=0.097
):
    rng = np.random.default_rng(random_state)

    best_row = advanced_blend_df.iloc[0]
    best_w = np.array([best_row[f"w_{name}"] for name in top_model_names], dtype=float)

    refined_results = []

    for i in range(n_rounds):
        noise = rng.normal(0, noise_scale, size=len(best_w))
        w = best_w + noise
        w = np.clip(w, 0, None)

        if w.sum() == 0:
            continue

        w = w / w.sum()

        score = val_matrix @ w
        metrics = best_tpr_under_fpr(y_val, score, fpr_limit=fpr_limit)

        refined_results.append({
            "stage": "local_refine",
            "auc": metrics["auc"],
            "fpr": metrics["fpr"],
            "tpr": metrics["tpr"],
            "threshold": metrics["threshold"],
            **{f"w_{top_model_names[j]}": w[j] for j in range(len(top_model_names))}
        })

    refined_df = pd.DataFrame(refined_results)

    combined_df = pd.concat(
        [advanced_blend_df, refined_df],
        ignore_index=True
    )

    combined_df = combined_df.sort_values(
        ["tpr", "auc"],
        ascending=False
    ).reset_index(drop=True)

    return combined_df


advanced_blend_refined_df = local_refine_blend(
    advanced_blend_df=advanced_blend_df,
    top_model_names=top_model_names,
    val_matrix=val_matrix,
    y_val=y_val,
    n_rounds=5000,
    noise_scale=0.04,
    random_state=RANDOM_STATE + 999,
    fpr_limit=FPR_LIMIT_FINAL
)

advanced_refined_path = os.path.join(OUTPUT_DIR, "super_advanced_blend_refined_results.csv")
advanced_blend_refined_df.to_csv(advanced_refined_path, index=False)

print("\nTop 20 refined blend results:")
print(advanced_blend_refined_df.head(20))
print("Saved:", advanced_refined_path)


# %% [source cell 48]

# ============================================================
# 23. Export Final Super Blend Submission
# ============================================================

best_adv = advanced_blend_refined_df.iloc[0]

best_weights = np.array([
    best_adv[f"w_{name}"] for name in top_model_names
])

best_threshold = best_adv["threshold"]

super_val_score = val_matrix @ best_weights
super_test_score = test_matrix @ best_weights

if not np.isclose(best_weights.sum(), 1.0, atol=1e-5):
    print("WARNING: final blend weights do not sum to 1. Sum:", best_weights.sum())
if np.any(best_weights < 0):
    print("WARNING: final blend contains negative weights.")

super_metrics = best_tpr_under_fpr(
    y_val,
    super_val_score,
    fpr_limit=FPR_LIMIT_FINAL
)

print_metrics("SUPER ADVANCED BLEND VALIDATION METRICS", super_metrics)
print("\nThreshold selection summary at FPR <=", FPR_LIMIT_FINAL)
print(f"AUC: {super_metrics['auc']:.6f}")
print(f"Best threshold: {super_metrics['threshold']:.12f}")
print(f"FPR: {super_metrics['fpr']:.6f}")
print(f"TPR: {super_metrics['tpr']:.6f}")

super_pred = (super_test_score >= best_threshold).astype(int)

super_submission = pd.DataFrame({
    "top_useful": super_pred
})

super_submission_path = os.path.join(
    OUTPUT_DIR,
    "group_7_yelp.csv"
)

super_submission.to_csv(
    super_submission_path,
    index=False,
    header=False
)

super_score_path = os.path.join(
    OUTPUT_DIR,
    "super_advanced_blend_test_scores.csv"
)

pd.DataFrame({
    "review_id": test_review_ids,
    "final_score": super_test_score,
    "prediction": super_pred
}).to_csv(super_score_path, index=False)

super_val_score_path = os.path.join(
    OUTPUT_DIR,
    "super_advanced_blend_validation_scores.csv"
)

pd.DataFrame({
    "review_id": val_review_ids,
    "final_score": super_val_score,
    "top_useful": pd.Series(y_val).reset_index(drop=True)
}).to_csv(super_val_score_path, index=False)

print("\n" + "=" * 80)
print("SUPER ADVANCED BLEND SAVED")
print("=" * 80)

print("Submission:", super_submission_path)
print("Scores:", super_score_path)
print("Validation scores:", super_val_score_path)

print("\nSelected models and weights:")
for name, w in zip(top_model_names, best_weights):
    print(f"{name:35s}: {w:.6f}")

print("\nSuper validation result:")
print("AUC:", super_metrics["auc"])
print("FPR:", super_metrics["fpr"])
print("TPR:", super_metrics["tpr"])
print("Threshold:", best_threshold)

print("\nPositive predictions:", super_pred.sum())
print("Positive ratio:", super_pred.mean())

print("\nPrevious best:")
print("FPR:", CURRENT_BEST_FPR)
print("TPR:", CURRENT_BEST_TPR)

print("\nLeaderboard comparison for final report:")
print(f"Our final TPR: {REPORT_FINAL_TPR:.6f}")
print(f"First-place TPR: {FIRST_PLACE_TPR:.4f}")
print(f"Difference: {FIRST_PLACE_TPR - REPORT_FINAL_TPR:.6f}")

if super_metrics["fpr"] <= FPR_LIMIT and super_metrics["tpr"] > CURRENT_BEST_TPR:
    print("\nUSE NEW SUBMISSION:")
    print(super_submission_path)
else:
    print("\nKEEP OLD SUBMISSION unless leaderboard confirms improvement.")
    print("New validation did not beat previous TPR safely.")


# %% [source cell 50]

# ============================================================
# 24. Save Main Final Tables
# ============================================================

final_summary = {
    "previous_best_fpr": CURRENT_BEST_FPR,
    "previous_best_tpr": CURRENT_BEST_TPR,
    "super_fpr": super_metrics["fpr"],
    "super_tpr": super_metrics["tpr"],
    "super_auc": super_metrics["auc"],
    "super_threshold": best_threshold,
    "positive_predictions": int(super_pred.sum()),
    "positive_ratio": float(super_pred.mean())
}

final_summary_path = os.path.join(OUTPUT_DIR, "super_final_summary.csv")
pd.DataFrame([final_summary]).to_csv(final_summary_path, index=False)

print("Saved final summary:", final_summary_path)
print(pd.DataFrame([final_summary]))

print("\nIf the reproduced blend result changes, check these first:")
print("1. Row order: compare review_id order in validation prediction CSVs.")
print("2. Train/validation split: confirm RANDOM_STATE, stratify=y, and test_size=0.20.")
print("3. Feature list: compare super_final_feature_list.json.")
print("4. Missing-value handling: compare merge diagnostics and model imputers.")
print("5. Model seeds: confirm LGB/XGB/MLP seeds and RANDOM_STATE.")
print("6. Target encoding leakage: confirm OOF target encoding folds are unchanged.")
print("7. Probability alignment: confirm assert_validation_prediction_alignment passed.")
print("8. Blend weights: compare super_advanced_blend_refined_results.csv.")
print("9. Threshold logic: confirm FPR_LIMIT_FINAL and best_tpr_under_fpr are unchanged.")
