# Combined Yelp Top-Useful model script
# Sections:
# 1. Baseline Logistic Regression
# 2. Decision Tree
# 3. Random Forest
# 4. LightGBM
# 5. Business-only Logistic Regression
# 6. Holiday Logistic Regression
# 7. Top-K Blended Ensemble Search

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

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import holidays
except ImportError:
    holidays = None

# -----------------------------------------------------------------------------
# Settings and data helpers
# -----------------------------------------------------------------------------

RANDOM_STATE = 42
N_JOBS = -1
DATA_DIR = r"D:\BUDT 758C - Predictive analytics\project work"
OUTPUT_DIR = DATA_DIR
FPR_LIMIT = 0.10
FPR_LIMIT_FINAL = 0.097

# file names used across notebook sections
TRAIN_X_FILE = "review_train_x.csv"
TRAIN_Y_FILE = "review_train_y.csv"
TEST_X_FILE = "review_test_x.csv"
BUSINESS_FILE = "business_data_cleaned.csv"
USER_FILE = "user_clean.csv"
TIP_FILE = "tips_cleaned.csv"


def read_csv_from_data_dir(filename):
    path = os.path.join(DATA_DIR, filename)
    print("Reading:", path)
    return pd.read_csv(path)


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


def plot_roc_with_operating_point(y_true, y_score, metrics, output_dir, filename="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    plt.axvline(FPR_LIMIT, linestyle=":", color="red", label="10% FPR limit")
    plt.scatter(metrics["fpr"], metrics["tpr"], color="red", zorder=5, label=f"TPR = {metrics['tpr']:.4f} at FPR = {metrics['fpr']:.4f}")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.tight_layout()

    fig_path = os.path.join(output_dir, filename)
    plt.savefig(fig_path, dpi=200)
    plt.show()

    print("Saved ROC curve:", fig_path)
    return fig_path

# -----------------------------------------------------------------------------
# Feature engineering helpers from the notebooks
# -----------------------------------------------------------------------------

def add_us_holiday_features(df):
    df = df.copy()

    if "date" not in df.columns or holidays is None:
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
    df["period_count"] = df["text"].str.count(r"\. ").astype("float32")
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

    return df


def add_review_features_with_holidays(df):
    df = add_review_features(df)
    return add_us_holiday_features(df)


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

                s1 = parts[0].split(":" )
                s2 = parts[1].split(":" )
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


def clean_user_features(user):
    u = user.copy()
    u = u.drop_duplicates(subset=["user_id"], keep="first")

    numeric_cols = ["review_count", "funny", "cool", "fans", "average_stars"]
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
    keep_cols += ["user_elite_count", "user_friend_count", "yelping_since_year"]
    keep_cols = [c for c in keep_cols if c in u.columns]
    return u[keep_cols]


def aggregate_tip_features(tip):
    t = tip.copy()

    if "text" not in t.columns:
        t["text"] = ""

    t["text"] = t["text"].fillna("").astype(str)
    t["tip_text_len"] = t["text"].str.len().astype("float32")
    t["tip_word_count"] = t["text"].str.split().str.len().astype("float32")

    if "compliment_count" in t.columns:
        t["compliment_count"] = pd.to_numeric(t["compliment_count"], errors="coerce").fillna(0).astype("float32")
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


def add_oof_target_encoding(
        train_df,
        test_df,
        y,
        cols,
        n_splits=5,
        smoothing=30,
        random_state=42
):
    y_arr = np.asarray(y, dtype="float32")
    global_mean = y_arr.mean()

    if isinstance(cols, str):
        cols = [cols]

    enc_name = "te_" + "_".join(cols)

    train_key_tuple = train_df[cols].apply(lambda x: tuple(x), axis=1)
    test_key_tuple = test_df[cols].apply(lambda x: tuple(x), axis=1)

    key_train, _ = pd.factorize(train_key_tuple)
    key_test, _ = pd.factorize(test_key_tuple)

    oof_values = np.full(len(train_df), global_mean, dtype="float32")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for tr_idx, va_idx in skf.split(train_df, y_arr):
        fold_key_train = key_train[tr_idx]
        fold_y_train = y_arr[tr_idx]

        fold_df = pd.DataFrame({"key": fold_key_train, "target": fold_y_train})
        stats = fold_df.groupby("key")["target"].agg(["mean", "size"])

        smoothed = (stats["mean"] * stats["size"] + global_mean * smoothing) / (stats["size"] + smoothing)
        stats_dict = smoothed.to_dict()

        val_keys = key_train[va_idx]
        for unique_key in np.unique(val_keys):
            if unique_key in stats_dict:
                mask = val_keys == unique_key
                oof_values[va_idx[mask]] = stats_dict[unique_key]

    full_df = pd.DataFrame({"key": key_train, "target": y_arr})
    full_stats = full_df.groupby("key")["target"].agg(["mean", "size"])
    full_smoothed = (full_stats["mean"] * full_stats["size"] + global_mean * smoothing) / (full_stats["size"] + smoothing)
    full_stats_dict = full_smoothed.to_dict()

    train_df[enc_name] = oof_values
    test_encoding = np.array([full_stats_dict.get(k, global_mean) for k in key_test], dtype="float32")
    test_df[enc_name] = test_encoding

    print("Added target encoding:", enc_name)
    return train_df, test_df

# -----------------------------------------------------------------------------
# Model sections
# -----------------------------------------------------------------------------

def run_baseline_logistic_regression(train_x, train_y, test_x, business, user, tip, train_indices, val_indices):
    print("\n" + "=" * 80)
    print("Section 1: Baseline Logistic Regression")
    print("=" * 80)

    business_clean = clean_business_features(business)
    user_clean = clean_user_features(user)
    tip_business, tip_user = aggregate_tip_features(tip)

    train_fe = add_review_features(train_x)
    test_fe = add_review_features(test_x)

    train_fe = train_fe.merge(business_clean, on="business_id", how="left")
    test_fe = test_fe.merge(business_clean, on="business_id", how="left")

    train_fe = train_fe.merge(user_clean, on="user_id", how="left")
    test_fe = test_fe.merge(user_clean, on="user_id", how="left")

    train_fe = train_fe.merge(tip_business, on="business_id", how="left")
    test_fe = test_fe.merge(tip_business, on="business_id", how="left")

    train_fe = train_fe.merge(tip_user, on="user_id", how="left")
    test_fe = test_fe.merge(tip_user, on="user_id", how="left")

    train_fe = train_fe.reset_index(drop=True)
    test_fe = test_fe.reset_index(drop=True)

    y_series = pd.Series(train_y).reset_index(drop=True).astype(int)

    for col in ["user_id", "business_id"]:
        if col in train_fe.columns and col in test_fe.columns:
            train_fe[col] = train_fe[col].astype(str)
            test_fe[col] = test_fe[col].astype(str)

    te_columns = ["user_id", "business_id", "review_year", "review_stars"]
    for col in te_columns:
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
    X_test_final = test_fe.reindex(columns=X.columns, fill_value=0).copy()
    X = X.replace([np.inf, -np.inf], np.nan).astype("float32")
    X_test_final = X_test_final.replace([np.inf, -np.inf], np.nan).astype("float32")

    X_train = X.iloc[train_indices].copy()
    X_val = X.iloc[val_indices].copy()
    y_train = y_series.iloc[train_indices].copy()
    y_val = y_series.iloc[val_indices].copy()

    baseline_logreg_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0,
            penalty="l2",
            class_weight="balanced",
            solver="saga",
            max_iter=1000,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE
        ))
    ])

    print("Training baseline logistic regression...")
    baseline_logreg_model.fit(X_train, y_train)

    baseline_val_score = baseline_logreg_model.predict_proba(X_val)[:, 1]
    baseline_test_score = baseline_logreg_model.predict_proba(X_test_final)[:, 1]

    baseline_metrics = best_tpr_under_fpr(y_val, baseline_val_score, fpr_limit=FPR_LIMIT)
    print_metrics("BASELINE LOGISTIC REGRESSION VALIDATION METRICS", baseline_metrics)

    plot_roc_with_operating_point(
        y_true=y_val,
        y_score=baseline_val_score,
        metrics=baseline_metrics,
        output_dir=OUTPUT_DIR,
        filename="baseline_logreg_roc_curve.png"
    )

    return {
        "model_name": "Baseline Logistic Regression",
        "val_score": baseline_val_score,
        "test_score": baseline_test_score,
        "metrics": baseline_metrics
    }


def run_decision_tree(train_x, train_y, business_data, train_indices, val_indices):
    print("\n" + "=" * 80)
    print("Section 2: Decision Tree")
    print("=" * 80)

    review_train_x = train_x.copy()
    review_train_y = pd.DataFrame({"top_useful": train_y})

    review_train_x["data_source"] = "train"
    combined_reviews = review_train_x.copy()

    numeric_cols = ["business_id"] + business_data.select_dtypes(include=[np.number]).columns.tolist()
    business_data_numeric = business_data[numeric_cols].copy()

    merged_data = combined_reviews.merge(business_data_numeric, on="business_id", how="inner")
    merged_data = merged_data.reset_index(drop=True)

    train_data = merged_data.copy()
    feature_cols = [col for col in business_data_numeric.columns if col in train_data.columns and col != "business_id"]

    X_train_tree = train_data.loc[train_indices, feature_cols].fillna(0).copy()
    X_val_tree = train_data.loc[val_indices, feature_cols].fillna(0).copy()
    y_train_tree = review_train_y.iloc[train_indices]["top_useful"].astype(int).copy()
    y_val_tree = review_train_y.iloc[val_indices]["top_useful"].astype(int).copy()

    dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=None, min_samples_split=10)
    dt_model.fit(X_train_tree, y_train_tree)

    dt_val_score = dt_model.predict_proba(X_val_tree)[:, 1]
    dt_test_score = np.zeros(0)

    dt_metrics = best_tpr_under_fpr(y_val_tree, dt_val_score, fpr_limit=FPR_LIMIT)
    print_metrics("DECISION TREE VALIDATION METRICS", dt_metrics)

    fpr, tpr, thresholds = roc_curve(y_val_tree, dt_val_score)
    idx = np.argmin(np.abs(fpr - 0.10))
    tpr_at_10 = tpr[idx]
    print(f"TPR@10%FPR: {tpr_at_10:.4f}")

    return {
        "model_name": "Decision Tree",
        "val_score": dt_val_score,
        "test_score": dt_test_score,
        "metrics": dt_metrics
    }


def run_random_forest(train_x, train_y, business, user, tip, test_x, train_indices, val_indices):
    print("\n" + "=" * 80)
    print("Section 3: Random Forest")
    print("=" * 80)

    train = pd.concat([train_x, pd.DataFrame({"top_useful": train_y})], axis=1)
    train = train.merge(user, on="user_id", how="left")
    train = train.merge(business, on="business_id", how="left")

    tip_biz = tip.groupby('business_id').agg(
        tip_count=('tip_text', 'count'),
        tip_avg_compliments=('tip_compliment', 'mean')
    ).reset_index()

    train = train.merge(tip_biz, on='business_id', how='left')
    test_x = test_x.merge(user, on="user_id", how="left")
    test_x = test_x.merge(business, on="business_id", how="left")
    test_x = test_x.merge(tip_biz, on='business_id', how='left')

    train[['tip_count', 'tip_avg_compliments']] = train[['tip_count', 'tip_avg_compliments']].fillna(0)
    test_x[['tip_count', 'tip_avg_compliments']] = test_x[['tip_count', 'tip_avg_compliments']].fillna(0)

    drop_cols = [
        'review_id', 'user_id', 'business_id', 'text', 'address',
        'city', 'state', 'postal_code', 'latitude', 'longitude',
        'date', 'top_useful'
    ]

    y = train['top_useful'].astype(int)
    X = train.drop(columns=drop_cols, errors='ignore')

    X_train = X.iloc[train_indices].copy()
    X_val = X.iloc[val_indices].copy()
    y_train = y.iloc[train_indices].copy()
    y_val = y.iloc[val_indices].copy()

    numeric_cols = X_train.select_dtypes(include=['number', 'bool']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )

    rf_model = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ])

    rf_model.fit(X_train, y_train)

    y_val_proba = rf_model.predict_proba(X_val)[:, 1]
    test_drop_cols = [
        'review_id', 'user_id', 'business_id', 'text', 'date', 'address',
        'city', 'state', 'postal_code', 'latitude', 'longitude', 'top_useful'
    ]

    X_test = test_x.drop(columns=test_drop_cols, errors='ignore')
    X_test = X_test.reindex(columns=X.columns, fill_value=np.nan)

    y_test_proba = rf_model.predict_proba(X_test)[:, 1]

    rf_metrics = best_tpr_under_fpr(y_val, y_val_proba, fpr_limit=FPR_LIMIT)
    print_metrics("RANDOM FOREST VALIDATION METRICS", rf_metrics)

    return {
        "model_name": "Random Forest",
        "val_score": y_val_proba,
        "test_score": y_test_proba,
        "metrics": rf_metrics
    }


def run_lightgbm(train_x, train_y, business, user, tip, test_x, train_indices, val_indices):
    print("\n" + "=" * 80)
    print("Section 4: LightGBM")
    print("=" * 80)

    if lgb is None:
        print("LightGBM is not installed. Skipping LightGBM section.")
        return {
            "model_name": "LightGBM",
            "val_score": np.array([]),
            "test_score": np.array([]),
            "metrics": {}
        }

    train_x = train_x.copy()
    test_x = test_x.copy()

    train_x["is_train"] = 1
    test_x["is_train"] = 0
    combined = pd.concat([train_x, test_x], axis=0).reset_index(drop=True)

    combined = combined.merge(user, on="user_id", how="left")
    combined = combined.merge(business, on="business_id", how="left")

    tip_aggregate = tip.groupby('business_id').agg(
        tip_count=('tip_text', 'count'),
        tip_avg_compliments=('tip_compliment', 'mean')
    ).reset_index()

    combined = combined.merge(tip_aggregate, on='business_id', how='left')
    combined[['tip_count', 'tip_avg_compliments']] = combined[['tip_count', 'tip_avg_compliments']].fillna(0)

    train = combined[combined['is_train'] == 1].reset_index(drop=True)
    train = train.merge(pd.DataFrame({"top_useful": train_y}), left_index=True, right_index=True)

    cols_to_drop = ['review_id', 'user_id', 'business_id', 'text', 'date',
                    'yelping_since', 'elite', 'categories']

    y = train['top_useful'].astype(int)
    X = train.drop(columns=cols_to_drop + ['top_useful'], errors='ignore')
    X_test = combined[combined['is_train'] == 0].drop(columns=cols_to_drop + ['is_train'], errors='ignore')

    for col in ['city', 'state']:
        if col in X.columns:
            X[col] = X[col].astype('category')
            X_test[col] = X_test[col].astype('category')

    X = X.select_dtypes(include=[np.number, 'bool', 'category'])
    X_test = X_test.select_dtypes(include=[np.number, 'bool', 'category'])

    for col in X.select_dtypes(include=['float64']).columns:
        X[col] = X[col].astype('float32')
        X_test[col] = X_test[col].astype('float32')

    for col in X.select_dtypes(include=['int64']).columns:
        X[col] = pd.to_numeric(X[col], downcast='integer')
        X_test[col] = pd.to_numeric(X_test[col], downcast='integer')

    X_train = X.iloc[train_indices].copy()
    X_val = X.iloc[val_indices].copy()
    y_train = y.iloc[train_indices].copy()
    y_val = y.iloc[val_indices].copy()

    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_jobs': -1,
        'random_state': RANDOM_STATE,
        'verbose': -1
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    lgb_val_score = model.predict(X_val)
    lgb_test_score = model.predict(X_test)

    lgb_metrics = best_tpr_under_fpr(y_val, lgb_val_score, fpr_limit=FPR_LIMIT)
    print_metrics("LIGHTGBM VALIDATION METRICS", lgb_metrics)

    return {
        "model_name": "LightGBM",
        "val_score": lgb_val_score,
        "test_score": lgb_test_score,
        "metrics": lgb_metrics
    }


def run_business_logistic(train_x, train_y, business_data, train_indices, val_indices, test_x):
    print("\n" + "=" * 80)
    print("Section 5: Business-only Logistic Regression")
    print("=" * 80)

    train = pd.concat([train_x, pd.DataFrame({"top_useful": train_y})], axis=1)
    test_x = test_x.copy()

    business_features = business_data.select_dtypes(include=[np.number]).columns.tolist()
    if 'is_open' in business_features:
        business_features.remove('is_open')

    train = train.merge(business_data, on='business_id', how='left')
    test_x = test_x.merge(business_data, on='business_id', how='left')

    feature_cols = [col for col in business_features if col in train.columns]
    X_train_full = train[feature_cols].fillna(0)
    X_test_full = test_x[feature_cols].fillna(0)

    X_train = X_train_full.iloc[train_indices].copy()
    X_val = X_train_full.iloc[val_indices].copy()
    y_train = pd.Series(train_y).iloc[train_indices].astype(int).copy()
    y_val = pd.Series(train_y).iloc[val_indices].astype(int).copy()

    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=N_JOBS)
    log_reg.fit(X_train, y_train)

    biz_val_score = log_reg.predict_proba(X_val)[:, 1]
    biz_test_score = log_reg.predict_proba(X_test_full)[:, 1]

    biz_metrics = best_tpr_under_fpr(y_val, biz_val_score, fpr_limit=FPR_LIMIT)
    print_metrics("BUSINESS-ONLY LOGISTIC REGRESSION VALIDATION METRICS", biz_metrics)

    return {
        "model_name": "Business-Only Logistic Regression",
        "val_score": biz_val_score,
        "test_score": biz_test_score,
        "metrics": biz_metrics
    }


def run_holiday_logistic(train_x, train_y, business, user, tip, train_indices, val_indices, test_x):
    print("\n" + "=" * 80)
    print("Section 6: Holiday Logistic Regression")
    print("=" * 80)

    business_clean = clean_business_features(business)
    user_clean = clean_user_features(user)
    tip_business, tip_user = aggregate_tip_features(tip)

    train_fe = add_review_features_with_holidays(train_x)
    test_fe = add_review_features_with_holidays(test_x)

    train_fe = train_fe.merge(business_clean, on="business_id", how="left")
    test_fe = test_fe.merge(business_clean, on="business_id", how="left")

    train_fe = train_fe.merge(user_clean, on="user_id", how="left")
    test_fe = test_fe.merge(user_clean, on="user_id", how="left")

    train_fe = train_fe.merge(tip_business, on="business_id", how="left")
    test_fe = test_fe.merge(tip_business, on="business_id", how="left")

    train_fe = train_fe.merge(tip_user, on="user_id", how="left")
    test_fe = test_fe.merge(tip_user, on="user_id", how="left")

    train_fe = train_fe.reset_index(drop=True)
    test_fe = test_fe.reset_index(drop=True)

    y_series = pd.Series(train_y).reset_index(drop=True).astype(int)

    for col in ["user_id", "business_id"]:
        if col in train_fe.columns and col in test_fe.columns:
            train_fe[col] = train_fe[col].astype(str)
            test_fe[col] = test_fe[col].astype(str)

    te_columns = ["user_id", "business_id", "review_year", "review_stars"]
    for col in te_columns:
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
    X_test_final = test_fe.reindex(columns=X.columns, fill_value=0).copy()
    X = X.replace([np.inf, -np.inf], np.nan).astype("float32")
    X_test_final = X_test_final.replace([np.inf, -np.inf], np.nan).astype("float32")

    X_train = X.iloc[train_indices].copy()
    X_val = X.iloc[val_indices].copy()
    y_train = y_series.iloc[train_indices].copy()
    y_val = y_series.iloc[val_indices].copy()

    holiday_logreg_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0,
            penalty="l2",
            class_weight="balanced",
            solver="saga",
            max_iter=1000,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE
        ))
    ])

    holiday_logreg_model.fit(X_train, y_train)

    holiday_val_score = holiday_logreg_model.predict_proba(X_val)[:, 1]
    holiday_test_score = holiday_logreg_model.predict_proba(X_test_final)[:, 1]

    holiday_metrics = best_tpr_under_fpr(y_val, holiday_val_score, fpr_limit=FPR_LIMIT)
    print_metrics("HOLIDAY LOGISTIC REGRESSION VALIDATION METRICS", holiday_metrics)

    return {
        "model_name": "Holiday Logistic Regression",
        "val_score": holiday_val_score,
        "test_score": holiday_test_score,
        "metrics": holiday_metrics
    }

# -----------------------------------------------------------------------------
# Top-K blend search functions from compiled notebook
# -----------------------------------------------------------------------------

def evaluate_blend_candidate(y_true, score, fpr_limit):
    metrics = best_tpr_under_fpr(y_true, score, fpr_limit=fpr_limit)
    return metrics


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

    val_matrix = np.column_stack([score_bank_val[name] for name in top_model_names]).astype("float32")
    test_matrix = np.column_stack([score_bank_test[name] for name in top_model_names]).astype("float32")

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

    # equal weight blend
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

    rng = np.random.default_rng(random_state)
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
    combined_df = pd.concat([advanced_blend_df, refined_df], ignore_index=True)
    combined_df = combined_df.sort_values(["tpr", "auc"], ascending=False).reset_index(drop=True)
    return combined_df

# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

def main():
    review_train_x = read_csv_from_data_dir(TRAIN_X_FILE)
    review_train_y = read_csv_from_data_dir(TRAIN_Y_FILE)
    review_test_x = read_csv_from_data_dir(TEST_X_FILE)
    business = read_csv_from_data_dir(BUSINESS_FILE)
    user = read_csv_from_data_dir(USER_FILE)
    tip = read_csv_from_data_dir(TIP_FILE)

    if review_train_y.shape[1] == 1:
        y = review_train_y.iloc[:, 0].astype(int).reset_index(drop=True)
    else:
        y = review_train_y["top_useful"].astype(int).reset_index(drop=True)

    idx = np.arange(len(review_train_x))
    train_indices, val_indices = train_test_split(
        idx,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    baseline_result = run_baseline_logistic_regression(
        review_train_x,
        y,
        review_test_x,
        business,
        user,
        tip,
        train_indices,
        val_indices
    )

    dt_result = run_decision_tree(
        review_train_x,
        y,
        business,
        train_indices,
        val_indices
    )

    rf_result = run_random_forest(
        review_train_x,
        y,
        business,
        user,
        tip,
        review_test_x,
        train_indices,
        val_indices
    )

    lgb_result = run_lightgbm(
        review_train_x,
        y,
        business,
        user,
        tip,
        review_test_x,
        train_indices,
        val_indices
    )

    biz_result = run_business_logistic(
        review_train_x,
        y,
        business,
        train_indices,
        val_indices,
        review_test_x
    )

    holiday_result = run_holiday_logistic(
        review_train_x,
        y,
        business,
        user,
        tip,
        train_indices,
        val_indices,
        review_test_x
    )

    results = [
        baseline_result,
        dt_result,
        rf_result,
        lgb_result,
        biz_result,
        holiday_result
    ]

    score_bank_val = {
        r['model_name']: r['val_score'] for r in results if len(r['val_score']) > 0
    }
    score_bank_test = {
        r['model_name']: r['test_score'] for r in results if len(r['test_score']) > 0
    }

    model_metrics = []
    for r in results:
        if r['metrics']:
            model_metrics.append({
                'model': r['model_name'],
                'auc': r['metrics'].get('auc', np.nan),
                'fpr': r['metrics'].get('fpr', np.nan),
                'tpr': r['metrics'].get('tpr', np.nan)
            })

    model_metrics_df = pd.DataFrame(model_metrics).sort_values('auc', ascending=False).reset_index(drop=True)
    model_metrics_path = os.path.join(OUTPUT_DIR, 'model_validation_metrics.csv')
    model_metrics_df.to_csv(model_metrics_path, index=False)
    print("Saved model validation metrics:", model_metrics_path)

    top_k = min(12, len(score_bank_val))
    top_model_names, val_matrix, test_matrix, blend_candidates_df, advanced_blend_df = advanced_topk_blend_search(
        model_metrics_df=model_metrics_df,
        score_bank_val=score_bank_val,
        score_bank_test=score_bank_test,
        y_val=y.iloc[val_indices].to_numpy(dtype="int32"),
        top_k=top_k,
        search_sample_size=200000,
        random_trials=30000,
        top_eval=300,
        random_state=RANDOM_STATE,
        fpr_limit=FPR_LIMIT_FINAL
    )

    blend_candidates_path = os.path.join(OUTPUT_DIR, "super_blend_candidates_sample_search.csv")
    advanced_blend_path = os.path.join(OUTPUT_DIR, "super_advanced_blend_full_eval.csv")
    blend_candidates_df.to_csv(blend_candidates_path, index=False)
    advanced_blend_df.to_csv(advanced_blend_path, index=False)

    print("Saved sample blend candidates:", blend_candidates_path)
    print("Saved full eval blend results:", advanced_blend_path)

    advanced_blend_refined_df = local_refine_blend(
        advanced_blend_df=advanced_blend_df,
        top_model_names=top_model_names,
        val_matrix=val_matrix,
        y_val=y.iloc[val_indices].to_numpy(dtype="int32"),
        n_rounds=2000,
        noise_scale=0.04,
        random_state=RANDOM_STATE + 999,
        fpr_limit=FPR_LIMIT_FINAL
    )

    advanced_refined_path = os.path.join(OUTPUT_DIR, "super_advanced_blend_refined_results.csv")
    advanced_blend_refined_df.to_csv(advanced_refined_path, index=False)
    print("Saved refined blend results:", advanced_refined_path)

    best_adv = advanced_blend_refined_df.iloc[0]
    best_weights = np.array([best_adv[f"w_{name}"] for name in top_model_names])
    best_threshold = best_adv["threshold"]
    super_val_score = val_matrix @ best_weights
    super_test_score = test_matrix @ best_weights

    super_metrics = best_tpr_under_fpr(y.iloc[val_indices].to_numpy(dtype="int32"), super_val_score, fpr_limit=FPR_LIMIT_FINAL)
    print_metrics("SUPER ADVANCED BLEND VALIDATION METRICS", super_metrics)

    super_pred = (super_test_score >= best_threshold).astype(int)
    super_submission = pd.DataFrame({"top_useful": super_pred})
    super_submission_path = os.path.join(OUTPUT_DIR, "group_7_yelp.csv")
    super_submission.to_csv(super_submission_path, index=False, header=False)

    super_score_path = os.path.join(OUTPUT_DIR, "super_advanced_blend_test_scores.csv")
    pd.DataFrame({"final_score": super_test_score, "prediction": super_pred}).to_csv(super_score_path, index=False)

    print("Submission:", super_submission_path)
    print("Scores:", super_score_path)
    print("Selected models and weights:")
    for name, w in zip(top_model_names, best_weights):
        print(f"{name:35s}: {w:.6f}")

    print("\nSuper validation result:")
    print("AUC:", super_metrics["auc"])
    print("FPR:", super_metrics["fpr"])
    print("TPR:", super_metrics["tpr"])
    print("Threshold:", best_threshold)


if __name__ == "__main__":
    main()
