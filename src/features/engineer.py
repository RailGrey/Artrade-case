"""Feature engineering utilities."""

import re

import numpy as np
import pandas as pd


def extract_datetime_features(
    df: pd.DataFrame, ts_col: str, prefix: str
) -> pd.DataFrame:
    """Extract datetime features from timestamp column."""
    df = df.copy()
    if ts_col not in df.columns:
        return df
    dt = pd.to_datetime(df[ts_col], unit="s", errors="coerce")

    df[f"{prefix}_hour"] = dt.dt.hour
    df[f"{prefix}_dayofweek"] = dt.dt.dayofweek
    df[f"{prefix}_day"] = dt.dt.day
    df[f"{prefix}_month"] = dt.dt.month
    df[f"{prefix}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    return df


def extract_order_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from order composition text."""
    df = df.copy()

    order_text_col = "lead_Состав заказа"
    if order_text_col not in df.columns:
        return df

    def count_items(text: str) -> int:
        if pd.isna(text):
            return 0
        return len(re.findall(r"\d+\)\s+[А-Яа-яёЁA-Za-z]", str(text)))

    def has_delivery(text: str) -> int:
        if pd.isna(text):
            return 0
        return 1 if re.search(r"Доставка", str(text)) else 0

    def extract_total_price(text: str) -> float:
        if pd.isna(text):
            return 0.0
        matches = re.findall(r"Розничная цена:\s*(\d+(?:\.\d+)?)\s*RUB", str(text))
        return sum(float(m) for m in matches) if matches else 0.0

    df["n_items"] = df[order_text_col].apply(count_items)
    df["has_delivery"] = df[order_text_col].apply(has_delivery)
    df["order_calculated_price"] = df[order_text_col].apply(extract_total_price)

    return df


def extract_product_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Extract product category flags from order text."""
    df = df.copy()

    order_text_col = "lead_Состав заказа"
    if order_text_col not in df.columns:
        return df

    categories = {
        "подушка": r"подушк",
        "матрас": r"матрас",
        "чехол": r"чехол",
        "жилет": r"жилет",
        "маска": r"маска",
        "тапки": r"тапк",
    }

    for cat_name, pattern in categories.items():
        df[f"product_{cat_name}"] = df[order_text_col].apply(
            lambda x: 1 if re.search(pattern, str(x).lower()) else 0
        )

    return df


def extract_discount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and process discount-related features."""
    df = df.copy()

    if "lead_Скидка" in df.columns:
        df["has_discount"] = (df["lead_Скидка"] > 0).astype(int)
        df["discount_percent"] = (
            df["lead_Скидка"] / df["lead_price"].replace(0, 1) * 100
        )

    if "lead_price" in df.columns and "lead_Сумма заказа" in df.columns:
        df["price_vs_order_diff"] = df["lead_price"] - df["lead_Сумма заказа"]

    return df


def create_utm_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Create UTM-related aggregate features."""
    df = df.copy()

    df["has_utm"] = (
        ~df["lead_utm_source"].isna() & (df["lead_utm_source"] != "Unknown")
    ).astype(int)

    if "lead_utm_medium" in df.columns:
        df["is_cpc"] = (df["lead_utm_medium"] == "cpc").astype(int)
        df["is_organic"] = (df["lead_utm_medium"].isin(["organic", "0"])).astype(int)

    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    # Extract datetime features from allowed columns
    df = extract_datetime_features(df, "sale_ts", "sale")

    # Add contact age (time from contact creation to order)
    if "contact_created_at" in df.columns and "sale_ts" in df.columns:
        df["contact_age_days"] = (df["sale_ts"] - df["contact_created_at"]) / 86400
        df["contact_age_days"] = df["contact_age_days"].fillna(0).clip(lower=0)

    # Lead creation to sale time
    if "lead_created_at" in df.columns and "sale_ts" in df.columns:
        df["lead_age_days"] = (df["sale_ts"] - df["lead_created_at"]) / 86400
        df["lead_age_days"] = df["lead_age_days"].fillna(0).clip(lower=0)

    df = extract_order_features(df)
    df = extract_product_categories(df)
    df = extract_discount_features(df)
    df = add_geo_features(df)
    df = create_utm_aggregates(df)
    df = add_lead_qualification_features(df)

    cols_to_drop = [
        "lead_Состав заказа",
        "contact_Адрес ПВЗ",
        "lead_Скидка",
        "lead_utm_source",
        "lead_utm_medium",
        "contact_Город",
        "lead_Статус заказа на сайте",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    return df


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add geographic features."""
    df = df.copy()

    if "contact_Город" in df.columns:
        top_cities = df["contact_Город"].value_counts().head(20).index.tolist()
        df["city_tier"] = df["contact_Город"].apply(
            lambda x: (
                "top_city"
                if x in top_cities[:10]
                else "mid_city"
                if x in top_cities
                else "other"
            )
        )
        df["city_is_moscow"] = (
            df["contact_Город"]
            .str.contains("Москва|Moscow", case=False, na=False)
            .astype(int)
        )
        df["city_is_spb"] = (
            df["contact_Город"]
            .str.contains("Петербург|Spb|Saint", case=False, na=False)
            .astype(int)
        )

    return df


def add_lead_qualification_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lead qualification features."""
    df = df.copy()

    qual_col = "lead_Квалификация лида"
    if qual_col in df.columns:
        df["is_a_lead"] = df[qual_col].str.contains("А - лид", na=False).astype(int)
        df["is_d_lead"] = df[qual_col].str.contains("D - лид", na=False).astype(int)

    status_col = "lead_Статус заказа на сайте"
    if status_col in df.columns:
        df["has_site_status"] = df[status_col].notna().astype(int)

    return df


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Get list of feature names for modeling."""
    exclude_cols = ["lead_id", "target", "buyout_flag"]
    return [c for c in df.columns if c not in exclude_cols]
