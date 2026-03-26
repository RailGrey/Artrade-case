"""Project configuration with paths and parameters."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RAW_CSV = RAW_DATA_DIR / "MIPT_hackathon_dataset.csv"
TRAIN_PARQUET = PROCESSED_DATA_DIR / "train.parquet"
TEST_PARQUET = PROCESSED_DATA_DIR / "test.parquet"
CLEANED_PARQUET = PROCESSED_DATA_DIR / "cleaned.parquet"

MODEL_BASELINE = MODELS_DIR / "baseline_logreg.pkl"
MODEL_LGBM = MODELS_DIR / "lgbm_model.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5

TARGET_COL = "buyout_flag"

COLS_TO_DROP_LEAKAGE = [
    "lead_id",
    "contact_id",
    "lead_responsible_user_id",
    "lead_group_id",
    "lead_status_id",
    "lead_pipeline_id",
    "contact_responsible_user_id",
    "handed_to_delivery_ts",
    "sale_ts",
    "lead_created_at",
    "lead_updated_at",
    "contact_created_at",
    "contact_updated_at",
    "issued_or_pvz_ts",
    "received_ts",
    "rejected_ts",
    "returned_ts",
    "days_sale_to_handed",
    "days_handed_to_issued_pvz",
    "days_to_outcome",
    "lifecycle_incomplete",
    "current_status_id",
    "closed_ts",
    "lead_closed_at",
    "lead_loss_reason_id",
    "outcome_unknown",
    "lead_Дата перехода в Сборку",
    "lead_Дата получения денег на Р/С",
    "lead_Дата возврата посылки на склад",
    "lead_Дата перехода Передан в доставку",
    "lead_Ответственный за доставку",
    "lead_Дата приобретения изделия",
    "lead_Счет оплачен",
    "lead_Оплачено клиентом",
    "lead_Статус заказа на сайте",
]

CATEGORICAL_COLS = [
    "lead_Источник",
    "lead_Квалификация лида",
    "lead_Категория и варианты выбора",
    "lead_Проблема",
    "lead_Тариф Доставки",
    "lead_Служба доставки",
    "lead_Вид оплаты",
    "lead_Компания Отправитель",
    "contact_Город",
    "contact_Источник трафика",
    "lead_LEADQUALIFYCATION",
    "lead_utm_source",
    "lead_utm_medium",
    "lead_utm_campaign",
    "lead_utm_content",
    "lead_utm_term",
    "lead_utm_group",
    "lead_utm_referrer",
    "lead_FORMNAME",
    "lead_type",
    "lead_Метод доставки",
    "lead_Тип отправления",
    "lead_Объявленная ценность (руб)",
    "lead_ПВЗ СДЭК",
]

NUMERIC_COLS = [
    "lead_price",
    "lead_Скидка",
    "lead_Вес (грамм)*",
    "lead_Высота",
    "lead_Ширина",
    "lead_Длина",
    "lead_Масса (гр)",
    "lead_Линейная высота (см)",
    "lead_Линейная ширина (см)",
    "lead_Линейная длина (см)",
    "lead_Сумма наложенного платежа (руб)",
    "lead_Стоимость доставки",
    "lead_Сумма заказа",
    "lead_LTV",
    "contact_LTV",
    "contact_Число сделок",
]

DATETIME_COLS = [
    "sale_ts",
    "lead_created_at",
    "lead_updated_at",
    "contact_created_at",
    "contact_updated_at",
]
