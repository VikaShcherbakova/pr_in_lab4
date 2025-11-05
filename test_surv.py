import pytest
import pandas as pd
from app import calculate_survival


@pytest.fixture
def titanic_df():
    return pd.DataFrame({
        "Age": [22, 38, 26, 35, 28, 2, None],
        "Survived": [0, 1, 1, 1, 0, 1, 1],
        "Embarked": ["S", "C", "S", "S", "Q", "C", None]
    })


def test_basic_calculation(titanic_df):
    result = calculate_survival(titanic_df, max_age=40)

    s_stats = result[result["Порт"] == "S"].iloc[0]
    assert s_stats["Количество"] == 3
    assert s_stats["Выжило"] == 2
    assert pytest.approx(s_stats["Доля выживших"], rel=1e-3) == 2 / 3

    c_stats = result[result["Порт"] == "C"].iloc[0]
    assert c_stats["Количество"] == 2
    assert c_stats["Выжило"] == 2
    assert pytest.approx(c_stats["Доля выживших"], rel=1e-3) == 1.0

    q_stats = result[result["Порт"] == "Q"].iloc[0]
    assert q_stats["Количество"] == 1
    assert q_stats["Выжило"] == 0
    assert q_stats["Доля выживших"] == 0.0


def test_filter_by_age(titanic_df):
    result = calculate_survival(titanic_df, max_age=3)
    assert len(result) == 1
    assert result.iloc[0]["Порт"] == "C"
    assert result.iloc[0]["Выжило"] == 1
    assert result.iloc[0]["Количество"] == 1


def test_missing_values_handling(titanic_df):
    result = calculate_survival(titanic_df, max_age=40)
    assert None not in set(result["Порт"])
