import streamlit as st
import pandas as pd


def calculate_survival(df: pd.DataFrame, max_age: int) -> pd.DataFrame:
    filtered_df = df[df["Age"] <= max_age]
    if filtered_df.empty:
        return pd.DataFrame(columns=["Порт", "Количество",
                                     "Выжило", "Доля выживших"])

    result = (
        filtered_df.groupby("Embarked")
        .agg(Количество=("Survived", "count"), Выжило=("Survived", "sum"))
        .reset_index()
    )
    result["Доля выживших"] = (result["Выжило"]
                               / result["Количество"]).round(3)
    result = result.rename(columns={"Embarked": "Порт"})
    return result


@st.cache_data
def load_data():
    url = ("https://raw.githubusercontent.com/datasciencedojo"
           "/datasets/master/titanic.csv")
    df = pd.read_csv(url)
    return df


data = load_data()

st.image("https://wallpapers.com/images/hd/"
         "titanic-1680-x-1050-background-ld95nte3gk5y0pad.jpg")

st.title("Подсчет доли выживших пассажиров Титаника по каждому пункту посадки")
st.subheader("Выбор максимального возраста")

min_age = max(1, int(data["Age"].min()))
max_age = int(data["Age"].max())

selected_age = st.slider(
    "Выберите максимальный возраст пассажиров:",
    min_value=min_age,
    max_value=max_age,
    value=max_age,
    step=1,
    help="Выберите максимальный возраст — таблица обновится автоматически"
)

filtered_df = data[data["Age"] <= selected_age]

display_mode = st.radio(
    "Формат отображения доли выживших:",
    ("Доля", "Проценты"),
    horizontal=True
)

if filtered_df.empty:
    st.warning("Нет пассажиров младше указанного возраста.")
    st.stop()

result_df = calculate_survival(data, selected_age)

# Форматирование для отображения
if display_mode == "Проценты":
    result_df["Доля выживших"] = (
        (result_df["Доля выживших"] * 100).round(1))
else:
    result_df["Доля выживших"] = result_df["Доля выживших"].round(3)
st.dataframe(result_df, use_container_width=True)
