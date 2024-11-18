import os
import pandas as pd
import streamlit as st
from src.utils import read_model

st.set_page_config(
    page_title="Flat App",
)


model_path = 'LRmodel.pkl'

# Получаем данные
square = st.sidebar.number_input("Площадь квартиры", 1.0, 3000.0, 100.0)
rooms = st.sidebar.number_input("Количество комнат", 1, 15, 2)
floor = st.sidebar.number_input("Этаж", 1, 100, 5)
latitude = st.sidebar.number_input("Широта", 1.0, 90.0, 56.0)
longitude = st.sidebar.number_input("Долгота", 1.0, 90.0, 38.0)


# Загружаем модель
model = read_model(model_path)

# Прогноз
if st.button('Прогнозировать цену'):
    if model:
        
        inputDF = pd.DataFrame(
            {
                "lat": [latitude],
                "lon": [longitude],
                "total_square": [square],
                "rooms": [rooms],
                "floor": [floor]
            }
        )

        preds = model.predict(inputDF)[0]
        preds_rounded = round(preds, 2)

        formatted_preds = "{:,.2f}".format(preds_rounded).replace(",", " ")

        st.write(f"Предсказанная цена квартиры: {formatted_preds} рублей")
    else:
        st.write("Ошибка при загрузке модели. Пожалуйста, проверьте файл модели.")
