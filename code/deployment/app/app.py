from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(page_title="Steam Success Predictor", page_icon="🎮")
st.title("🎮 Steam Pre-launch Success Predictor")
st.caption("Оценка вероятности получить ≥100 отзывов при доле положительных отзывов ≥80%.")

description = st.text_area(
    "Краткое описание",
    "A cooperative action game where players explore a mysterious world and fight challenging bosses.",
)
genre = st.text_input("Жанры (через ;)", "Action;Adventure;Indie")
tags = st.text_input("Теги (через ;)", "Action;Co-op;Multiplayer;Atmospheric")
categories = st.text_input("Категории (через ;)", "Single-player;Online Co-op;Steam Achievements")
price = st.number_input("Стартовая цена, USD", min_value=0.0, max_value=1000.0, value=19.99, step=1.0)
platforms = st.multiselect("Платформы", ["windows", "mac", "linux"], default=["windows"])
languages = st.text_input("Языки (через ;)", "English;Russian")
required_age = st.number_input("Возрастное ограничение", min_value=0, max_value=100, value=0)

if st.button("Оценить успех", type="primary"):
    payload = {
        "short_description": description,
        "genre": genre,
        "tags": tags,
        "categories": categories,
        "price": price,
        "platforms": platforms,
        "languages": [item.strip() for item in languages.split(";") if item.strip()],
        "required_age": required_age,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        probability = float(result["success_probability"])
        st.metric("Вероятность успеха", f"{probability:.1%}")
        st.progress(probability)
        if result["prediction"]:
            st.success("Модель относит концепцию к потенциально успешным.")
        else:
            st.warning("Модель не относит концепцию к потенциально успешным.")
        st.caption(f"Целевая метка: {result['target_definition']}. Это статистическая оценка, а не гарантия.")
    except requests.RequestException as exc:
        st.error(f"API недоступен: {exc}")

