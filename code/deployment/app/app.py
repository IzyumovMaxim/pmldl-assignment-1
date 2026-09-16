from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")

GENRES = [
    "Action", "Adventure", "Casual", "Indie", "Massively Multiplayer",
    "Racing", "RPG", "Simulation", "Sports", "Strategy", "Free to Play",
    "Early Access",
]
TAGS = [
    "2D", "3D", "Action", "Adventure", "Atmospheric", "Casual", "Co-op",
    "Competitive", "Difficult", "Exploration", "Fantasy", "First-Person",
    "Free to Play", "Funny", "Horror", "Indie", "Multiplayer", "Open World",
    "Pixel Graphics", "Platformer", "Puzzle", "PvE", "PvP", "RPG", "Racing",
    "Retro", "Roguelike", "Shooter", "Simulation", "Singleplayer", "Story Rich",
    "Strategy", "Survival", "Third Person", "Turn-Based",
]
CATEGORIES = [
    "Single-player", "Multi-player", "PvP", "Online PvP", "Co-op",
    "Online Co-op", "Shared/Split Screen", "Cross-Platform Multiplayer",
    "Steam Achievements", "Steam Trading Cards", "Steam Cloud",
    "Full controller support", "Partial Controller Support", "Remote Play Together",
    "Steam Workshop", "Includes level editor", "VR Supported",
]
LANGUAGES = [
    "English", "Russian", "French", "German", "Spanish - Spain", "Italian",
    "Portuguese - Brazil", "Portuguese - Portugal", "Polish", "Dutch",
    "Turkish", "Ukrainian", "Czech", "Hungarian", "Romanian", "Swedish",
    "Norwegian", "Danish", "Finnish", "Greek", "Japanese", "Korean",
    "Simplified Chinese", "Traditional Chinese", "Thai", "Vietnamese",
    "Arabic", "Hindi",
]

st.set_page_config(page_title="Steam Success Predictor", page_icon="🎮")
st.title("🎮 Steam Pre-launch Success Predictor")
st.caption("Оценка вероятности получить ≥100 отзывов при доле положительных отзывов ≥80%.")

description = st.text_area(
    "Краткое описание",
    "A cooperative action game where players explore a mysterious world and fight challenging bosses.",
)
genre = st.multiselect(
    "Жанры", GENRES, default=["Action", "Adventure", "Indie"],
    accept_new_options=True, help="Можно найти пункт поиском или добавить собственный.",
)
tags = st.multiselect(
    "Теги", TAGS, default=["Action", "Co-op", "Multiplayer", "Atmospheric"],
    accept_new_options=True,
)
categories = st.multiselect(
    "Категории", CATEGORIES, default=["Single-player", "Online Co-op", "Steam Achievements"],
    accept_new_options=True,
)
developer = st.text_input("Разработчик", "Independent Studio")
publisher = st.text_input("Издатель", "Self-published")
price = st.number_input("Стартовая цена, USD", min_value=0.0, max_value=1000.0, value=19.99, step=1.0)
platforms = st.multiselect("Платформы", ["windows", "mac", "linux"], default=["windows"])
languages = st.multiselect(
    "Поддерживаемые языки", LANGUAGES, default=["English", "Russian"],
    accept_new_options=True,
)
required_age = st.number_input("Возрастное ограничение", min_value=0, max_value=100, value=0)

if st.button("Оценить успех", type="primary"):
    if not genre:
        st.warning("Выберите хотя бы один жанр.")
        st.stop()
    payload = {
        "short_description": description,
        "genre": ";".join(genre),
        "tags": ";".join(tags),
        "categories": ";".join(categories),
        "developer": developer,
        "publisher": publisher,
        "price": price,
        "platforms": platforms,
        "languages": languages,
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
