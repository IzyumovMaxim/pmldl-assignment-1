import pandas as pd

from code.common import MODEL_COLUMNS, build_features


def test_build_features_has_stable_schema():
    row = pd.DataFrame([{
        "Short Description": "A cooperative space adventure",
        "Genre": "Action;Adventure",
        "Tags": "Co-op;Space",
        "Categories": "Multi-player;Steam Achievements",
        "Price": 19.99,
        "Platforms": "windows;linux",
        "Languages": "English;Russian",
        "Required Age": 12,
    }])
    result = build_features(row)
    assert result.columns.tolist() == MODEL_COLUMNS
    assert result.loc[0, "windows"] == 1
    assert result.loc[0, "mac"] == 0
    assert result.loc[0, "language_count"] == 2
    assert "5426" not in build_features(pd.DataFrame([{
        "Short Description": "test", "Genre": "Action", "Tags": "Action: 5426; Co-op: 42",
        "Categories": "", "Price": 10, "Platforms": "windows", "Languages": "English",
        "Required Age": 0,
    }])).loc[0, "text"]
