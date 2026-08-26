from mlx_tune_kit.metrics import classification_metrics, exact_match_metrics


def test_exact_match_normalizes_case_and_space() -> None:
    metrics = exact_match_metrics(["Hello world", "No"], [" hello   WORLD ", "yes"])
    assert metrics == {"total": 2, "exact_match": 0.5}


def test_classification_metrics_include_invalid_outputs() -> None:
    metrics = classification_metrics(["YES", "NO", "YES"], ["yes", "NO", "maybe"], ["YES", "NO"])
    assert metrics["accuracy"] == 2 / 3
    assert metrics["invalid_rate"] == 1 / 3
    assert metrics["confusion_matrix"]["YES"]["INVALID"] == 1
