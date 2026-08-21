"""Frozen lexical baseline recipe shared by training and selection experiments."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def build_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            (
                "chars",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=160_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "words",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=80_000,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    C=5.0,
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=20260820,
                ),
            ),
        ]
    )
