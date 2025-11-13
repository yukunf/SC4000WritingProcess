from model.data_preparation import merge_preprocessed_data
from model.advanced_ensemble import AdvancedStackedEnsemble
from model.model_enhanced import EnhancedEnsembleModel
from feature_extraction.linking_38th_features import (
    CharacterNGramFeatureExtractor,
    LDATopicFeatureExtractor,
    TSNEFeatureExtractor,
    PolarsFeatureExtractor,
)
from preprocess.text_process import EssayConstructor, TextProcessor
from preprocess.preprocess import Preprocess
import os
import sys
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent
sys.path.append(str(project_root))


def run_integration_pipeline(
    data_path="data/",
    use_existing_features=True,
    use_advanced_ensemble=True,
    use_tabpfn=False,
    n_splits=5,
    random_state=42,
):
    train_logs = pd.read_csv(f"{data_path}train_logs.csv")
    test_logs = pd.read_csv(f"{data_path}test_logs.csv")
    train_scores = pd.read_csv(f"{data_path}train_scores.csv")
    train_essays = pd.read_csv(f"{data_path}train_logs_extracted_text.csv")[
        ["id", "text"]
    ]
    print("Loaded cleaned csv")
    train_essays = train_essays.rename(columns={"text": "essay"})
    test_essays = pd.read_csv(f"{data_path}test_logs_extracted_text.csv")[
        ["id", "text"]
    ]
    test_essays = test_essays.rename(columns={"text": "essay"})
    print("Loaded reconstructed essays")
    train_basic = pd.read_csv(f"{data_path}train_preprocessed.csv")
    test_basic = pd.read_csv(f"{data_path}test_preprocessed.csv")
    print("Loaded existing preprocessed features")

    # Extract IDs and scores
    train_ids = train_basic["id"].values
    test_ids = test_basic["id"].values
    y = train_basic["score"].values

    train_basic = train_basic.drop(columns=["id", "score"], errors="ignore")
    test_basic = test_basic.drop(columns=["id"], errors="ignore")

    # ngram
    char_ngram_extractor = CharacterNGramFeatureExtractor(
        ngram_range=(1, 4),
        analyzer="char_wb",
        max_features=1200,
    )

    train_char_ngrams, test_char_ngrams = char_ngram_extractor.fit_transform(
        train_essays, test_essays
    )

    train_char_ngrams = train_char_ngrams.drop(columns=["id"])
    test_char_ngrams = test_char_ngrams.drop(columns=["id"])

    variance = train_char_ngrams.var()
    top_features = variance.nlargest(500).index
    train_char_ngrams = train_char_ngrams[top_features]
    test_char_ngrams = test_char_ngrams[top_features]

    train_basic = pd.concat(
        [
            train_basic.reset_index(drop=True),
            train_char_ngrams.reset_index(drop=True),
        ],
        axis=1,
    )
    test_basic = pd.concat(
        [
            test_basic.reset_index(drop=True),
            test_char_ngrams.reset_index(drop=True),
        ],
        axis=1,
    )

    # lda
    lda_extractor = LDATopicFeatureExtractor(
        n_topics=6, max_iter=10, random_state=random_state
    )

    train_lda, test_lda = lda_extractor.fit_transform(
        train_essays, test_essays)

    train_lda = train_lda.drop(columns=["id"])
    test_lda = test_lda.drop(columns=["id"])
    train_basic = pd.concat(
        [train_basic.reset_index(drop=True), train_lda.reset_index(drop=True)],
        axis=1,
    )
    test_basic = pd.concat(
        [test_basic.reset_index(drop=True), test_lda.reset_index(drop=True)], axis=1
    )

    # polars
    polars_extractor = PolarsFeatureExtractor()

    train_polars = polars_extractor.extract_features(train_logs)
    test_polars = polars_extractor.extract_features(test_logs)

    train_polars_temp = (
        train_polars.set_index("id").loc[train_ids].reset_index(drop=True)
    )
    test_polars_temp = test_polars.set_index(
        "id").loc[test_ids].reset_index(drop=True)

    polars_cols = [
        c for c in train_polars_temp.columns if c not in train_basic.columns]
    train_polars_temp = train_polars_temp[polars_cols]
    test_polars_temp = test_polars_temp[polars_cols]

    train_basic = pd.concat(
        [
            train_basic.reset_index(drop=True),
            train_polars_temp.reset_index(drop=True),
        ],
        axis=1,
    )
    test_basic = pd.concat(
        [
            test_basic.reset_index(drop=True),
            test_polars_temp.reset_index(drop=True),
        ],
        axis=1,
    )
    # tsne
    tsne_extractor = TSNEFeatureExtractor(
        perplexities=[20, 50, 80],
        n_components=2,
        random_state=random_state,
        n_jobs=-1,
    )

    train_tsne, test_tsne = tsne_extractor.fit_transform(
        train_basic, test_basic)

    train_basic = pd.concat(
        [train_basic.reset_index(drop=True),
         train_tsne.reset_index(drop=True)],
        axis=1,
    )
    test_basic = pd.concat(
        [test_basic.reset_index(drop=True), test_tsne.reset_index(drop=True)],
        axis=1,
    )

    # training
    ensemble = EnhancedEnsembleModel(
        n_splits=n_splits, random_state=random_state)

    scores, train_preds, test_preds = ensemble.fit(train_basic, y, test_basic)

    submission = pd.DataFrame({"id": test_ids, "score": test_preds})
    return submission


if __name__ == "__main__":
    submission = run_integration_pipeline()
