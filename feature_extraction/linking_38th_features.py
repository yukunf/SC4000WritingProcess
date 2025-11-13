# features from 38 th notebook
import pandas as pd
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.manifold import TSNE
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import warnings
import pickle
import os

warnings.filterwarnings("ignore")


class CharacterNGramFeatureExtractor:
    def __init__(self, ngram_range=(1, 4), analyzer="char_wb", max_features=None):
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.max_features = max_features
        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range, analyzer=analyzer, max_features=max_features
        )

    def fit_transform(self, train_essays, test_essays):
        combined = pd.concat([train_essays["essay"], test_essays["essay"]])
        self.vectorizer.fit(combined)

        X_train = self.vectorizer.transform(train_essays["essay"])
        X_train_dense = X_train.todense()

        train_features = pd.DataFrame()
        for i in range(X_train_dense.shape[1]):
            L = list(X_train_dense[:, i])
            train_features[f"char_ngram_{i}"] = [int(x) for x in L]
        train_features["id"] = train_essays["id"].values

        X_test = self.vectorizer.transform(test_essays["essay"])
        X_test_dense = X_test.todense()

        test_features = pd.DataFrame()
        for i in range(X_test_dense.shape[1]):
            L = list(X_test_dense[:, i])
            test_features[f"char_ngram_{i}"] = [int(x) for x in L]
        test_features["id"] = test_essays["id"].values

        return train_features, test_features


class LDATopicFeatureExtractor:
    def __init__(self, n_topics=6, max_iter=10, random_state=42):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.random_state = random_state
        self.models = {}
        self.vectorizers = {}

    def fit_transform(self, train_essays, test_essays):
        train_topics = pd.DataFrame({"id": train_essays["id"]})
        test_topics = pd.DataFrame({"id": test_essays["id"]})

        # Combine for fitting
        combined = pd.concat([train_essays, test_essays])

        # 1 stopwords
        train_topics, test_topics = self._fit_lda_variant(
            combined,
            train_essays,
            test_essays,
            train_topics,
            test_topics,
            "word_topics",
            CountVectorizer(stop_words="english"),
        )

        # 2 char wb
        train_topics, test_topics = self._fit_lda_variant(
            combined,
            train_essays,
            test_essays,
            train_topics,
            test_topics,
            "char_topics",
            CountVectorizer(analyzer="char_wb"),
        )

        # 3. ngram = (5, 6)
        train_topics, test_topics = self._fit_lda_variant(
            combined,
            train_essays,
            test_essays,
            train_topics,
            test_topics,
            "char_ngram_topics",
            CountVectorizer(analyzer="char_wb", ngram_range=(5, 6)),
        )

        return train_topics, test_topics

    def _fit_lda_variant(
        self,
        combined_essays,
        train_essays,
        test_essays,
        train_topics,
        test_topics,
        prefix,
        vectorizer,
    ):
        vectorizer.fit(combined_essays["essay"])
        self.vectorizers[prefix] = vectorizer

        train_words = pd.DataFrame(
            vectorizer.transform(train_essays["essay"]).toarray()
        )
        test_words = pd.DataFrame(vectorizer.transform(test_essays["essay"]).toarray())

        lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=False,
        )

        combined_words = pd.concat([train_words, test_words])
        lda.fit(combined_words)

        self.models[prefix] = lda

        topic_cols = [f"{prefix}_{i}" for i in range(self.n_topics)]
        train_topics[topic_cols] = lda.transform(train_words)
        test_topics[topic_cols] = lda.transform(test_words)

        return train_topics, test_topics


class TSNEFeatureExtractor:
    def __init__(
        self, perplexities=[20, 50, 80], n_components=2, random_state=42, n_jobs=-1
    ):
        self.perplexities = perplexities
        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit_transform(self, train_features, test_features):
        train_tsne = pd.DataFrame()
        test_tsne = pd.DataFrame()

        combined = pd.concat([train_features, test_features])

        print("tsne")
        for perplexity in self.perplexities:
            tsne = TSNE(
                n_components=self.n_components,
                random_state=self.random_state,
                perplexity=perplexity,
                n_jobs=self.n_jobs,
                verbose=False,
            )

            embeddings = tsne.fit_transform(combined.fillna(0))

            train_size = len(train_features)
            for i in range(self.n_components):
                col_name = f"tsne_p{perplexity}_{i}"
                train_tsne[col_name] = embeddings[:train_size, i]
                test_tsne[col_name] = embeddings[train_size:, i]

        return train_tsne, test_tsne


class PolarsFeatureExtractor:
    def __init__(self):
        self.num_cols = [
            "down_time",
            "up_time",
            "action_time",
            "cursor_position",
            "word_count",
        ]
        self.activities = ["Input", "Remove/Cut", "Nonproduction", "Replace", "Paste"]
        self.events = [
            "q",
            "Space",
            "Backspace",
            "Shift",
            "ArrowRight",
            "Leftclick",
            "ArrowLeft",
            ".",
            ",",
            "ArrowDown",
            "ArrowUp",
            "Enter",
            "CapsLock",
            "'",
            "Delete",
            "Unidentified",
        ]
        self.text_changes = [
            "q",
            " ",
            ".",
            ",",
            "\n",
            "'",
            '"',
            "-",
            "?",
            ";",
            "=",
            "/",
            "\\",
            ":",
            "n",
        ]

    def extract_features(self, df):
        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)

        feats = self._count_by_values(df, "activity", self.activities)
        feats = feats.join(
            self._count_by_values(df, "text_change", self.text_changes),
            on="id",
            how="left",
        )
        feats = feats.join(
            self._count_by_values(df, "down_event", self.events), on="id", how="left"
        )
        feats = feats.join(
            self._count_by_values(df, "up_event", self.events), on="id", how="left"
        )
        feats = feats.join(self._input_word_stats(df), on="id", how="left")
        feats = feats.join(self._numerical_stats(df), on="id", how="left")
        feats = feats.join(self._categorical_stats(df), on="id", how="left")
        feats = feats.join(self._idle_time_features(df), on="id", how="left")
        feats = feats.join(self._p_burst_features(df), on="id", how="left")
        feats = feats.join(self._r_burst_features(df), on="id", how="left")

        return feats.to_pandas()

    def _count_by_values(self, df, colname, values):
        fts = df.select(pl.col("id").unique(maintain_order=True))
        for i, value in enumerate(values):
            tmp_df = df.group_by("id").agg(
                pl.col(colname).is_in([value]).sum().alias(f"{colname}_{i}_cnt")
            )
            fts = fts.join(tmp_df, on="id", how="left")
        return fts

    def _input_word_stats(self, df):
        temp = df.filter(
            (~pl.col("text_change").str.contains("=>"))
            & (pl.col("text_change") != "NoChange")
        )
        temp = temp.group_by("id").agg(
            pl.col("text_change").str.concat("").str.extract_all(r"q+")
        )
        temp = temp.with_columns(
            input_word_count=pl.col("text_change").list.len(),
            input_word_length_mean=pl.col("text_change").map_elements(
                lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0),
                return_dtype=pl.Float64,
            ),
            input_word_length_max=pl.col("text_change").map_elements(
                lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0),
                return_dtype=pl.Float64,
            ),
            input_word_length_std=pl.col("text_change").map_elements(
                lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0),
                return_dtype=pl.Float64,
            ),
            input_word_length_median=pl.col("text_change").map_elements(
                lambda x: np.median([len(i) for i in x] if len(x) > 0 else 0),
                return_dtype=pl.Float64,
            ),
            input_word_length_skew=pl.col("text_change").map_elements(
                lambda x: skew([len(i) for i in x] if len(x) > 0 else 0),
                return_dtype=pl.Float64,
            ),
        )
        return temp.drop("text_change")

    def _numerical_stats(self, df):
        return df.group_by("id").agg(
            [
                pl.sum("action_time").alias("action_time_sum"),
                *[pl.mean(col).alias(f"{col}_mean") for col in self.num_cols],
                *[pl.std(col).alias(f"{col}_std") for col in self.num_cols],
                *[pl.median(col).alias(f"{col}_median") for col in self.num_cols],
                *[pl.min(col).alias(f"{col}_min") for col in self.num_cols],
                *[pl.max(col).alias(f"{col}_max") for col in self.num_cols],
                *[
                    pl.quantile(col, 0.5).alias(f"{col}_quantile")
                    for col in self.num_cols
                ],
            ]
        )

    def _categorical_stats(self, df):
        return df.group_by("id").agg(
            [
                pl.n_unique("activity").alias("activity_nunique"),
                pl.n_unique("down_event").alias("down_event_nunique"),
                pl.n_unique("up_event").alias("up_event_nunique"),
                pl.n_unique("text_change").alias("text_change_nunique"),
            ]
        )

    def _idle_time_features(self, df):
        temp = df.with_columns(
            pl.col("up_time").shift().over("id").alias("up_time_lagged")
        )
        temp = temp.with_columns(
            (pl.col("down_time") - pl.col("up_time_lagged"))
            .abs()
            .truediv(1000)
            .fill_null(0)
            .alias("time_diff")
        )
        temp = temp.filter(pl.col("activity").is_in(["Input", "Remove/Cut"]))

        return temp.group_by("id").agg(
            [
                pl.max("time_diff").alias("inter_key_largest_latency"),
                pl.median("time_diff").alias("inter_key_median_latency"),
                pl.mean("time_diff").alias("mean_pause_time"),
                pl.std("time_diff").alias("std_pause_time"),
                pl.sum("time_diff").alias("total_pause_time"),
                pl.col("time_diff")
                .filter((pl.col("time_diff") > 0.5) & (pl.col("time_diff") < 1))
                .count()
                .alias("pauses_half_sec"),
                pl.col("time_diff")
                .filter((pl.col("time_diff") > 1) & (pl.col("time_diff") < 1.5))
                .count()
                .alias("pauses_1_sec"),
                pl.col("time_diff")
                .filter((pl.col("time_diff") > 1.5) & (pl.col("time_diff") < 2))
                .count()
                .alias("pauses_1_half_sec"),
                pl.col("time_diff")
                .filter((pl.col("time_diff") > 2) & (pl.col("time_diff") < 3))
                .count()
                .alias("pauses_2_sec"),
                pl.col("time_diff")
                .filter(pl.col("time_diff") > 3)
                .count()
                .alias("pauses_3_sec"),
            ]
        )

    def _p_burst_features(self, df):
        temp = df.with_columns(
            pl.col("up_time").shift().over("id").alias("up_time_lagged")
        )
        temp = temp.with_columns(
            (pl.col("down_time") - pl.col("up_time_lagged"))
            .abs()
            .truediv(1000)
            .fill_null(0)
            .alias("time_diff")
        )
        temp = temp.filter(pl.col("activity").is_in(["Input", "Remove/Cut"]))
        temp = temp.with_columns((pl.col("time_diff") < 2).alias("time_diff"))
        temp = temp.with_columns(
            pl.when(pl.col("time_diff") & pl.col("time_diff").is_last_distinct())
            .then(pl.count())
            .over(pl.col("time_diff").rle_id())
            .alias("P_bursts")
        )
        temp = temp.drop_nulls()

        return temp.group_by("id").agg(
            [
                pl.mean("P_bursts").alias("P_bursts_mean"),
                pl.std("P_bursts").alias("P_bursts_std"),
                pl.count("P_bursts").alias("P_bursts_count"),
                pl.median("P_bursts").alias("P_bursts_median"),
                pl.max("P_bursts").alias("P_bursts_max"),
                pl.first("P_bursts").alias("P_bursts_first"),
                pl.last("P_bursts").alias("P_bursts_last"),
            ]
        )

    def _r_burst_features(self, df):
        temp = df.filter(pl.col("activity").is_in(["Input", "Remove/Cut"]))
        temp = temp.with_columns(
            pl.col("activity").is_in(["Remove/Cut"]).alias("activity")
        )
        temp = temp.with_columns(
            pl.when(pl.col("activity") & pl.col("activity").is_last_distinct())
            .then(pl.count())
            .over(pl.col("activity").rle_id())
            .alias("R_bursts")
        )
        temp = temp.drop_nulls()

        return temp.group_by("id").agg(
            [
                pl.mean("R_bursts").alias("R_bursts_mean"),
                pl.std("R_bursts").alias("R_bursts_std"),
                pl.median("R_bursts").alias("R_bursts_median"),
                pl.max("R_bursts").alias("R_bursts_max"),
                pl.first("R_bursts").alias("R_bursts_first"),
                pl.last("R_bursts").alias("R_bursts_last"),
            ]
        )
