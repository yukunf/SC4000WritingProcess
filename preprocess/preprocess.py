import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class Preprocess:

    def label_encoding(self, df, col="id"):
        label_encoder = LabelEncoder()
        label_encoder.fit(df[col])
        df[col + "_encoded"] = label_encoder.transform(df[col])
        return df

    # remove time that the author havent start writing or is resting
    # reference: remove_margin for https://www.kaggle.com/code/tomooinubushi/1st-place-solution-training-and-inference-code

    def remove_start_and_end_time(
        self, df, start_margin=2 * 60 * 1000, end_margin=2 * 60 * 1000
    ):
        df = df[df["up_event"] != "Unidentified"].reset_index(drop=True)
        result_df = []
        grouped_df = df.groupby("id_encoded")

        for _, log in tqdm(grouped_df):
            valid_events = log[
                (log.activity != "Nonproduction")
                | (log.up_event != "Shift")
                | (log.up_event != "CapsLock")
            ].down_time.values
            if len(valid_events) == 0:
                continue
            log = log[
                (log.down_time > valid_events.min() - start_margin)
                & (log["down_time"] <= valid_events.max() + end_margin)
            ].copy()
            log["event_id"] = range(len(log))
            result_df.append(log)

        result = pd.concat(result_df, ignore_index=True)

        return result

    def remove_rest_time(
        self, df, time_margin=1 * 60 * 1000, action_margin=5 * 60 * 1000
    ):
        down_times, up_times = [], []
        prev_idx = -1
        result_df = df[["id_encoded", "down_time", "up_time"]].values
        for row in tqdm(result_df):
            idx, down_time, up_time = int(row[0]), int(row[1]), int(row[2])
            if prev_idx != idx:
                prev_down_time = down_time
                prev_corrected_down_time = 0
            gap_down_time = np.clip(down_time - prev_down_time, 0, time_margin)
            action_time = np.clip(up_time - down_time, 0, action_margin)

            new_down_time = prev_corrected_down_time + gap_down_time
            new_up_time = new_down_time + action_time
            down_times.append(new_down_time)
            up_times.append(new_up_time)
            prev_idx, prev_corrected_down_time, prev_down_time = (
                idx,
                new_down_time,
                down_time,
            )
        df["down_time"], df["up_time"] = down_times, up_times
        return df


if __name__ == "__main__":
    preprocessor = Preprocess()
    df = pd.read_csv("data/train_logs.csv")
    df = preprocessor.label_encoding(df)
    df = preprocessor.remove_start_and_end_time(df)
    df = preprocessor.remove_rest_time(df)
    df.to_csv("data/train_logs_clean.csv", index=False)
