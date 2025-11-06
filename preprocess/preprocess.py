import pandas as pd
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

    def remove_procrastination_time(self, df, start_margin=2*60*1000, end_margin=2*60*1000):
        df = df[df['up_event'] != 'Unidentified'].reset_index(drop=True)
        result_df = []
        grouped_df = df.groupby('id_encoded')

        for _, log in tqdm(grouped_df):
            valid_events = log[(log.activity != 'Nonproduction') | (
                log.up_event != 'Shift') | (log.up_event != 'CapsLock')].down_time.values
            if len(valid_events) == 0:
                continue
            log = log[(log.down_time > valid_events.min() - start_margin)
                      & (log['down_time'] <= valid_events.max() + end_margin)].copy()
            log['event_id'] = range(len(log))
            result_df.append(log)

        result = pd.concat(result_df, ignore_index=True)

        return result


train_log_df = pd.read_csv("data/train_logs.csv")
train_scores_df = pd.read_csv("data/train_scores.csv")
test_log_df = pd.read_csv("data/test_logs.csv")

preprocessor = Preprocess()
train_log_df = preprocessor.label_encoding(train_log_df)
train_log_df = preprocessor.remove_procrastination_time(train_log_df)
train_log_df.head(5)
