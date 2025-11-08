import pandas as pd
import os


def merge_preprocessed_data(data_dir='data', dataset_name='train'):
    dataset_behaviour = pd.read_csv(os.path.join(
        data_dir, f'{dataset_name}_behaviour_features.csv'))
    dataset_text = pd.read_csv(os.path.join(
        data_dir, f'{dataset_name}_text_features.csv'))
    dataset_tfidf_text = pd.read_csv(
        os.path.join(data_dir, f'{dataset_name}_tfidf_text.csv'))
    dataset_tfidf_operation = pd.read_csv(
        os.path.join(data_dir, f'{dataset_name}_tfidf_operation.csv'))

    # merge on 'id'
    merged = dataset_behaviour.merge(dataset_text, on='id', how='inner')

    # rename column name
    tfidf_text_renamed = dataset_tfidf_text.rename(
        columns={col: f'tfidf_text_{col}' if col != 'id' else col
                 for col in dataset_tfidf_text.columns}
    )
    tfidf_operation_renamed = dataset_tfidf_operation.rename(
        columns={col: f'tfidf_operation_{col}' if col != 'id' else col
                 for col in dataset_tfidf_operation.columns}
    )

    merged = merged.merge(tfidf_text_renamed, on='id', how='inner')
    merged = merged.merge(tfidf_operation_renamed, on='id', how='inner')

    if dataset_name == 'train':
        scores = pd.read_csv(os.path.join(data_dir, 'train_scores.csv'))
        merged = merged.merge(scores, on='id', how='inner')

    output_path = os.path.join(data_dir, f'{dataset_name}_preprocessed.csv')
    merged.to_csv(output_path, index=False)

    return merged


if __name__ == '__main__':
    merge_preprocessed_data("data", "train")
    merge_preprocessed_data("data", "test")
