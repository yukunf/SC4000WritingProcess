import pandas as pd
import os


def merge_preprocessed_data(data_dir='data', dataset_name='train'):
    dataset_behaviour = pd.read_csv(os.path.join(
        data_dir, f'{dataset_name}_behaviour_features.csv'))
    dataset_text = pd.read_csv(os.path.join(
        data_dir, f'{dataset_name}_text_features.csv'))
    dataset_operations = pd.read_csv(os.path.join(
        data_dir, f'{dataset_name}_operations.csv'))
    dataset_essay = pd.read_csv(os.path.join(
        data_dir, f'{dataset_name}_logs_extracted_text.csv'))
    dataset_essay = dataset_essay.iloc[:, 0:2]

    # merge on 'id'
    merged = dataset_behaviour.merge(dataset_text, on='id', how='inner')
    merged = merged.merge(dataset_operations, on='id', how='inner')
    merged = merged.merge(dataset_essay, on='id', how='inner')

    if dataset_name == 'train':
        scores = pd.read_csv(os.path.join(data_dir, 'train_scores.csv'))
        merged = merged.merge(scores, on='id', how='inner')

    output_path = os.path.join(data_dir, f'{dataset_name}_preprocessed.csv')
    merged.to_csv(output_path, index=False)

    return merged


if __name__ == '__main__':
    merge_preprocessed_data("data", "train")
    merge_preprocessed_data("data", "test")
