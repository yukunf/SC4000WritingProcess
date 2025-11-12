import pandas as pd


def rebuild_text(grp):
    buf = []
    for op in grp['activity']:
        # Get first character: I(nput), R(emove), M(ove), P(aste)
        buf.append(op[0])
    return "".join(buf)


if __name__ == "__main__":
    # Load cleaned data
    print("Loading data...")
    df_train = pd.read_csv("data/train_logs_clean.csv")
    df_test = pd.read_csv("data/test_logs_clean.csv")

    # Extract only id and activity columns
    df_train = df_train[['id', 'activity']]
    df_test = df_test[['id', 'activity']]

    print(f"Train data shape: {df_train.shape}")
    print(f"Test data shape: {df_test.shape}")
    print(f"\nSample train data:\n{df_train.head(3)}")

    # Group by id and rebuild text (operation sequences)
    print("\nGenerating operation sequences for training data...")
    df_train_operations = (
        df_train.groupby('id')
        .apply(rebuild_text, include_groups=False)
        .reset_index(name='operation')
    )

    print("Generating operation sequences for test data...")
    df_test_operations = (
        df_test.groupby('id')
        .apply(rebuild_text, include_groups=False)
        .reset_index(name='operation')
    )

    # Save to CSV
    print("\nSaving to CSV files...")
    df_train_operations.to_csv("data/train_operations.csv", index=False)
    df_test_operations.to_csv("data/test_operations.csv", index=False)

    print(f"\nTrain operations saved: {df_train_operations.shape}")
    print(f"Test operations saved: {df_test_operations.shape}")
    print(f"\nSample train operations:\n{df_train_operations.head(3)}")
    print(f"\nSample test operations:\n{df_test_operations.head(3)}")
