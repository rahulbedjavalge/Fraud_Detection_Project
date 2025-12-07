import pandas as pd
import os

def load_data():
    """
    Loads transaction and identity data from the data folder.
    Returns merged dataframe.
    """
    base_path = os.path.join("data")

    transaction_path = os.path.join(base_path, "train_transaction.csv")
    identity_path = os.path.join(base_path, "train_identity.csv")

    print("Loading data...")
    transaction = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)

    print("Merging data...")
    df = transaction.merge(identity, on="TransactionID", how="left")

    print("Data loaded and merged successfully.")
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.shape)
