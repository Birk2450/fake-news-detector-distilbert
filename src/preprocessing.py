import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    print("Running preprocessing pipeline...")

    # Load data
    fake = pd.read_csv('../data/Fake.csv')
    true = pd.read_csv('../data/True.csv')

    import re

    # --- Remove Reuters source tags from TRUE articles (to avoid overfitting on the source) ---
    if 'text' in true.columns:
        true['text'] = (
            true['text']
            .astype(str)
            # [Reuters] or (Reuters), with optional spaces
            .str.replace(r'[\(\[]\s*Reuters\s*[\)\]]', '', regex=True)
            # patterns like "Reuters - " at the start
            .str.replace(r'^\s*Reuters\s*-\s*', '', regex=True)
        )

    if 'title' in true.columns:
        true['title'] = (
            true['title']
            .astype(str)
            .str.replace(r'[\(\[]\s*Reuters\s*[\)\]]', '', regex=True)
            .str.replace(r'^\s*Reuters\s*-\s*', '', regex=True)
        )

    # Label
    fake['label'] = 0   # 0 = Fake
    true['label'] = 1   # 1 = Real

    # Merge + light clean + concat title+text
    df = pd.concat([fake, true], ignore_index=True)
    df['title'] = df['title'].fillna('')
    df['text']  = df['text'].fillna('')
    df['text_full'] = (df['title'] + ' ' + df['text']).str.strip()

    # Make the stratified splitting (train/val/test = 80/10/10)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['label'],
        random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['label'],
        random_state=42
    )

    print({'train': len(train_df), 'val': len(val_df), 'test': len(test_df)})
    print('label ratios:',
          train_df['label'].mean(),
          val_df['label'].mean(),
          test_df['label'].mean())

    # Save merged and splits
    df.to_csv('../data/merged_dataset.csv', index=False)
    train_df.to_csv('../data/train.csv', index=False)
    val_df.to_csv('../data/val.csv', index=False)
    test_df.to_csv('../data/test.csv', index=False)

    print("✅ Datasets saved to ../data/")
    print("✅ Done.")
