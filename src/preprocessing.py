import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    print("Running preprocessing pipeline...")

    # Load data
    fake = pd.read_csv('../data/Fake.csv')
    true = pd.read_csv('../data/True.csv')

    print("Preprocessing done.")

    train, test = train_test_split(fake, test_size=0.2)
    # Label
    fake['label'] = 0   # 0 = Fake
    true['label'] = 1   # 1 = Real

    # Merge + light clean + concat title+text
    df = pd.concat([fake, true], ignore_index=True) #combine the two dataframes into one
    df['title'] = df['title'].fillna('')
    df['text']  = df['text'].fillna('') # these fill any missing (NaN) values in the title or text columns with empty strings
    df['text_full'] = (df['title'] + ' ' + df['text']).str.strip() # creates a new column text_full with merged title and article body into one string

    # Make the stratified splitting (train/val/test = 80/10/10)
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    val_df,   test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    print({ 'train': len(train_df), 'val': len(val_df), 'test': len(test_df) }) # print how many samples are in eachsubset
    print('label ratios:', train_df['label'].mean(), val_df['label'].mean(), test_df['label'].mean()) # and print  average label value for each subset

    # Save merged and splits
    df.to_csv('../data/merged_dataset.csv', index=False)
    train_df.to_csv('../data/train.csv', index=False)
    val_df.to_csv('../data/val.csv', index=False)
    test_df.to_csv('../data/test.csv', index=False)
    print("✅ Datasets saved to ../data/")
    print("✅ Done.")