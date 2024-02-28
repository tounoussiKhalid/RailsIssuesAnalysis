import json
from sklearn.model_selection import train_test_split
import re
import pandas as pd
from transformers import AutoTokenizer

def load_issues_data(filepath='data/issues.json'):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    return df
def clean_text(text):
    if text is None:
        return ""
    # Remove newline characters and excessive whitespace
    text = re.sub(r'\r\n', ' ', text)  # Replace newline characters with space
    text = re.sub(r'\s+', ' ', text)   # Replace multiple spaces with a single space
    # Add additional cleaning steps here if needed
    return text.strip()

def eda(df):
    print("Number of elements with labels empty before cleaning:", df['labels'].apply(lambda x: x is None or x == []).sum())
    print("Number of elements with body null before cleaning:", df['body'].isnull().sum())
    print("Number of elements with user null before cleaning:", df['user'].isnull().sum())
    print("Number of elements with title null before cleaning:", df['title'].isnull().sum())
    # Filter and print titles where 'body' is null
    titles_with_body_null = df[df['body'].isnull()]['title']
    print("Titles of rows with body null:")
    print(titles_with_body_null)
def preprocess_issues(df):
    df['labels'] = df['labels'].apply(lambda x: x if isinstance(x, list) and x != [] else ['Other'])

    df['description'] = df['title'] + ' ' + df['body'].fillna('')
    df['description'] = df['description'].apply(clean_text)
    # Select only the columns of interest
    df = df[[ 'description', 'labels' ]]
    return df

def tokenize_texts(texts, tokenizer_name='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return tokenized_texts

if __name__ == "__main__":
    df = load_issues_data()
    # Preprocess the issues data
    eda(df)
    #df_preprocessed = preprocess_issues(df)
    #print(df_preprocessed.columns )

    # Now you can print or use the preprocessed data frame as needed
