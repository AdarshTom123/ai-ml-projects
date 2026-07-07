import pandas as pd
import spacy

# Load spaCy's English NLP model
nlp = spacy.load("en_core_web_sm")

# Load the dataset
df = pd.read_csv("dataset.csv")
print(df.head())
print(df.columns)
print("Total samples:", len(df))
print("Missing values:", df['text'].isnull().sum())

# Remove rows with missing text
df = df.dropna(subset=['text'])

# Basic text cleaning
def clean_text(text):
    text = text.lower()
    cleaned = ""
    for ch in text:
        if ch.isalpha() or ch == " ":
            cleaned += ch
    return cleaned

df['cleaned_text'] = df['text'].apply(clean_text)

# Extract named entities (people, places, organizations, dates)
def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities

df['entities'] = df['cleaned_text'].apply(extract_entities)
print(df[['text', 'cleaned_text', 'entities']].head())
