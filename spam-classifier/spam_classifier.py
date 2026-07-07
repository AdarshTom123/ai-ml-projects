import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("spam.csv")
print(df.head())
print(df.columns)

# Basic text cleaning: lowercase and keep only letters/spaces
def clean_text(text):
    text = text.lower()
    cleaned = ""
    for ch in text:
        if ch.isalpha() or ch == " ":
            cleaned += ch
    return cleaned

df['clean_text'] = df['text'].apply(clean_text)

# Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test with a new email
new_email = ["Congratulations you have won a free prize click here to claim"]
new_clean = [clean_text(new_email[0])]
new_vector = vectorizer.transform(new_clean)
prediction = model.predict(new_vector)
print("Prediction:", prediction[0])
