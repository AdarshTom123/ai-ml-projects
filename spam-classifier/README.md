# Email Spam Classifier

A machine learning model that classifies emails as **spam** or **ham** (legitimate) using the Naive Bayes algorithm.

## How it works
1. Loads a labeled dataset of emails
2. Cleans the text (lowercase, removes special characters)
3. Converts text into numerical features using CountVectorizer (bag-of-words)
4. Trains a Multinomial Naive Bayes classifier
5. Predicts whether a new email is spam or ham

## Tech Stack
- Python
- scikit-learn (MultinomialNB, CountVectorizer)
- Pandas

## How to run

Install the required libraries:

    pip install pandas scikit-learn

Then run the script:

    python spam_classifier.py
