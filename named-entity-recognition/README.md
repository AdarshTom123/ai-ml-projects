# Named Entity Recognition (NER)

An NLP program that automatically identifies and extracts named entities such as people, places, organizations, and dates from text.

## How it works
1. Loads a dataset of sentences
2. Cleans the text (lowercase, removes special characters)
3. Uses spaCy's NLP model to detect named entities
4. Labels each entity by type (PERSON, ORG, GPE, DATE, etc.)

## Example
Input: "Apple opened a new office in Dubai on Monday."
Output: [(Apple, ORG), (Dubai, GPE), (Monday, DATE)]

## Tech Stack
- Python
- spaCy
- Pandas

## How to run

1. Install libraries: pip install pandas spacy
2. Download the language model: python -m spacy download en_core_web_sm
3. Run the script: python ner_spacy.py
