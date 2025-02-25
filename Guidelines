# 📚 Natural Language Processing (NLP) Quick Guide

A collection of **useful NLP techniques** with **mini-code snippets** for quick reference. 🚀

---

## 📌 1. Tokenization (Splitting Text into Words)
```python
from nltk.tokenize import word_tokenize

text = "Hello! This is a simple NLP showcase."
tokens = word_tokenize(text)
print(tokens)
```

---

## 📌 2. Removing Stopwords
```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in tokens if word.lower() not in stop_words]
print(filtered_words)
```

---

## 📌 3. Stemming & Lemmatization
```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print(stemmer.stem("running"))  # Output: run
print(lemmatizer.lemmatize("running", pos='v'))  # Output: run
```

---

## 📌 4. Named Entity Recognition (NER)
```python
import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("Barack Obama was the 44th President of the United States.")
for ent in doc.ents:
    print(ent.text, ent.label_)
```

---

## 📌 5. Sentiment Analysis
```python
from textblob import TextBlob

text = "I absolutely love this product!"
sentiment = TextBlob(text).sentiment
print(sentiment.polarity)  # Output: 0.85 (Positive Sentiment)
```

---

## 📌 6. TF-IDF for Keyword Extraction
```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = ["I love NLP!", "NLP is amazing and powerful."]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

---

## 📌 7. Word Cloud Generation
```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = "NLP is fun and useful. AI and NLP are the future!"
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```

---

## 📌 8. Language Detection
```python
from langdetect import detect

print(detect("Bonjour, comment ça va?"))  # Output: 'fr' (French)
```

---

## 📌 9. Text Summarization
```python
from transformers import pipeline

summarizer = pipeline("summarization")
text = """Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans through language. It involves tasks like tokenization, translation, and sentiment analysis."""
print(summarizer(text, max_length=30, min_length=10, do_sample=False))
```

---

## 📌 10. Chatbot Response Generation
```python
from transformers import pipeline

chatbot = pipeline("text-generation", model="microsoft/DialoGPT-small")
response = chatbot("Hello, how are you?", max_length=50)
print(response)
```

---


