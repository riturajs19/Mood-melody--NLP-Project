# Mood Melody – Emotion-Based Music Recommendation System

## Overview

This project focuses on building an end-to-end system that can understand human emotions from text and recommend music accordingly. The idea was to solve a simple but practical problem: choosing the right music based on how someone feels at a given moment.

The system takes user input in the form of text, predicts the underlying emotion using a deep learning model, and then connects to the Spotify API to fetch relevant songs. The final system is deployed as an interactive Streamlit application.

---

## Dataset and Understanding

The dataset used contains text samples labeled with emotions. Each record follows the format:

sentence ; emotion

Example:
i didnt feel humiliated ; sadness

The dataset is already divided into:

* Training set
* Validation set
* Test set

This structure made it easier to:

* Train the model
* Tune hyperparameters
* Evaluate performance on unseen data

Before building the model, the dataset was explored to understand:

* Distribution of emotions
* Possible class imbalance
* Nature of sentences

---

## Text Preprocessing

The preprocessing was kept simple and focused on preserving meaning:

* Converted text to lowercase
* Removed punctuation and special characters
* Tokenized sentences into words
* Removed stopwords

Example:

"I am feeling very happy today"
→ ["feeling", "happy", "today"]

This ensures that the model focuses on meaningful words rather than noise.

---

## Tokenization and Sequence Preparation

Since neural networks require numerical input:

* A tokenizer was used to build a vocabulary
* Words were converted into integer sequences
* All sequences were padded to the same length

This step standardizes input and allows batch processing.

---

## Word Embeddings (GloVe)

Instead of training embeddings from scratch, pretrained GloVe embeddings were used.

Reasons for choosing GloVe:

* Captures global semantic relationships
* Works well on smaller datasets
* Provides stable representations for emotion-related tasks

An embedding matrix was created to map tokenizer indices to GloVe vectors.

---

## Model Architecture (BiLSTM)

A Bidirectional LSTM model was used to capture context from both directions in a sentence.

Architecture:

* Embedding layer initialized with GloVe
* Stacked BiLSTM layers
* Dropout layers for regularization
* Dense softmax output layer for classification into six emotions

This helps the model understand context such as negation and tone.

---

## Training Strategy

During training:

* Early stopping was used to prevent overfitting
* Validation loss was monitored instead of only training accuracy
* Batch size was kept small for stable learning

The model converged smoothly and achieved strong performance.

---

## Evaluation

The model was evaluated using:

* Accuracy
* Precision, Recall, F1-score
* Confusion matrix (both raw and normalized)

This helped identify:

* Which emotions were frequently confused
* Model strengths and weaknesses
* Real-world ambiguity between emotions

---

## Prediction Pipeline

A prediction pipeline was created to:

* Take raw user text
* Apply the same preprocessing steps
* Convert text into padded sequences
* Pass it through the trained model
* Map output to the corresponding emotion

This allows the system to work on completely new inputs.

---

## Spotify API Integration

To make the system practical:

* The predicted emotion is mapped to a music query
* Spotify API is used to fetch songs based on that query
* Songs are returned with title, artist, and link

This transforms the model from a classification system into a recommendation system.

---

## Deployment (Streamlit)

The complete system was deployed using Streamlit.

The application allows users to:

* Enter a sentence describing their mood
* View the predicted emotion
* Get music recommendations in real time

This step converts the model into an interactive and usable product.

---

## Key Takeaways

* Built a complete NLP pipeline from preprocessing to deployment
* Used pretrained embeddings to improve performance
* Designed a BiLSTM model for contextual understanding
* Integrated machine learning with an external API (Spotify)
* Deployed the system as a real-time interactive application

---

## Future Improvements

* Use transformer-based models such as BERT
* Improve emotion classification for edge cases
* Enhance recommendation logic using user history
* Add audio preview and richer UI

---

## References

Dataset: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp
Spotify API Documentation: https://developer.spotify.com/documentation/web-api

