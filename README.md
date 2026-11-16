# Mood-melody--NLP-Project

🎧 MoodMelody

Emotion-Aware Music Recommendation using NLP & Deep Learning

## Why I Built This

I spend a lot of time listening to music while studying, coding, or just trying to unwind.
At some point I noticed a pattern, the hardest part wasn’t playing music, it was deciding what to play.
Am I feeling happy? Tired? A bit anxious but hopeful?
Those small “what should I listen to now?” moments kept breaking my flow.
So I decided to build MoodMelody, a system that:
understands how I’m feeling from text
predicts the underlying emotion
and (in the extended version) recommends music that matches that mood
This project became my way of combining NLP, deep learning, and practical system design into one end-to-end pipeline.

## What This Project Does 

Takes free-form text from a user
Predicts the emotion expressed in that text
Classifies it into one of six emotions
Evaluates the model rigorously using standard ML metrics
The emotions used are:
anger
love
fear
joy
sadness
surprise

## Dataset Used

I used a publicly available emotion-labeled NLP dataset, where each sample consists of:
sentence ; emotion
https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp

Example:
i didnt feel humiliated ; sadness

The dataset is already split into:
training set
validation set
test set

This allowed me to:
train the model properly
tune it using validation data
evaluate it on completely unseen examples

## Step-by-Step: What I Did and Why
### Data Exploration & Understanding

Before touching any model, I explored the dataset:
checked total number of samples
visualized emotion distribution
looked for class imbalance
This step helped me understand:
which emotions are dominant
which ones might be harder to predict
why accuracy alone is not enough

### Text Preprocessing

The dataset text was already fairly clean, so I kept preprocessing simple and intentional.
What I did:
converted text to lowercase
removed punctuation and special characters
tokenized sentences into words
removed stopwords to focus on emotionally meaningful words
Example:

"I am feeling very happy today"
→ ["feeling", "happy", "today"]


This ensures the model focuses on semantic content, not grammar noise.
### Tokenization & Sequence Preparation

Neural networks don’t understand words they understand numbers.

So I:
used a Keras Tokenizer to build a vocabulary
converted words into integer indices
padded all sequences to the same length
This step standardizes input so the model can process text in batches.

### Why I Chose GloVe Embeddings

Instead of training word embeddings from scratch, I used pretrained GloVe embeddings.
Why GloVe?
trained on billions of words
captures global semantic relationships
performs better than Word2Vec on small datasets
very stable for emotion-related tasks
I aligned GloVe vectors with my tokenizer to build an embedding matrix, which acts as a bridge between raw text and the neural network.

### Model Architecture (BiLSTM)

Emotion in language often depends on context from both directions.
Example:
"I am not happy today"
To handle this, I used a Bidirectional LSTM (BiLSTM).
Final architecture:
Embedding layer (initialized with GloVe)
Stacked BiLSTM layers
Dropout for regularization
Dense softmax layer for multi-class emotion prediction
This setup balances expressive power with generalization.

### Training Strategy

During training, I focused on:
early stopping to prevent overfitting
validation loss instead of just training accuracy
smaller batch size for stable gradients
The model converged smoothly and achieved strong performance on the validation set.

### Evaluation & Analysis

I evaluated the model using:
accuracy
precision, recall, F1-score
confusion matrix (both raw and normalized)
This helped me understand:
which emotions are confused with each other
where the model performs strongly
where ambiguity is natural (e.g. joy vs love, fear vs sadness)

### Custom Predictions

I tested the model on custom sentences to simulate real usage:

"I am feeling very happy today" → joy
"I feel lonely and sad" → sadness
"This situation makes me angry" → anger
"I am scared but hopeful" → fear

This step ensured the model behaves sensibly outside the dataset.

## Tech Stack

Language: Python
NLP: NLTK
Deep Learning: TensorFlow / Keras
Embeddings: GloVe
Data Handling: Pandas, NumPy
Visualization: Matplotlib, Seaborn

## What I Learned From This Project

Emotion classification is inherently ambiguous
High accuracy does not guarantee good performance on minority classes
Pretrained embeddings save time and improve stability
Debugging data pipelines is just as important as model design
Good ML systems are built step-by-step, not rushed end-to-end

## Production Engineering Upgrades (v2)

While keeping the core **Stacked BiLSTM (`128 -> 256 -> 128`) + 100d GloVe** architecture intact, the pipeline was upgraded to solve 5 real-world NLP & recommendation bottlenecks:

1. **Contraction Expansion & Negation-Aware Stopword Removal:**
   - Standard NLTK `stopwords.words('english')` strips polarity shifters like `not`, `never`, and `no` (turning *"I am not happy"* into `["happy"]`).
   - Added regex contraction normalization (`didn't` / `didnt` → `did not`) prior to punctuation removal and whitelisted negations (`STOPWORDS - NEGATION_WHITELIST`).
2. **Eliminated Data Leakage + `<OOV>` Tokenization:**
   - Fit the Keras `Tokenizer(num_words=15000, oov_token="<OOV>")` strictly on `x_train` (never on `x_test` or `x_validation`).
3. **Gaussian OOV Initialization in GloVe Matrix:**
   - Initialized out-of-vocabulary words with small Gaussian noise $\mathcal{N}(0, 0.05)$ while keeping index `0` (`<PAD>`) as strict zeros so LSTM gates distinguish unknown words from padding.
4. **Cost-Sensitive Learning (`class_weight='balanced'`):**
   - Applied balanced class weights during `model.fit()` to counter dataset imbalance where `surprise` (~4%) and `love` (~8%) are underrepresented compared to `joy` and `sadness`.
5. **Top-2 Softmax Probability Blending, Continuous Audio Feature Mapping & Therapy Mode:**
   - Replaced blind `np.argmax` with **Top-2 Mixed Mood Blending** (detecting ambiguous states like *Sadness + Fear*).
   - Mapped the 6-class softmax distribution into **Russell's 2D Valence-Energy Audio Space** (`Target Valence`, `Target Energy`, `Target Tempo`) with a psychological toggle between **Match My Mood (Iso-Principle)** and **Uplift / Soothe My Mood (Mood Regulation)**.


