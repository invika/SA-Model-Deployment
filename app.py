from flask import Flask, render_template, request
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import torch
from transformers import BertTokenizer, BertForSequenceClassification

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

# Helper functions
def preprocess_text(input_text):
    """Clean and preprocess the input text."""
    cleaned_text = str(input_text).lower()
    cleaned_text = re.sub(r'https?://\S+|www\.\S+', '', cleaned_text)  # Remove URLs
    cleaned_text = re.sub(r'<.*?>', '', cleaned_text)  # Remove HTML tags
    cleaned_text = re.sub(r'\n', ' ', cleaned_text)  # Replace newlines with spaces
    cleaned_text = re.sub(r'\w*\d\w*', '', cleaned_text)  # Remove words with numbers
    table = str.maketrans('', '', string.punctuation)
    cleaned_text = cleaned_text.translate(table)  # Remove punctuation
    return cleaned_text

def process_input_sentence(input_sentence):
    processed_sentence = preprocess_text(input_sentence)
    tokenized_sentence = tweet_tokenizer.texts_to_sequences([processed_sentence])
    padded_sentence = pad_sequences(tokenized_sentence, maxlen=max_length, padding='post')
    return padded_sentence

def predict_with_bert(sentence):
    sentence = preprocess_text(sentence)
    inputs = tokenizer_bert(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model_bert(**inputs)
        logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    probability = torch.softmax(logits, dim=1)[0][predicted_class].item()
    return "REAL" if predicted_class == 1 else "FAKE", probability

# Load models
model_lstm = tf.keras.models.load_model('model_LSTM23_Final.h5')
model_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_bert.load_state_dict(torch.load('bert_finetuned_model.pth', map_location=torch.device('cpu')))
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')

# Load dataset
train_dataset = pd.read_csv("train.csv", encoding="latin-1")
test_dataset = pd.read_csv("test.csv", encoding="latin-1")

# Preprocess dataset
lemmatizer = WordNetLemmatizer()
stop_words_list = stopwords.words('english')
train_dataset['text_clean'] = train_dataset['text'].apply(preprocess_text)
test_dataset['text_clean'] = test_dataset['text'].apply(preprocess_text)

tweet_tokenizer = tf.keras.preprocessing.text.Tokenizer()
tweet_tokenizer.fit_on_texts(train_dataset['text_clean'])
max_length = 23

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction_text = None
    prediction_label = None
    probability = None

    if request.method == 'POST':
        text = request.form['text']
        model_type = request.form['model_type']

        if model_type == "lstm":
            processed_text = process_input_sentence(text)
            prediction = model_lstm.predict(processed_text)
            prediction_label = "REAL" if prediction >= 0.6 else "FAKE"
            probability = prediction[0][0] if prediction >= 0.6 else 1 - prediction[0][0]
        else:
            prediction_label, probability = predict_with_bert(text)

        probability = f"{probability:.2f}"

    return render_template(
        'index.html',
        prediction_text=prediction_text,
        prediction_label=prediction_label,
        prediction_probability=probability
    )

if __name__ == '__main__':
    app.run(debug=True, port=8199)