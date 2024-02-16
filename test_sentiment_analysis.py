from tensorflow import keras
import numpy as np

# Load the model
model = keras.models.load_model('model.h5')

# Load the dictionary to convert text to integer sequences
word2idx = keras.datasets.imdb.get_word_index()

# Define a function to encode the review text
def encode(text):
    words = text.lower().replace(".", "").split()
    encoded = [1]  
    for word in words:
        if word in word2idx and word2idx[word] < 100:
            encoded.append(word2idx[word] + 3)
        else:
            encoded.append(2)  
    return encoded

# Define a function to get the predicted sentiment from the model
def predict_sentiment(text):
    encoded_text = encode(text)
    max_len = 200
    padded_text = keras.preprocessing.sequence.pad_sequences([encoded_text], maxlen=max_len)
    numpy_text = np.array(padded_text[0])
    pred = round(model.predict(numpy_text)[0][0])
    if pred == 0:
        return "Negative"
    else:
        return "Positive"

# Test the Sentiment Analysis model for a few reviews
reviews = ["The movie was great! I loved it.",
           "The acting was terrible and the plot was boring.",
           "I'm not sure how I feel about this movie.",
           "One of the best movies I've ever seen! Highly recommend.",
           "Complete waste of time. Extremely disappointed.",
           "I found the movie to be average. Nothing special."]

for review in reviews:
    sentiment = predict_sentiment(review)
    print(f"\nReview: {review}")
    print(f"Sentiment: {sentiment}")
