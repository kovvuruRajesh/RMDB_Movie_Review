from flask import Flask

app = Flask(__name__)

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the trained model
model_path = 'trained_model.h5'
model = load_model(model_path)

# Load the tokenizer
tokenizer = Tokenizer(oov_token='<OOV>')
# Assuming you have your dataset available for fitting the tokenizer
# Adjust the path to your dataset
df = pd.read_csv('IMDB_dataset.csv')
tokenizer.fit_on_texts(df['review'])

@app.route("/", methods=["GET"])
def hello_world():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form.get('review')
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    prediction = model.predict(np.array(padded_sequence))
    sentiment = "Positive" if prediction[0] > 0.5 else "Negative"
    return render_template('index.html', review=review, sentiment=sentiment)



if __name__ == '__main__':
    app.run(debug='True')