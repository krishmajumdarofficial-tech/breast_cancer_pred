from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

model_path = Path(__file__).resolve().parent / 'model.pkl'
with model_path.open('rb') as model_file:
    model = pickle.load(model_file)

#flask.app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    features = request.form.get('feature', '').strip()
    expected_features = getattr(model, 'n_features_in_', 31)

    try:
        features_lst = [value.strip() for value in features.split(',')]
        if not features or any(value == '' for value in features_lst):
            raise ValueError('Enter comma-separated numeric values.')
        if len(features_lst) != expected_features:
            raise ValueError(f'Enter exactly {expected_features} values, separated by commas.')

        np_features = np.array(features_lst, dtype=np.float32)
        if not np.isfinite(np_features).all():
            raise ValueError('All values must be finite numbers.')
    except ValueError as error:
        return render_template('index.html', error=str(error), features=features), 400

    pred = model.predict(np_features.reshape(1, -1))

    output = ["cancerous" if pred[0] == 1 else "non-cancerous"]
    return render_template('index.html', message = output)
#python main
if __name__ == '__main__':
    app.run(debug=True)