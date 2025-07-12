from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os


app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'titanic_model.pkl')
model = joblib.load(MODEL_PATH)


@app.route('/')
def home():
    index_path = os.path.join(os.path.dirname(
        __file__), 'templates/index.html')
    return render_template(index_path)


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()

    # Convert inputs into DataFrame
    df = pd.DataFrame([input_data])
    df['Age'] = float(df['Age'])
    df['SibSp'] = int(df['SibSp'])
    df['Parch'] = int(df['Parch'])
    df['Fare'] = float(df['Fare'])

    prediction = model.predict(df)[0]
    result = "Survived" if prediction == 1 else "Did not survive"
    return render_template(index_path=os.path.join(os.path.dirname(__file__), 'templates/index.html'), prediction_text=f'Prediction: {result}')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render sets PORT automatically
    app.run(host="0.0.0.0", port=port)
