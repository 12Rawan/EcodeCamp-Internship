from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd


def create_app():
    app = Flask(__name__)

    # Load model and scaler
    model_filename = 'SVM_model.pkl'
    scaler_filename = 'scaler.pkl'

    try:
        model = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)
    except Exception as e:
        print(f"Error loading model or scaler: {e}")

    # HTML form
    HTML_FORM = """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predict Form</title>
</head>
<body>
    <h1>Predict Survival</h1>
    <form id="predict-form">
        <label for="Pclass">Pclass:</label>
        <input type="number" id="Pclass" name="Pclass" required><br><br>

        <label for="Sex">Sex (0 for male, 1 for female):</label>
        <input type="number" id="Sex" name="Sex" required><br><br>

        <label for="SibSp">SibSp:</label>
        <input type="number" id="SibSp" name="SibSp" required><br><br>

        <label for="Parch">Parch:</label>
        <input type="number" id="Parch" name="Parch" required><br><br>

        <button type="submit">Submit</button>
    </form>

    <h2>Response:</h2>
    <pre id="response"></pre>

    <script>
        document.getElementById('predict-form').addEventListener('submit', function(event) {
            event.preventDefault();

            const formData = new FormData(event.target);
            const data = {
                Pclass: formData.get('Pclass'),
                Sex: formData.get('Sex'),
                SibSp: formData.get('SibSp'),
                Parch: formData.get('Parch')
            };

            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('response').textContent = `Survived: ${data.Survived}`;
            })
            .catch(error => console.error('Error:', error));
        });
    </script>
</body>
</html>
    """

    @app.route('/')
    def index():
        return render_template_string(HTML_FORM)

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()
        input_df = pd.DataFrame([data])
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)
        return jsonify({'Survived': int(prediction[0])})

    return app
