from flask import Flask, request, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model safely
model = None
model_path = "model.pkl"

if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
else:
    print("Error: 'model.pkl' not found. Ensure the pickle file is renamed to 'model.pkl'.")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        * { box-sizing: border-box; }
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 24px;
            color: #f8fafc;
        }

        .card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 36px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            width: 100%;
            max-width: 900px;
        }

        h2 {
            text-align: center;
            color: #ffffff;
            margin-top: 0;
            font-weight: 700;
            font-size: 28px;
            letter-spacing: -0.5px;
            margin-bottom: 8px;
        }

        .subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 14px;
            margin-bottom: 28px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }

        .form-group { display: flex; flex-direction: column; }

        label {
            font-weight: 600;
            font-size: 11px;
            margin-bottom: 6px;
            color: #cbd5e1;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        input {
            width: 100%;
            padding: 10px 14px;
            border: 1px solid #334155;
            border-radius: 8px;
            font-size: 14px;
            transition: all 0.2s ease;
            background-color: #0f172a;
            color: #ffffff;
        }

        input:focus {
            outline: none;
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
        }

        button {
            grid-column: 1 / -1;
            margin-top: 12px;
            padding: 14px;
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 15px;
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: all 0.2s ease;
        }

        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.35);
        }

        .result-box {
            margin-top: 24px;
            padding: 20px;
            text-align: center;
            border-radius: 12px;
            font-size: 20px;
            font-weight: 700;
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(52, 211, 153, 0.3);
            animation: fadeIn 0.4s ease-out;
        }

        .error-box {
            margin-top: 24px;
            padding: 16px;
            text-align: center;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(248, 113, 113, 0.3);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="card">
        <h2>House Price Predictor</h2>
        <div class="subtitle">Enter property characteristics to estimate market value</div>

        <form action="/predict" method="POST" class="form-grid">
            <div class="form-group"><label>Bedrooms</label><input type="number" name="bedrooms" required placeholder="e.g. 3" step="any"></div>
            <div class="form-group"><label>Bathrooms</label><input type="number" name="bathrooms" required placeholder="e.g. 2" step="any"></div>
            <div class="form-group"><label>Living Area (sqft)</label><input type="number" name="living_area" required placeholder="e.g. 2000" step="any"></div>
            <div class="form-group"><label>Lot Area (sqft)</label><input type="number" name="lot_area" required placeholder="e.g. 5000" step="any"></div>
            <div class="form-group"><label>Floors</label><input type="number" name="floors" required placeholder="e.g. 1.5" step="any"></div>
            <div class="form-group"><label>Waterfront (0/1)</label><input type="number" name="waterfront" required placeholder="0 or 1" min="0" max="1" step="any"></div>
            <div class="form-group"><label>Views Count</label><input type="number" name="views" required placeholder="e.g. 0 to 4" step="any"></div>
            <div class="form-group"><label>House Condition</label><input type="number" name="condition" required placeholder="e.g. 1 to 5" step="any"></div>
            <div class="form-group"><label>House Grade</label><input type="number" name="grade" required placeholder="e.g. 7" step="any"></div>
            <div class="form-group"><label>Area Excl. Basement</label><input type="number" name="area_above" required placeholder="e.g. 1500" step="any"></div>
            <div class="form-group"><label>Basement Area</label><input type="number" name="area_basement" required placeholder="e.g. 500" step="any"></div>
            <div class="form-group"><label>Built Year</label><input type="number" name="built_year" required placeholder="e.g. 1995" step="any"></div>
            <div class="form-group"><label>Renovation Year</label><input type="number" name="renovation_year" required placeholder="0 if none" step="any"></div>
            <div class="form-group"><label>Lot Area Renovated</label><input type="number" name="lot_area_renov" required placeholder="e.g. 5000" step="any"></div>
            <div class="form-group"><label>Schools Nearby</label><input type="number" name="schools_nearby" required placeholder="e.g. 3" step="any"></div>
            <div class="form-group"><label>Airport Distance</label><input type="number" name="airport_distance" required placeholder="e.g. 15" step="any"></div>

            <button type="submit">Calculate Estimated Value</button>
        </form>

        {% if prediction_text %}
            <div class="result-box">{{ prediction_text }}</div>
        {% endif %}

        {% if error_text %}
            <div class="error-box">{{ error_text }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, error_text="Model file not found. Please upload model.pkl.")

    try:
        features = [
            float(request.form.get("bedrooms")),
            float(request.form.get("bathrooms")),
            float(request.form.get("living_area")),
            float(request.form.get("lot_area")),
            float(request.form.get("floors")),
            float(request.form.get("waterfront")),
            float(request.form.get("views")),
            float(request.form.get("condition")),
            float(request.form.get("grade")),
            float(request.form.get("area_above")),
            float(request.form.get("area_basement")),
            float(request.form.get("built_year")),
            float(request.form.get("renovation_year")),
            float(request.form.get("lot_area_renov")),
            float(request.form.get("schools_nearby")),
            float(request.form.get("airport_distance"))
        ]

        final_features = np.array([features], dtype=float)
        prediction = model.predict(final_features)

        predicted_value = float(np.ravel(prediction)[0])
        output = round(predicted_value, 2)

        return render_template_string(
            HTML_TEMPLATE,
            prediction_text=f"Estimated House Value: ${output:,.2f}"
        )

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error_text=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
