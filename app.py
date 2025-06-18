from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
from recommendation import prepare_recommendation_system, get_recommendations

# Load the model
churn_model = joblib.load("churn_model.pkl")

# Load customer data
df = pd.read_excel("Assignment_Data.xlsx")  # Replace with your dataset file path

# Prepare the recommendation system
rec_system = prepare_recommendation_system(df)

# Initialize Flask app
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    """Root endpoint to show the prediction result on the webpage."""
    try:
        prediction = None
        is_churn = None
        recommendations = None
        error = None

        if request.method == "POST":
            # Get customer data from the form
            try:
                data = {
                    "Gender": int(request.form["Gender"]),  # 0 = Male, 1 = Female
                    "ServiceUsage1": float(request.form["ServiceUsage1"]),
                    "ServiceUsage2": float(request.form["ServiceUsage2"]),
                    "ServiceUsage3": float(request.form["ServiceUsage3"]),
                    "MonthlyCharges": float(request.form["MonthlyCharges"]),
                    "Tenure": int(request.form["Tenure"]),
                }
            except KeyError as e:
                error = f"Missing input: {e.args[0]}"
                return render_template("home.html", error=error)

            # Derived features
            average_spend_per_month = data["MonthlyCharges"]
            average_service_usage = (
                data["ServiceUsage1"] + data["ServiceUsage2"] + data["ServiceUsage3"]
            ) / 3
            charges_tenure_interaction = data["MonthlyCharges"] * data["Tenure"]
            totalcharges_to_monthlycharges_ratio = data["Tenure"]

            # Combine input and derived features
            features = [
                data["Gender"],  # Include Gender
                data["ServiceUsage1"],
                data["ServiceUsage2"],
                data["ServiceUsage3"],
                data["MonthlyCharges"],
                data["Tenure"],
                average_spend_per_month,
                average_service_usage,
                charges_tenure_interaction,
                totalcharges_to_monthlycharges_ratio,
            ]

            # Convert to NumPy array and reshape
            features = np.array(features).reshape(1, -1)

            # Predict churn probability
            probability = churn_model.predict_proba(features)[0][1]
            prediction = round(probability, 2)

            # Check if the customer is likely to churn (threshold = 0.5)
            is_churn = probability > 0.5

            # Fetch recommendations if the customer is likely to churn
            if is_churn:
                customer_id = "CUST0011"  # Default customer ID
                recommendations = get_recommendations(customer_id, rec_system, df)

        return render_template(
            "home.html",
            prediction=prediction,
            is_churn=is_churn,
            recommendations=recommendations,
            error=error,
        )
    except Exception as e:
        # Catch any unexpected errors
        return render_template("home.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
