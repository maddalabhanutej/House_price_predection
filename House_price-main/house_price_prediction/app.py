from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        data = {
            "area": int(request.form["area"]),
            "bedrooms": int(request.form["bedrooms"]),
            "bathrooms": int(request.form["bathrooms"]),
            "stories": int(request.form["stories"]),
            "mainroad": request.form["mainroad"],
            "guestroom": request.form["guestroom"],
            "basement": request.form["basement"],
            "hotwaterheating": request.form["hotwaterheating"],
            "airconditioning": request.form["airconditioning"],
            "parking": int(request.form["parking"]),
            "prefarea": request.form["prefarea"],
            "furnishingstatus": request.form["furnishingstatus"]
        }

        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
        return render_template("form.html", prediction=round(prediction, 2))

    return render_template("form.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
