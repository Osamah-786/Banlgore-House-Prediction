from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load cleaned data
data = pd.read_csv("Cleaned_data.csv")

# Load trained model
model = pickle.load(open("RidgeModel.pkl", "rb"))


@app.route("/")
def index():
    locations = sorted(data["location"].unique())

    return render_template(
        "index.html",
        locations=locations
    )


@app.route("/predict", methods=["POST"])
def predict():
    location = request.form.get("location")
    bhk = int(request.form.get("bhk"))
    bath = int(request.form.get("bath"))
    sqft = float(request.form.get("sqft"))

    # create size column
    size = str(bhk) + " BHK"

    input_data = pd.DataFrame(
        [[location, size, sqft, bath, bhk]],
        columns=["location", "size", "total_sqft", "bath", "bhk"]
    )

    prediction = model.predict(input_data)[0]

    return str(round(prediction, 2))


if __name__ == "__main__":
    app.run(debug=True, port=5001)