from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

house_df = pd.read_csv("static/data/data.csv")
house_df = house_df.dropna()

X_house = np.array(house_df[["bedrooms", "bathrooms", "sqft_living", "sqft_lot",
                               "floors", "waterfront", "view", "condition",
                               "sqft_above", "sqft_basement", "yr_built", "yr_renovated"]])
Y_house = np.array(house_df["price"]).reshape(-1, 1)

def normalize(X):
    mean = np.mean(X, axis=0)
    std  = np.std(X, axis=0)
    return (X - mean) / std, mean, std

x_house, house_mean_X, house_std_X = normalize(X_house)
y_house, house_mean_Y, house_std_Y = normalize(Y_house)

def train(X, Y):
    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0
    for i in range(10000):
        yhat = np.dot(X, w) + b
        dw   = (1/m) * np.dot(X.T, (yhat - Y))
        db   = (1/m) * np.sum(yhat - Y)
        w    = w - 0.1 * dw
        b    = b - 0.1 * db
    return w, b

house_w, house_b = train(x_house, y_house)

titanic_df = pd.read_csv("static/data/tested.csv")
titanic_df = titanic_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]].dropna()

titanic_df["Sex"] = titanic_df["Sex"].map({"male": 0, "female": 1})

X_titanic = np.array(titanic_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]])
Y_titanic = np.array(titanic_df["Survived"]).reshape(-1, 1)

x_titanic, titanic_mean_X, titanic_std_X = normalize(X_titanic)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic(X, Y, lr=0.1, epochs=10000):
    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0
    for i in range(epochs):
        yhat = sigmoid(np.dot(X, w) + b)
        dw   = (1/m) * np.dot(X.T, (yhat - Y))
        db   = (1/m) * np.sum(yhat - Y)
        w    = w - lr * dw
        b    = b - lr * db
    return w, b

titanic_w, titanic_b = train_logistic(x_titanic, Y_titanic)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/House_Price_Predictor", methods=['GET','POST'])
def House_Price_Predictor():
    price = None
    if request.method == "POST":

        bedrooms      = float(request.form.get("bedroom"))
        bathrooms     = float(request.form.get("bathroom"))
        sqft_living   = float(request.form.get("sqft_living"))
        sqft_lot      = float(request.form.get("sqft_lot"))
        floors        = float(request.form.get("floors"))
        waterfront    = float(request.form.get("waterfront"))
        view          = float(request.form.get("view"))
        condition     = float(request.form.get("condition"))
        sqft_above    = float(request.form.get("sqft_above"))
        sqft_basement = float(request.form.get("sqft_basement"))
        yr_built      = float(request.form.get("yr_built"))
        yr_renovated  = float(request.form.get("yr_renovated"))

        X_input = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot,
                             floors, waterfront, view, condition,
                             sqft_above, sqft_basement, yr_built, yr_renovated]])

        X_input = (X_input - house_mean_X) / house_std_X
        y_pred = np.dot(X_input, house_w) + house_b
        price = (y_pred * house_std_Y) + house_mean_Y
        price = int(price[0][0])

    return render_template("House Price Predictor.html", price=price)

@app.route("/Titanic_ML_Predictor", methods=['GET','POST'])
def titanic_ml_predictor():
    survived    = None
    probability = None

    if request.method == "POST":

        pclass = float(request.form.get("pclass"))
        sex    = float(request.form.get("sex"))
        age    = float(request.form.get("age"))
        sibsp  = float(request.form.get("sibsp"))
        parch  = float(request.form.get("parch"))
        fare   = float(request.form.get("fare"))

        X_input = np.array([[pclass, sex, age, sibsp, parch, fare]])
        X_input = (X_input - titanic_mean_X) / titanic_std_X
        prob    = float(sigmoid(np.dot(X_input, titanic_w) + titanic_b)[0][0])

        survived    = 1 if prob >= 0.5 else 0
        probability = round(prob * 100)

    return render_template("Titanic-ML-Predictor.html",
                           survived=survived,
                           probability=probability)

@app.route("/Loan_Approval_Predictor", methods=['GET','POST'])
def Decision_Tree():
    approved= None
    if request.method == "POST":
        person_age        = float(request.form.get("person_age"))
        person_gender     = request.form.get("person_gender")
        person_education  = request.form.get("person_education")
        person_income     = float(request.form.get("person_income"))
        person_emp_exp    = float(request.form.get("person_emp_exp"))
        home_ownership    = request.form.get("person_home_ownership")
        loan_amnt         = float(request.form.get("loan_amnt"))
        loan_intent       = request.form.get("loan_intent")
        loan_int_rate     = float(request.form.get("loan_int_rate"))
        loan_pct_income   = float(request.form.get("loan_percent_income"))
        cred_hist         = float(request.form.get("cb_person_cred_hist_length"))
        credit_score      = float(request.form.get("credit_score"))
        prev_defaults     = request.form.get("previous_loan_defaults_on_file")

        approved = 0  # default

    # 🌳 START TREE LOGIC

        if prev_defaults == "No":

            if loan_pct_income < 0.25:

                if loan_int_rate < 14.0:

                    if person_income < 24599:

                        if loan_pct_income < 0.13:

                            if loan_intent == "EDUCATION":

                                if person_age < 31:

                                    if loan_pct_income < 0.12:

                                        if credit_score < 596:
                                            approved = 1
                                        else:
                                            if loan_int_rate < 13.57:
                                                if person_age < 22:
                                                    if person_gender == "female":
                                                        approved = 1
                                                    else:
                                                        approved = 0
                                                else:
                                                    approved = 0
                                            else:
                                                approved = 1
                                    else:
                                        approved = 1
                                else:
                                    approved = 1

                            elif loan_intent == "PERSONAL":

                                if person_income < 24064:

                                    if loan_amnt < 1400:
                                        approved = 1
                                    else:
                                        if loan_int_rate < 11.21:

                                            if home_ownership == "OWN":
                                                approved = 0
                                            elif home_ownership == "MORTGAGE":
                                                approved = 1
                                            else:
                                                approved = 0
                                        else:
                                            if loan_amnt < 2400:
                                                approved = 1
                                            else:
                                                approved = 0
                                else:
                                    approved = 1

                            elif loan_intent == "MEDICAL":

                                if person_age < 24:
                                    if person_emp_exp < 1:
                                        approved = 1
                                    else:
                                        if person_income < 17618:
                                            approved = 1
                                        else:
                                            approved = 0
                                else:
                                    approved = 1

                            else:
                                approved = 1

                        else:
                            approved = 1

                    else:
                        approved = 1

                else:
                    approved = 0

            else:
                approved = 0

        else:
            approved = 0

    return render_template("Loan Approval Predictor.html", approved=approved)

if __name__ == "__main__":
    app.run(debug=True)