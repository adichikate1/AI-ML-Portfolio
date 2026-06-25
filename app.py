import pandas as pd
from flask import Flask, render_template, request
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catboost import CatBoostClassifier
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px

app = Flask(__name__)

#--------------------------------------------------------------------------------------------------------------------------------------
# Home
#--------------------------------------------------------------------------------------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')

#--------------------------------------------------------------------------------------------------------------------------------------
# Home End
#--------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------
# House Price Predictor
#--------------------------------------------------------------------------------------------------------------------------------------

df_house = pd.read_csv("static/data/data.csv").drop(
    ["date", "street", "city", "statezip", "country"],
    axis=1
)

X_house = df_house.drop("price", axis=1)
y_house = df_house["price"]

X_house_train, X_house_test, y_house_train, y_house_test = train_test_split(
    X_house,
    y_house,
    test_size=0.2,
    random_state=42
)

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

model.fit(X_house_train, y_house_train)

test_pred = model.predict(X_house_test)

confidence = round(r2_score(y_house_test, test_pred) * 100, 2)

accuracy_range = round(
    mean_absolute_percentage_error(y_house_test, test_pred) * 100,
    2
)

@app.route("/house_price_predictor", methods=["GET", "POST"])
def house_price_predictor():

    price = None

    if request.method == "POST":

        input_data = pd.DataFrame([{
            "bedrooms": float(request.form["bedroom"]),
            "bathrooms": float(request.form["bathroom"]),
            "sqft_living": int(request.form["sqft_living"]),
            "sqft_lot": int(request.form["sqft_lot"]),
            "floors": float(request.form["floors"]),
            "waterfront": int(request.form["waterfront"]),
            "view": int(request.form["view"]),
            "condition": int(request.form["condition"]),
            "sqft_above": int(request.form["sqft_above"]),
            "sqft_basement": int(request.form["sqft_basement"]),
            "yr_built": int(request.form["yr_built"]),
            "yr_renovated": int(request.form["yr_renovated"])
        }])

        price = float(model.predict(input_data)[0])

    return render_template(
        "House Price Predictor.html",
        price=price,
        confidence=confidence,
        accuracy_range=accuracy_range
    )

#--------------------------------------------------------------------------------------------------------------------------------------
# House Price Predictor End
#--------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------
# Loan Approval Predictor
# --------------------------------------------------------------------------------------------------------------------------------------

df_loan = pd.read_csv("static/data/loan_data.csv")

cat_col_loan = df_loan.select_dtypes(include=["object"]).columns.tolist()

encoders_loan = {}

for col in cat_col_loan:
    le = LabelEncoder()
    df_loan[col] = le.fit_transform(df_loan[col])
    encoders_loan[col] = le

X_loan = df_loan.drop("loan_status", axis=1)
y_loan = df_loan["loan_status"]

dec_tree = DecisionTreeClassifier(
    criterion="entropy",
    splitter="best",
    max_depth=5,
    random_state=42
)

dec_tree.fit(X_loan, y_loan)

@app.route('/Loan_Approval_Predictor', methods=['GET', 'POST'])
def loan_approval():

    prediction = None

    if request.method == 'POST':

        input_data = pd.DataFrame([{
            "person_age": int(request.form["person_age"]),
            "person_gender": request.form["person_gender"],
            "person_education": request.form["person_education"],
            "person_income": float(request.form["person_income"]),
            "person_emp_exp": int(request.form["person_emp_exp"]),
            "person_home_ownership": request.form["person_home_ownership"],
            "loan_amnt": float(request.form["loan_amnt"]),
            "loan_intent": request.form["loan_intent"],
            "loan_int_rate": float(request.form["loan_int_rate"]),
            "loan_percent_income": float(request.form["loan_percent_income"]),
            "cb_person_cred_hist_length": int(request.form["cb_person_cred_hist_length"]),
            "credit_score": int(request.form["credit_score"]),
            "previous_loan_defaults_on_file": request.form["previous_loan_defaults_on_file"]
        }])

        for col in cat_col_loan:
            input_data[col] = encoders_loan[col].transform(input_data[col])

        pred = dec_tree.predict(input_data)[0]

        if pred == 1:
            prediction = 1
        else:
            prediction = 0

        print(prediction)

    return render_template(
        "Loan Approval Predictor.html",
        approved=prediction
    )

# --------------------------------------------------------------------------------------------------------------------------------------
# Loan Approval Predictor End
# --------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------
# Titanic ML Predictor
# --------------------------------------------------------------------------------------------------------------------------------------

df_titanic = pd.read_csv("static/data/tested.csv").drop(["PassengerId","Ticket","Cabin","Embarked"], axis=1)
df_titanic["Title"] = df_titanic["Name"].str.extract(r",\s*([^\.]+)\.")
df_titanic.drop("Name", axis=1, inplace=True)

cat_col_titanic = [c for c in df_titanic.columns if df_titanic[c].dtype == "str"]

X_titanic = df_titanic.drop('Survived', axis=1)
y_titanic = df_titanic['Survived']

catboost = CatBoostClassifier(
    n_estimators=50,
    max_depth=5,
    cat_features=cat_col_titanic,
    random_state=42
)
catboost.fit(X_titanic, y_titanic)

@app.route('/Titanic_ML_Predictor', methods=['GET','POST'])
def Titanic():
    pred = None
    probability = None

    if request.method == 'POST':
        input_data = pd.DataFrame([{
            "Pclass": int(request.form["pclass"]),
            "Sex": (request.form["sex"]),
            "Age": float(request.form["age"]),
            "SibSp": int(request.form["sibsp"]),
            "Parch": int(request.form["parch"]),
            "Fare": float(request.form["fare"]),
            "Title": (request.form["title"])
        }])

        pred = int(catboost.predict(input_data)[0])
        probability = round(catboost.predict_proba(input_data)[0][1] * 100, 2)

    return render_template("Titanic-ML-Predictor.html", survived=pred, probability=probability)

# --------------------------------------------------------------------------------------------------------------------------------------
# Titanic ML Predictor End
# --------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------
# Customer Segmentation End
# --------------------------------------------------------------------------------------------------------------------------------------

df_mall = pd.read_excel("static/data/Mall Customers.xlsx").drop(['CustomerID', 'Gender', 'Education ', 'Marital Status'], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_mall)

kmeans = KMeans(
    n_clusters=4,
    init='k-means++',
    random_state=42
)
df_mall['Cluster'] = kmeans.fit_predict(X_scaled)

fig = px.scatter_3d(
    df_mall,
    x='Age',
    y='Annual Income (k$)',
    z='Spending Score (1-100)',
    color='Cluster',
    title='Customer Segmentation (3D)'
)

@app.route('/Customer_Segmentation', methods=['GET', 'POST'])
def customer_segmentation():

    age = 28
    income = 55
    spend = 50

    fig = px.scatter_3d(
        df_mall,
        x="Age",
        y="Annual Income (k$)",
        z="Spending Score (1-100)",
        color="Cluster",
        title="Customer Segmentation (3D)"
    )

    if request.method == "POST":

        age = int(request.form["age"])
        income = int(request.form["income"])
        spend = int(request.form["spend"])

        input_data = pd.DataFrame([{
            "Age": age,
            "Annual Income (k$)": income,
            "Spending Score (1-100)": spend
        }])

        input_scaled = scaler.transform(input_data)
        predicted_cluster = kmeans.predict(input_scaled)[0]

        fig.add_trace(
            go.Scatter3d(
                x=[age],
                y=[income],
                z=[spend],
                mode="markers",
                marker=dict(
                    color="black",
                    size=10,
                    symbol="diamond"
                ),
                name=f"Customer (Cluster {predicted_cluster})"
            )
        )

    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn"
    )

    return render_template(
        "Customer Segmentation.html",
        age=age,
        income=income,
        spend=spend,
        cluster=predicted_cluster if request.method == "POST" else None,
        plot_html=plot_html
    )

# --------------------------------------------------------------------------------------------------------------------------------------
# Customer Segmentation End
# --------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)