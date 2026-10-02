import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

## 1. Load the BankLoan training dataset.

# ================================
# LOAD DATA
# ================================
data = pd.read_csv('BANK LOAN.csv')

print('datatypes of dataset:\n',data.dtypes,  "\n")
print('Shape of dataset:', data.shape,  "\n")
print( "\n", data.head(), "\n")
print( "\n", data.tail(), "\n")
print(data.info())
print(data.describe(), "\n")
print("\n", data['DEFAULTER'].value_counts(normalize=True) * 100)


## 2. Use Defaulter as the dependent (target) variable.
## 3. Use all variables except SN as independent variables.
Y = data['DEFAULTER']
X = data.drop(columns=['SN', 'DEFAULTER'])

## 4. Develop a Random Forest classification model using 500 trees.
# 70% training, 30% test; stratified to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, stratify=Y
)

X_train.shape, X_test.shape

rf_model = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X, Y)

print('OOB Score (baseline RF):', round(rf_model.oob_score_, 3))

## 5. Load the BankLoan_Test dataset and estimate the probability of default.

# ================================
# 5. LOAD TEST DATA & PREDICT
# ================================
test_data = pd.read_csv('BANK LOAN_TEST.csv')      # use the exact file name you were given
print('Shape of test dataset:', test_data.shape)
print(test_data.columns.tolist())                  # check the columns match the training file
test_data.head()

# ================================
# PREDICTIONS (THRESHOLD 0.26)
# ================================
THRESHOLD = 0.26      

X_test = test_data.drop(columns=['SN', 'DEFAULTER'])
X_test = X_test[X.columns]                        
print('Shape of test dataset:', X_test.shape)

#test_data['Probability'] = rf_model.predict(X_test)
test_data['Probability'] = rf_model.predict_proba(X_test)[:, 1]          
test_data['Predicted_Class'] = (test_data['Probability'] > THRESHOLD).astype(int)
         

prediction_table = test_data[['SN', 'DEFAULTER', 'Probability', 'Predicted_Class']].copy()
prediction_table['Probability'] = prediction_table['Probability'].round(3)
prediction_table.head(10)



## 7. Create a FastAPI endpoint (/predict) to return the predicted probability of default.

def generate_prediction():
    print("called predict()")
    df = prediction_table 
    summary = df.round(4).to_dict(orient="records") 
    print(f"returning from predict() {summary}")
    return summary 

df = generate_prediction()
df


##############################################################
# The Fast-PAI applicaiton
###############################################################


# -------------------------------------------------
# Create FastAPI app
# -------------------------------------------------
app = FastAPI()



# -------------------------------------------------
# API endpoint returning JSON data
# -------------------------------------------------
@app.get("/predict")
def predict():
    df  = generate_prediction()
    return df


# -------------------------------------------------
# HTML frontend embedded directly in endpoint
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bank Loan Defaulter Summary</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
            }
            button {
                padding: 10px 16px;
                font-size: 16px;
                margin-bottom: 20px;
                cursor: pointer;
            }
            table {
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid #ccc;
                padding: 8px;
                text-align: center;
            }
            th {
                background-color: #f4f4f4;
            }
        </style>
    </head>
    <body>

        <h2>Bank Loan Defaulter Summary</h2>

        <button onclick="loadData()">Load Summary</button>

        <table id="summaryTable">
            <thead>
                <tr>
                    <th>SN</th>
                    <th>DEFAULTER</th>                    
                    <th>Probability</th>
                    <th>Predicted_Class</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>

        <script>
            function loadData() {
                fetch('/predict')
                    .then(response => response.json())
                    .then(data => {
                        const tbody = document.querySelector('#summaryTable tbody');
                        tbody.innerHTML = '';

                        data.forEach(row => {
                            const tr = document.createElement('tr');
                            tr.innerHTML = `
                                <td>${row.SN}</td>
                                <td>${row.DEFAULTER}</td>
                                <td>${row.Probability}</td>
                                <td>${row.Predicted_Class}</td>
                            `;
                            tbody.appendChild(tr);
                        });
                    })
                    .catch(error => {
                        alert('Error fetching data');
                        console.error(error);
                    });
            }
        </script>

    </body>
    </html>
    """

    return html_content