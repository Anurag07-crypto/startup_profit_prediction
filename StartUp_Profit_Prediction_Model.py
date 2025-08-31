import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder

startup = pd.read_csv("50_Startups.csv",encoding="latin1")
sc = StandardScaler()
numeric_cols = startup.drop(columns=["State", "Profit"]).columns
sc.fit(startup[numeric_cols])
data_scaled = pd.DataFrame(sc.transform(startup[numeric_cols]), columns=numeric_cols)
data = pd.concat([data_scaled, startup[["Profit"]]], axis=1)
ohe = OneHotEncoder(sparse_output=False,handle_unknown="ignore")
smp = startup[["State"]]
ohe.fit(smp)
startup_new = pd.DataFrame(ohe.transform(smp),columns=ohe.get_feature_names_out(["State"]))
n_startup =pd.concat((data,startup_new),axis=1)

x = n_startup.drop(columns="Profit")
y = n_startup["Profit"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.23,random_state=42)

def linear(x_train, x_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    lr_pred = lr.predict(x_test)
    return lr, r2_score(y_test, lr_pred) * 100  # Return model and score

def tree(x_train, x_test, y_train, y_test):
    dtc = DecisionTreeRegressor(max_depth=4, min_impurity_decrease=0)
    dtc.fit(x_train, y_train)
    dtc_pred = dtc.predict(x_test)
    return dtc, r2_score(y_test, dtc_pred) * 100  # Return model and score

def support_vector(x_train, x_test, y_train, y_test):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    scores = {}
    models = {}
    for ker in kernels:
        sv = SVR(kernel=ker)
        sv.fit(x_train, y_train)
        sv_pred = sv.predict(x_test)
        scores[ker] = r2_score(y_test, sv_pred)
        models[ker] = sv
    best_kernel = max(scores, key=scores.get)
    return models[best_kernel], scores[best_kernel] * 100  # Return best model and score

def knn(x_train, x_test, y_train, y_test):
    scores = {}
    models = {}
    for i in range(1, 10):
        knr = KNeighborsRegressor(n_neighbors=i)
        knr.fit(x_train, y_train)
        knr_pred = knr.predict(x_test)
        scores[i] = r2_score(y_test, knr_pred)
        models[i] = knr
    best_n = max(scores, key=scores.get)
    return models[best_n], scores[best_n] * 100  # Return best model and score

def run_all(x_train, x_test, y_train, y_test):
    results = {}
    models = {}

    models['Linear Regression'], results['Linear Regression'] = linear(x_train, x_test, y_train, y_test)
    models['Decision Tree'], results['Decision Tree'] = tree(x_train, x_test, y_train, y_test)
    models['Support Vector'], results['Support Vector'] = support_vector(x_train, x_test, y_train, y_test)
    models['KNN'], results['KNN'] = knn(x_train, x_test, y_train, y_test)

    for key, values in results.items():
        print(key, values)

    best_model_name = max(results, key=results.get)
    return models[best_model_name], results

# Run once and get the best model
best_model, scores = run_all(x_train, x_test, y_train, y_test)
print("\nThe Accuracy of all Models:\n", scores)
print("Best model:", type(best_model).__name__)

# Interactive prediction loop
while True:
    try:
        print("\nIf you want to exit, press Ctrl+C or enter non-numeric values")
        r_d_spend = float(input("Enter the R&D Spend: "))
        admin = float(input("Enter the Administration cost: "))
        marketing = float(input("Enter the Marketing cost: "))

        # Get state input and convert to proper format
        state_input = input("Enter State code (new york=1 0 0, california=0 1 0, florida=0 0 1): ")
        state = list(map(int, state_input.split()))
        
        # Create feature array and scale only the numeric features
        numeric_features = [r_d_spend, admin, marketing]
        scaled_features = sc.transform([numeric_features])[0].tolist()
        
        # Combine scaled features with state encoding
        features = [scaled_features + state]
        
        y_pred = best_model.predict(features)
        print("The Predicted Profit Value is->", y_pred[0])

    except (ValueError, KeyboardInterrupt):
        print("Exiting...")
        break