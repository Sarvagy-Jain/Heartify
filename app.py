from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
app = Flask(__name__)

model = pickle.load(open("classifier.pkl","rb"))
sc = StandardScaler()
df = pd.read_csv('heart.csv')
X = df.iloc[: , 0:-1 ]
Y = df['output']
X_trainc, X_testc, y_train, y_test = train_test_split(X, Y, test_size = 0.25 ,random_state=8)
sc.fit_transform(X_testc)

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    # for x in request.form.values(): print(x)
    float_features = [float(x) for x in request.form.values()]
    for x in request.form.values(): print(x)
    features =  [np.array(float_features)]
    features = sc.transform(features)
    for x in features: print(x)
    prediction  = model.predict(features)
    if(prediction ==  1): ans="Heart at Risk"
    else:    ans = "Heart is Happy"
    return render_template("index.html",prediction_text="{}".format(ans))

if __name__ == "__main__":
    app.run(debug=True)
