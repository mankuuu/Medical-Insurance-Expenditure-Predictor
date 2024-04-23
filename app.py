from flask import Flask , request, render_template
import pickle , numpy as np , sklearn

app = Flask(__name__)

data = pickle.load(open("models/exp_predictor.pkl", "rb"))

model = data['model']
encoder = data['encoder']
scaler = data['scaler']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST', 'GET'])
def predict():
    age = request.form.get("age")
    if age == "":
        age = float(18)
    else :
        age = float(age)
    bmi = request.form.get("bmi")
    if bmi == "":
        bmi = float(20)
    else :
        bmi = float(bmi)
    children = request.form.get("children")
    if children =="":
        children = float(0)
    else :
        children = float(children)
    numerical_data = np.array([age,bmi,children])
    scaled_num_data = scaler.transform([numerical_data])
    smoker = request.form['smoker']
    if smoker == "":
        smoker = float(0)
    else :
        smoker = float(smoker)
    gender = request.form['gender']
    if gender == "":
        gender = float(0)
    else :
        gender = float(gender)
    region = request.form['region']
    if region == "":
        region = "northeast"
    region = encoder.transform([[region]]).toarray()
    # print(f" Type : {type(scaled_num_data)}")
    # print(f"Type of gender : {type(gender)} and : {gender}")
    # print(f"Type of region : {type(region)}")
    # print(f"Type of smoker : {type(smoker)} and : {smoker}")
    # print(f"Variables : {age, bmi, children,scaled_num_data,smoker,gender,region}")
    cat_data = np.array([smoker,gender])
    cat_data = cat_data.reshape(1,-1)
    # print(f"Cat Data : {cat_data.shape}")
    # print(f"Region : {region.shape}")
    cate_data = np.concatenate([cat_data , region], axis = 1)
    # print(f"Cate_data : {cate_data}")
    # print(f"Scaled_num : {scaled_num_data.shape}")
    # print(f"Cate_data shape : {cat_data.shape}")
    X = np.concatenate([scaled_num_data,cate_data], axis = 1)
    # print(numerical_data)
    # print(cate_data)
    # print(X)
    # print(f"Shape : {X.shape}")
    y = model.predict(X)
    # y = y[0]
    if y < 0 :
        y = -y
    # y = round(y,2)
    res = "Your Annual Expenditure is ₹ "
    return render_template("index.html" , res = res, pred = round(y[0],2) , data = X)



if __name__ == "__main__":
    print("Running expenditure predictor flask server")
    app.run(debug = True)

# from flask import Flask, request, render_template
# import pickle
# import numpy as np

# app = Flask(__name__)

# data = pickle.load(open("models/exp_predictor.pkl", "rb"))

# model = data['model']
# encoder = data['encoder']
# scaler = data['scaler']

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=['POST', 'GET'])
# def predict():
#     age = float(request.form.get("age"))
#     bmi = float(request.form.get("bmi"))
#     children = int(request.form.get("children"))
#     numerical_data = np.array([age, bmi, children])
#     scaled_num_data = scaler.transform(numerical_data.reshape(1, -1))
#     smoker = request.form['smoker']
#     gender = request.form['gender']
#     region = request.form['region']
#     region_encoded = encoder.transform([[region]])
#     cat_data = np.array([smoker, gender])
#     input_data = np.concatenate((scaled_num_data, cat_data, region_encoded), axis=1)
#     y = model.predict(input_data)
#     return render_template("index.html", pred=y, data=input_data)

# if __name__ == "__main__":
#     print("Running expenditure predictor flask server")
#     app.run(debug=True)
