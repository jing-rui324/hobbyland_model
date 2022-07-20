from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    IntLvl = int(request.form['IntLvl'])
    ComLvl = int(request.form['ComLvl'])
    Age = int(request.form['Age'])
    prediction = model.predict([[IntLvl, ComLvl, Age]])
    output = round(prediction[0], 0)
    return render_template('index.html', prediction_text=f'Your Hobbyscore is {output}')


if __name__ == "__main__":
    app.run()
