from flask import Flask, render_template, request
import pandas as pd
import pickle
from datetime import datetime

model = pickle.load(open('energy.pkl', 'rb'))
df = pd.read_pickle('df.pkl')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict')
def predict_ui():
    return render_template('predict.html', **locals())


@app.route('/predict_energy', methods=['POST', 'GET'])
def predict():
    end = request.form["Pred_month"]
    end_year = int(pd.to_datetime(end, format="%Y-%m").year)
    end_month = int(pd.to_datetime(end, format="%Y-%m").month)

    if end_month == 1:
        month_name = 'January'

    elif end_month == 2:
        month_name = 'February'

    elif end_month == 3:
        month_name = 'March'

    elif end_month == 4:
        month_name = 'April'

    elif end_month == 5:
        month_name = 'May'

    elif end_month == 6:
        month_name = 'June'

    elif end_month == 7:
        month_name = 'July'

    elif end_month == 8:
        month_name = 'August'

    elif end_month == 9:
        month_name = 'September'

    elif end_month == 10:
        month_name = 'October'

    elif end_month == 11:
        month_name = 'November'

    else:
        month_name = 'December'

    forecasted = model.predict(start=datetime(2023, 5, 1), end=datetime(end_year, end_month, 1))

    for i in range(len(forecasted)):
        if forecasted.index[i] == datetime(end_year, end_month, 1):
            output = round(forecasted.values[i], 2)

    return render_template('predict.html', **locals())


if __name__ == '__main__':
    app.run(debug=True, port=8000)
