import flask
import pickle
from flask import render_template
from sklearn.ensemble import RandomForestClassifier

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')
    if flask.request.method == 'POST':
        with open('rf_width.pkl', 'rb') as model_width:
            load_model_width = pickle.load(model_width)
        with open('rf_depth.pkl', 'rb') as model_depth:
            load_model_depth = pickle.load(model_depth)
        IW = float(flask.request.form['IW'])
        IF = float(flask.request.form['IF'])
        VW = float(flask.request.form['VW'])
        FP = float(flask.request.form['FP'])
        y_pred_width = load_model_width.predict([[IW, IF, VW, FP]])
        y_pred_depth = load_model_depth.predict([[IW, IF, VW, FP]])

        return render_template('main.html', result=(round(y_pred_width[0], 2), round(y_pred_depth[0], 2)))


if __name__ == '__main__':
    app.run()