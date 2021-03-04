import joblib
from flask import Flask, render_template, request, url_for
app = Flask(__name__)

@app.route('/check', methods=['POST'])
def checker():
    password = request.form["pwd"]
    colordict = {0: "cardred", 1: "cardyellow", 2:"cardgreen"}
    strength = {}
    
    modelinput = [password]

    LogisticRegressionModel = joblib.load('LogisticRegression.joblib')
    DecisionTreeModel = joblib.load('DecisionTree.joblib')
    NaiveBayesModel = joblib.load('NaiveBayes.joblib')
    RandomForestModel = joblib.load('RandomForestClassifier.joblib')

    LROutput = LogisticRegressionModel.predict(modelinput)
    DTOutput = DecisionTreeModel.predict(modelinput)
    NBOutput = NaiveBayesModel.predict(modelinput)
    RFOutput = RandomForestModel.predict(modelinput)
    
    c0=0
    c1=0
    c2=0

    if(LROutput[0]==0):
        c0=c0+1
    elif(LROutput[0]==1):
        c1=c1+1
    else:
        c2=c2+1
    
    if(DTOutput[0]==0):
        c0=c0+1
    elif(DTOutput[0]==1):
        c1=c1+1
    else:
        c2=c2+1
    
    if(RFOutput[0]==0):
        c0=c0+1
    elif(RFOutput[0]==1):
        c1=c1+1
    else:
        c2=c2+1
    
    if(NBOutput[0]==0):
        c0=c0+1
    elif(NBOutput[0]==1):
        c1=c1+1
    else:
        c2=c2+1

    if(c0>=(c1+c2)):
        res = 0
    elif(c1>=(c0+c2)):
        res = 1
    else:
        res = 2
    
    strength['lr'] = colordict[LROutput[0]]
    strength['dt'] = colordict[DTOutput[0]]
    strength['rf'] = colordict[RFOutput[0]]
    strength['nb'] = colordict[NBOutput[0]]
    strength['res'] = colordict[res]
    return render_template("strength.html", output=strength)

@app.route('/')
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
