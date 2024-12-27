from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomDataDen,PredictPipelineDen
from src.pipeline.predict_pipeline import CustomDataVis,PredictPipelineVis

app = Flask(__name__)

## Route for a home page

@app.route('/')

def index():
    return render_template("home.html")

@app.route('/predictionDenstiy',methods=["Post","GET"])
def densityPrediction():
    if request.method == "GET":
        return render_template("predictionDensity.html")
    else:
        data = CustomDataDen(
            float(request.form.get("MR")),
            float(request.form.get("tk")),
            float(request.form.get("MCI")),
            float(request.form.get("wm")),
            float(request.form.get("MW"))
        )
        
        dataframe = data.get_data_as_df()
        prediction = PredictPipelineDen()
        result = prediction.pre(dataframe)

        return render_template("predictionDensity.html",result=result[0])
    

@app.route('/predictionViscosity',methods=['POST','GET'])
def viscosityPrediction():
    if request.method == "GET":
        return render_template("predictionViscosity.html")
    else:
        data = CustomDataVis(
            1/float(request.form.get("temp")),
            float(request.form.get("MCI")),
            float(request.form.get("m")),
            float(request.form.get("omega"))
        )
        
        dataframe = data.get_data_as_df()
        prediction = PredictPipelineVis()
        result = prediction.pre(dataframe)

        return render_template("predictionViscosity.html",result=result[0])
    



if __name__ == '__main__':
    app.run()
