from flask import Flask
import pandas as pd
from sklearn import preprocessing
from flask import Flask, render_template, json, request,jsonify
from sklearn.externals import joblib
import traceback
import datetime
app = Flask(__name__)

@app.route("/")
def main():
    return render_template("dashboard.html" )


@app.route('/result',methods = ['POST', 'GET'])
def result():
  
    _rf = 0
    if request.method == 'POST':
      result = request.form
      print(result)
      for key, value in result.items():
        if(key=='tid'):
            _tid=value
        elif(key=='cid'):
            _cid=value
        elif(key=='site'):
            _site=value
        elif(key=='sid'):
            _sid=value
        elif(key=='shid'):
            _shid=value
        elif(key=='ItemClassification'):
            _ItemClassification=value   
        elif(key=='gs'):
            _gs=value 
        elif(key=='CapturedDate'):
            _CapturedDate=value 

    ##Create day and month
    print(_CapturedDate)
    w=_CapturedDate
    a = int(w[0]+w[1]+w[2]+w[3]) 
    b = int(w[5]+w[6])
    c= int(w[8]+w[9])
    x = datetime.datetime(a,b,c)
    _day=(x.strftime("%A"))
    _month=(x.strftime("%B")) 

    d = {'CapturedMonth':[_month],'CapturedDay': [_day], ' ShiftId': [_sid], ' ShiftHourIdF':[_shid], 'ItemClassificationId':[_ItemClassification], ' GarmentStatusId':[_gs] }
    #d = {'CapturedMonth':[1],'CapturedDay': [5], ' ShiftId': [16], ' ShiftHourId':[5], ' ItemClassificationId':[2], ' GarmentStatusId':[1] }
    df = pd.DataFrame(data=d)
    #df = df.reindex(columns=model_columns, fill_value=0)
    #print('df',df)
    print("*******************************")

    if gb:
        try:
            
            q = pd.get_dummies(df)
            #print(q)
            q = q.reindex(columns=model_columns, fill_value=0)
            #q=q.fillna(0.0, inplace=True)
            print('df',q)

           
            prediction = list(gb.predict(q))
            print(prediction)

            #return jsonify({'prediction': str(prediction)})

        except:
            print("#####")
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

    
    if request.method == 'POST' :
        return render_template("dashboard.html", data='Predicted BandID is : '+ str(prediction) )
        
   
    print("******")
    return render_template("dashboard.html", value=prediction)
    #result.add('a','a')
    #result.extend(a=a)
    
    
   
    
    



if __name__ == "__main__":
    gb = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    print(gb)
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')
    print(model_columns)

    app.run()
    