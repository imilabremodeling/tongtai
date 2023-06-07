from flask import Flask, request , jsonify, render_template
#cors跨來源的資源共享（可以跨網域）
from flask_cors import CORS
import csv
import pandas as pd

app = Flask(__name__)
CORS(app)
print(pd)


index1 = 1
index2 = 0
"""
#給mac端
@app.route('/test1') 
def home(): 
    global index1
    index = 0
  
    with open('train.csv', newline='') as csvfile:

    # 讀取 CSV 檔案內容
        myData=list(csv.reader(csvfile))
	# to read the csv file using the pandas library 
    #data = pd.read_csv(filename, header=0) 
    #myData = data.values
        
        stringA= str(myData[index1][index+1]) 
        stringB= str(myData[index1][index+2]) 
        stringC= str(myData[index1][index+3])
        stringD= str(myData[index1][index+4])
        stringE= str(myData[index1][index+5]) 
        stringF= str(myData[index1][index+6])  
        stringG= str(myData[index1][index+7])    
        stringH= str(myData[index1][index+8]) 
        stringI= str(myData[index1][index+9]) 
        stringJ= str(myData[index1][index+10])          
        print(len(myData))
        print(len(myData[0]))
        if index1==1301:

            index1=1
        
        else:

             index1 +=1
        
       
        return  '{} {} {} {} {} {} {} {} {} {}'.format(stringA,stringB,stringC,stringD,stringE,stringF,stringG,stringH,stringI,stringJ)

"""
#給linux

@app.route('/test2') 
def home(): 
    global index2
    index = 0
    filename = 'train.csv' 
    data = pd.read_csv(filename, header=0) 
    myData = data.values    
    stringA1= str(myData[index2][index]) 
    stringB2= str(myData[index2][index+1]) 
    stringC3= str(myData[index2][index+2])
    stringD4= str(myData[index2][index+3])
    stringE5= str(myData[index2][index+4]) 
    stringF6= str(myData[index2][index+5])  
    stringG7= str(myData[index2][index+6])    
    stringH8= str(myData[index2][index+7]) 
    stringI9= str(myData[index2][index+8]) 
    stringJ10= str(myData[index2][index+9])          

    if index2==1300:

        index2=0
        
    else:

        index2 +=1
   
    return  '{} {} {} {} {} {} {} {} {} {}'.format(stringA1,stringB2,stringC3,stringD4,stringE5,stringF6,stringG7,stringH8,stringI9,stringJ10)


if __name__ == '__main__':
    from waitress import serve	    
    #serve(app, host="0.0.0.0", port=3888)
    app.run(host='0.0.0.0', port=3888, debug=True)
