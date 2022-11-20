# Import Libraries
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import LabelEncoder

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
# API_KEY = "Qo9j8ni7qMJ8j1C8VFDRFHbuGRAhYWcTlkVqnYg1AGkE"
# token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
# mltoken = token_response.json()["access_token"]
# header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}


app = Flask(__name__)#initiate flask app

def load_model(file='../Result/resale_model.sav'):#load the saved model
	return pickle.load(open(file, 'rb'))

@app.route('/')
def index():#main page
	return render_template('car.html')

@app.route('/predict_page')
def predict_page():#predicting page
	return render_template('value.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
	reg_year = int(request.args.get('regyear'))
	powerps = float(request.args.get('powerps'))
	kms= float(request.args.get('kms'))
	reg_month = int(request.args.get('regmonth'))

	gearbox = request.args.get('geartype')
	damage = request.args.get('damage')
	model = request.args.get('model')
	brand = request.args.get('brand')
	fuel_type = request.args.get('fuelType')
	veh_type = request.args.get('vehicletype')
	data = [reg_year,powerps,kms,reg_month,gearbox,damage,model,brand,fuel_type,veh_type]
	ml_model = pickle.load(open('model.pkl', 'rb'))

	new_row = {'yearOfReg':reg_year, 'powerPS':powerps, 'kilometer':kms,
				'monthOfRegistration':reg_month, 'gearbox':gearbox,
				'notRepairedDamage':damage,
				'model':model, 'brand':brand, 'fuelType':fuel_type,
				'vehicletype':veh_type}

	print(new_row)

	new_df = pd.DataFrame(columns=['vehicletype','yearOfReg','gearbox',
		'powerPS','model','kilometer','monthOfRegistration','fuelType',
		'brand','notRepairedDamage'])
	new_df = new_df.append(new_row, ignore_index=True)
	labels = ['gearbox','notRepairedDamage','model','brand','fuelType','vehicletype']
	mapper = {}

	for i in labels:
		mapper[i] = LabelEncoder()
		mapper[i].classes = np.load('../Result/'+str('classes'+i+'.npy'), allow_pickle=True)
		transform = mapper[i].fit_transform(new_df[i])
		new_df.loc[:,i+'_labels'] = pd.Series(transform, index=new_df.index)

	labeled = new_df[['yearOfReg','powerPS','kilometer','monthOfRegistration'] + [x+'_labels' for x in labels]]

	X = labeled.values.tolist()
	predict = ml_model.predict(X)
	return render_template('predict.html',predict=predict[0])
