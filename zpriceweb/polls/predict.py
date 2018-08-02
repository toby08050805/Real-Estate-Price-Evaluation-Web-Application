 #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#most import code here       
import pandas
import xgboost as xgb
data = pd.read_csv('example_input.csv') #read the input file
data.info()
data.columns
bst = xgb.Booster()                #initial instance
bst.load_model('xgb_model')        #load the xgboost model
xgb_data = xgb.DMatrix(data)       #transfer input data to xgb matrix
result = bst.predict(xgb_data)     #get the result