import pandas as pd
from xgboost import XGBClassifier

def cal(data):
  xnew = pd.DataFrame.from_dict(data)
  xnew = xnew.drop(['orario'],1).drop(['partite'],1)
  model = XGBClassifier()
  model.load_model("modelsnai.txt")
  ynew1 = str(model.predict(xnew)[-1])
  ynew2 = str(model.predict_proba(xnew)[-1])
  result = ynew1 + ynew2
  return result