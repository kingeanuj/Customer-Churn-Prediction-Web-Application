# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

        

class XGBoostModel():

       
    def preprocessing(self,file):
        # Importing the dataset
        dataset = pd.read_csv(file)
        
        
        dataset = dataset.iloc[:,3:]
        dataset= pd.get_dummies(data = dataset, columns=["Geography","Gender"])
        y = dataset["Exited"].values
        dataset = dataset.drop(columns=["Geography_Spain", "Gender_Female", "Exited"])
        dataset = pd.DataFrame(data= dataset, columns= ["CreditScore",
                                                        "Age",
                                                        "Tenure",
                                                        "Balance",
                                                        "NumOfProducts", 
                                                        "EstimatedSalary",
                                                        "HasCrCard",
                                                       "IsActiveMember",
                                                       "Geography_France",
                                                       "Geography_Germany",
                                                       "Gender_Male"])
        X = dataset.iloc[:,:].values
        return X, y
    
    
    def train_model(self, X_train,y_train):
        # Training XGBoost Model
        from xgboost import XGBClassifier
        classifier = XGBClassifier()
        classifier.fit(X_train, y_train)                              
        return classifier
    
    def saving_model(self, file):
        # Saving model to disk
        pickle.dump(classifier, open(file,'wb'))
        
    def load_model(self, file):
        # Loading model to compare the results
        return pickle.load(open(file,'rb'))
    

if __name__=="__main__":
    
    instance = XGBoostModel()
    
    X, y= instance.preprocessing(file="Training batch file.csv")
    
    
    classifier = instance.train_model(X, y)
    
    instance.saving_model(file="XGBoostCLassifier.pkl")
    
    model = instance.load_model(file = 'XGBoostCLassifier.pkl')
    
     
    # standardized = np.array(sc.transform([[597.00, 35.00, 8.00, 131101.04, 1.00, 192852.67]]))
    # print(standardized[0])
    
    # non_standardized = np.array ([1.00, 1.00, 0.00, 1.00, 1.00])
    # print(non_standardized)
    # to_be_predicted = np.concatenate((standardized[0],non_standardized)).reshape(1,11)
    
    # pred = model.predict(to_be_predicted)
    # print(pred)