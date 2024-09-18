import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


class sleeppredictions:
    X_train = []
    X_valid = []
    y_train = []
    y_valid = []
    categorical_columns = []
    categorical_filtered_columns = []
    numerical_columns = []

    def __init__(self, data):
        self.data = data
        

    def dataset_description(self):
        """DATASET VALUES DESCRIPTION"""
        print(data.describe())
        print(data.dtypes)
        print(data.columns)

    def identify_null_values(self):
        """FUNCTION TO CHECK IF THE DATASET CONTAINS NULL VALUES ON ITS' COLUMNS
        """
        columns_with_nulls = [col for col in data.columns if data[col].isnull().any() > 0]
        if len(columns_with_nulls) == 0:
            print("The dataset has no null value in it's columns")
        else:
            print(columns_with_nulls)
    
    def divide_data_train_valid(self):
        """CREATING FEATURES AND LABELS
           SPLTTING DATASET INTO TRAIN DATA AND TEST DATA(VALID)"""
        features_list = ['User ID','Age','Gender','Bedtime','Wake-up Time','Daily Steps','Calories Burned','Physical Activity Level','Dietary Habits','Sleep Disorders','Medication Usage']
        features = data[features_list]
        labels = data["Sleep Quality"]
        print("Features",len(features))
        print("Labels\n",len(labels))
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(features,labels,train_size=0.5, test_size=0.5, random_state=0)
        
        
    
    def check_categorical_numerical_data(self):
        """IDENTIFYING CATEGORICAL AND NUMERICAL DATA
           FILTERING CATEGORICAL DATA BY NOT HAVING < 10 POSSIBLE VALUES
           APPLYING THIS FILTER TO THE TRAIN AND TEST DATA"""
        self.categorical_columns = [col for col in self.X_train.columns if self.X_train[col].dtype == 'object']
        self.categorical_filtered_columns = [col for col in self.X_train.columns if self.X_train[col].dtype == 'object' and self.X_train[col].nunique() < 10]
        self.numerical_columns = [col for col in self.X_train.columns if self.X_train[col].dtype in ['int64', 'float64']]
        
        print("List of every categorical data in the dataset\n", self.categorical_columns)
        print("List of every categorical data in the dataset without a large number of possibilities\n", self.categorical_filtered_columns)
        print("List of every numerical data in the dataset\n", self.numerical_columns)

        final_columns = self.categorical_filtered_columns + self.numerical_columns
        self.X_train = self.X_train[final_columns].copy()
        self.X_valid = self.X_train[final_columns].copy()

    def data_preprocessing(self):
        """CREATING PIPELINES FOR PROCESSING NUMERICAL AND CATEGORICAL DATA
           DESIGNED A COLUMNTRANSFORMER FOR NUMERICAL AND CATEGORICAL DATA 
           FINALLY APPLYING A RANDOM FOREST REGRESSOR TRAINING AND DOING PREDICTIONS"""
           
        numerical_trasnformer = SimpleImputer(strategy = 'constant')
        categorical_transformer = Pipeline(steps = [
            ('imputer', SimpleImputer(strategy = 'most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers = [
                ('num', numerical_trasnformer, self.numerical_columns),
                ('cat', categorical_transformer, self.categorical_filtered_columns)
            ])
        
        model = RandomForestRegressor(n_estimators = 100, random_state = 0)

        clf = Pipeline(steps = [
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        print("Tamaño x train", len(self.X_train))
        print("Tamaño y_valid", len(self.y_valid))
        print("Tamaño y_train", len(self.y_train))
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_valid)
        print(self.X_valid)
        print(preds)
        #mae = mean_absolute_error(preds ,self.y_valid)
        #print(mae)

    def check_mutual_information(self,data):
        """FUNCTION TO CHECK MUTUAL INFORMATION IN THE DATASET SO WE CAN DO SOME GUESSINGS OF OUR DATA"""
        x = data.copy()
        y = x.pop('Sleep Quality')
        print(y)
        
        for colname in x.select_dtypes("object"):
            x[colname], _ = x[colname].factorize()
        
        discrete_features = x.dtypes == int
        mi_scores = mutual_info_regression(x,y,discrete_features = discrete_features)
        mi_scores = pd.Series(mi_scores,name = "MI scores",index=x.columns)
        mi_scores = mi_scores.sort_values(ascending = True)
        width = np.arange(len(mi_scores))
        ticks = list(mi_scores.index)
        
        plt.barh(width, mi_scores)
        plt.yticks(width,ticks)
        plt.title("Mutual Information Scores")
        sns.relplot(x="Calories Burned", y ="Sleep Quality", data = data)
        plt.show()

        
        plt.show()

    def gridsearch_applying(self,data):
        """CREATED A GRIDSEARCH FUNCTION SO WE CAN FIND WHAT MODELS GIVE US A BETTER ACCURACY
           ON NEXT FUNCTION"""
        models = {
            'linear regression' : {
                'model' : LinearRegression(),
                'parameters': {

                }
            },

            'lasso' : {
                'model': Lasso(),
                'parameters': {
                    'alpha': [1,2],
                    'selection' : ['random', 'cyclic']
                }
            },

            'svr' : {
                'model': SVR(),
                'parameters': {
                    'gamma': ['auto', 'scale']
                }
            },
            'decision_tree': {
                'model': DecisionTreeRegressor(),
                'parameters': {
                    'criterion' : ['friedman_mse'],
                    'splitter': ['best', 'random']
                }
            },

            'random_forest' : {
                'model': RandomForestRegressor(criterion = 'squared_error'),
                'parameters': {
                    'n_estimators' : [5,10,15,20]
                }
            },
            'knn': {
                'model': KNeighborsRegressor(algorithm = 'auto'),
                'parameters': {
                    'n_neighbors': [2,5,10,20]
                }
            }
        }
        return models
    def grid_preprocessor(self,data,models):
        """GETTING EVERY MODEL ACCURACY WITH A GRIDSEARCH 
            FINALLY RETURNING A DATAFRAME WITH SCORES, AND BEST PARAMETERS FOR EACH MODEL
            CREATED THE KFOLD IN ORDER TO CHECK IF WOULD GIVE US A BETTER ACCURACY BUT WASN'T THE CASE"""
        kFold = StratifiedKFold(n_splits = 5)
        categorical_columns = [col for col in data.columns if data[col].dtypes == 'object']
        numerical_columns = [col for col in data.columns if data[col].dtypes in ['float64', 'int64']]
        
        data = pd.get_dummies(data, columns = categorical_columns)
        """
        for column in categorical_columns:
            data[column] = LabelEncoder().fit_transform(data[column])
        """
        features = data.drop('Sleep Quality', axis = 'columns')
        labels = data['Sleep Quality']
        scores = []
        
        for model_name,model_params in models.items():
            gs = GridSearchCV(model_params['model'], model_params['parameters'], cv = 10, return_train_score=False)
            gs.fit(features,labels)
            scores.append({
                'model' : model_name,
                'best_parameters': gs.best_params_,
                'score': gs.best_score_
            })
            
            print("Test made for ", model_params['model'])
            X_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
            predictions = gs.predict(x_test)
            
            print(predictions)
            print("Y Values to train",y_train)
            print("Y Values to test",y_test)
        
        return pd.DataFrame(scores, columns=['model', 'best_parameters', 'score'])
    
        
        
    


        

data = pd.read_csv('data/Health_Sleep_Statistics.csv')
sleep1 = sleeppredictions(data)

#sleep1.dataset_description()
#sleep1.show_specific_column('Age')
#sleep1.identify_null_values()
#leep1.divide_data_train_valid()
#sleep1.check_categorical_numerical_data()
#sleep1.data_preprocessing()
#sleep1.check_mutual_information(data)
#sleep1.data_treatmente_pregrid(data)
dict = sleep1.gridsearch_applying(data)
dataframe = sleep1.grid_preprocessor(data, dict)
print(dataframe)