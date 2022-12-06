import numpy as np
import pandas as pd

file ="heart_tratado.csv"

df = pd.read_csv(file,sep =';', encoding="utf-8")

print(df.dtypes)

df2 = pd.DataFrame.copy(df)

print(df2.columns)
from sklearn.preprocessing import StandardScaler, LabelEncoder


"""
Age                 int64
Sex                object
ChestPainType      object
RestingBP           int64
Cholesterol       float64
FastingBS           int64
RestingECG         object
MaxHR               int64
ExerciseAngina     object
Oldpeak           float64
ST_Slope           object
HeartDisease        int64
"""
previsores =df2.iloc[:, 0:11].values
classe_alvo  =df2.iloc[:,11].values

#Aplica Label Enconder()
previsores[:,1] = LabelEncoder().fit_transform(previsores[:,1])
previsores[:,2] = LabelEncoder().fit_transform(previsores[:,2])
previsores[:,6] = LabelEncoder().fit_transform(previsores[:,6])
previsores[:,8] = LabelEncoder().fit_transform(previsores[:,8])
previsores[:,10] = LabelEncoder().fit_transform(previsores[:,10])



from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

previsores3 = ColumnTransformer(transformers=[('Onehot', OneHotEncoder(), [1,2,6,8,10])], remainder='passthrough').fit_transform(previsores) 

previsores_scaler = StandardScaler().fit_transform(previsores3)

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste =train_test_split(previsores_scaler,classe_alvo, test_size =0.3, random_state=0)

print(f'x_treino: {x_treino.shape}')
print(f'y_treino: {y_treino.shape}')
print(f'x_teste: {x_teste.shape}')
print(f'y_teste: {y_teste.shape}')

from predicao_rede_neural import predicao_rede_neural 
predicao_rede_neural(x_treino, x_teste, y_treino, y_teste)


from validacao_cruzada_rede_neural import validacao_cruzada_rede_neural
validacao_cruzada_rede_neural(previsores_scaler,classe_alvo)
