import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

data_set = pd.read_csv("insurance.csv")
print(data_set.to_string())

print(data_set.describe())
print(data_set.isnull().sum())

age_statistic = data_set["age"].value_counts()
sex_statistic = data_set["sex"].value_counts()
bmi_statistic = data_set["bmi"].value_counts()
children_statistic = data_set["children"].value_counts()
smoker_statistic = data_set["smoker"].value_counts()
region_statistic = data_set["region"].value_counts()

print("yaş kolonundaki değerlerin kaç defa tekrarladıkları:\n", age_statistic)
print("\n\n")
print("medeni durum değerlerinin kaç defa tekrarladıkları:\n", sex_statistic)
print("\n\n")
print("vücut kitle indeksi durumlarının kaç defa tekrarladıkları:\n", bmi_statistic)
print("çocuk sayısı verilerinin kaç defa tekrarladıkları:\n", children_statistic)
print("\n\n")
print("sigara içip içmediklerine ilişkin değerlerinin kaç defa tekrarladıkları:\n", smoker_statistic)
print("bölge özelliğine ilişkin verilerin kaç defa tekrarladıkları:\n", region_statistic)

sex_types = pd.get_dummies(data_set.sex, prefix='sex')
smoker_types = pd.get_dummies(data_set.smoker, prefix='smoker')
region_types = pd.get_dummies(data_set.region, prefix='region')
data_set = pd.concat([data_set, sex_types, smoker_types, region_types], axis=1)
print(data_set.to_string())

data_set.drop(['sex', 'smoker', 'region', 'sex_female', 'smoker_no'], axis=1, inplace=True)
print(data_set.to_string())

y = data_set["charges"]
data_set.drop(["charges"], axis=1, inplace=True)
x = data_set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=46)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 4.1 - KARAR AĞACI REGRESYON ALGORİTMASI
tree_regression = DecisionTreeRegressor(random_state=42, max_depth=2)
tree_regression = tree_regression.fit(x_train, y_train)
tahmin_tree_regression = tree_regression.predict(x_test)

# 4.2 - RANDOM FOREST REGRESYON ALGORİTMASI
random_regression = RandomForestRegressor(max_depth=4, random_state=42)
random_regression.fit(x_train, y_train)
tahmin_random_regression = random_regression.predict(x_test)

# 4.3 - LASSO REGRESYON ALGORİTMASI
lassoReg = Lasso(alpha=2)
lassoReg.fit(x_train, y_train)
tahmin_lasso = lassoReg.predict(x_test)

# 4.4 - ELASTICNET REGRESYON ALGORİTMASI
elastic_reg = ElasticNet(random_state=0)
elastic_reg.fit(x_train, y_train)
tahmin_elastic = elastic_reg.predict(x_test)

# 4.5 - RIDGE REGRESYON ALGORİTMASI
ridge_reg = Ridge()
ridge_reg.fit(x_train, y_train)
tahmin_ridge = ridge_reg.predict(x_test)

# SONUÇLARI HESAPLAMA FONKSİYONU
def performance_calculate(predict):
    mae = mean_absolute_error(y_test, predict)
    mse = mean_squared_error(y_test, predict)
    rmse = np.sqrt(mse)  # or mse**(0.5)
    r2 = r2_score(y_test, predict)
    data = [mae, mse, rmse, r2]
    return data

predicts = [tahmin_tree_regression, tahmin_random_regression, tahmin_lasso, tahmin_elastic, tahmin_ridge]
algoritma_names = ["Karar Ağacı Regresyon", "Random Forest Regresyon", "Lasso Regresyon", "ElasticNet Regresyon", "Ridge Regresyon"]

# EKRANA YAZDIRMAK
seriler = []
metrics = ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "R2"]

for i in predicts:
    data = performance_calculate(i)
    seriler.append(data)

from IPython.display import HTML
df = pd.DataFrame(data=seriler, index=algoritma_names, columns=metrics)
pd.set_option('display.colheader_justify', 'center')  # kolon isimlerini ortaliyoruz
print(df.to_string())
