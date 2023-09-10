import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn as sl
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import multiprocessing


pd.options.display.max_columns = 2
pd.options.display.max_rows = 10000000

data = pd.read_csv('csv files/nobel-prize-laureates.csv',sep=';')

data.drop(['Overall motivation','Born country code','Died country code','Organization city','Geo Shape','Geo Point 2D','Born city','Died city','Motivation','Organization name','Organization country','Firstname','Surname','Born country','Died country'],axis=1,inplace=True)
data['Died_sep'] = data['Died'].str.split('-')
data['Died_sep'].explode()
data['Born_sep'] = data['Born'].str.split('-')
data['Born_sep'].explode()
i = 0
def sep(var):
    if type(var) == list:
        return var[i]
    else:return var
def Int(var):
    return int(var)
def gender(var):
    if var == 'male':
        return 0
    elif var == 'female':
        return 1

def category(var):
    if var == 'Chemistry':
        return 0
    elif var == 'Physics':
        return 1
    elif var == 'Medicine':
        return 2
    elif var == 'Economics':
        return 3
    elif var == 'Literature':
        return 4
    elif var == 'Peace':
        return 5
data['Died_year'] = data['Died_sep'].apply(sep)
data['Born_year'] = data['Born_sep'].apply(sep)
i+=1
data['Died_month'] = data['Died_sep'].apply(sep)
data['Born_month'] = data['Born_sep'].apply(sep)
i+=1
data['Died_day'] = data['Died_sep'].apply(sep)
data['Born_day'] = data['Born_sep'].apply(sep)
data.drop(['Died_sep','Died','Born','Born_sep'],axis=1,inplace=True)
data.dropna(axis=0,how='any',inplace=True)
data.reset_index(inplace=True)
data['Gender'] = data['Gender'].apply(gender)
data['Category'] = data['Category'].apply(category)
data['Died_year'] = data['Died_year'].apply(Int)
data['Died_month'] = data['Died_month'].apply(Int)
data['Died_day'] = data['Died_day'].apply(Int)
data['Born_year'] = data['Born_year'].apply(Int)
data['Born_month'] = data['Born_month'].apply(Int)
data['Born_day'] = data['Born_day'].apply(Int)

x = data['Category'].values.reshape(-1, 1)
y = data['Gender'].values.reshape(-1, 1)
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.25)

lr = KNeighborsClassifier()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)


"""sns.displot(data,x='Gender',y='Category')
plt.plot(x_test,y_pred)
plt.show()"""

y_pred_round = np.round(y_pred)
print('regular accuracy:',metrics.accuracy_score(y_true=y_test,y_pred=y_pred_round)*100,'%')
print('balanced accuracy:',metrics.balanced_accuracy_score(y_true=y_test,y_pred=y_pred_round)*100,'%')
print('precision:',metrics.precision_score(y_true=y_test,y_pred=y_pred_round,average='weighted',zero_division=1)*100,'%')
print('F1:',metrics.f1_score(y_true=y_test,y_pred=y_pred_round,average='weighted')*100,'%')
print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
matrix = metrics.confusion_matrix(y_true=y_test,y_pred=y_pred_round)

display = metrics.ConfusionMatrixDisplay(matrix,display_labels=['Chemistry','Physics','Medicine','Economics','Literature','Peace'])
display.plot()
plt.show()