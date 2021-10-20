
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import ipywidgets as widgets
widgets.IntSlider()
from IPython.display import display
w = widgets.IntSlider()
uploader = widgets.FileUpload(
    accept='*.csv',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
    multiple=False  # True to accept multiple files upload else False
)

display(uploader)



import io
input_file = list(uploader.value.values())[0]
content = input_file['content']
content = io.StringIO(content.decode('utf-8'))
df = pd.read_csv(content)
df.shape


df.head()


df.describe()


df.plot(x='Double', y='Integer', style='o')
plt.title('Double vs Integer')
plt.xlabel('Double values')
plt.ylabel('Percentage Score')
plt.show()


X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)



print(regressor.intercept_)


print(regressor.coef_)



y_pred = regressor.predict(X_test)
dataset = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
dataset



from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))



print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))



print('Root Mean Squared Error:',    np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


