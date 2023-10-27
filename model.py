# Importing the libraries
import pandas as pd
import pickle

dataset = pd.read_csv('Powerlifting.csv')
X = dataset.iloc[:, :6]
y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[295, 3, 175, 5, 335, 6]]))