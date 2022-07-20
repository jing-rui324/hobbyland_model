from sklearn import linear_model
import pandas as pd
import pickle
df = pd.read_csv('Hobbyscore.csv')

y = df['HobScore']
X = df[['IntLvl', 'ComLvl', 'Age']]

lm = linear_model.LinearRegression()
lm.fit(X.values, y.values)
pickle.dump(lm, open('model.pkl', 'wb'))

print(lm.predict([[9, 9, 18]]))
print(f'score:{lm.score(X,y)}')
