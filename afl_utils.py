import numpy as np
import pandas as pd 

def get_train_data(val_years  = [2015,2016,2017,2018] ):
	# Read data and turn into a numpy array
	data = pd.read_csv('afl_data_train.csv').fillna(0).drop("url",axis = 1)
	val = model_validation(data,val_years)
		
	# Get the data frame
	data_for_predict = data.drop(['season','round','percentage'],axis = 1).values

	return data_for_predict,val, data



# generate a elo classifier
class elo_classifier():

	def __init__(self):
		pass

	def fit(self,X,Y):
		pass

	def predict(self,X):
		return map(int,X[:,-1]>0)



class model_validation:

	def __init__(self, df,seasons_to_split):

		# This is where we define the indexes
		self.df = df
		self.seasons = seasons_to_split

		self.idx_train = []
		self.idx_test = []

		for year in seasons_to_split:
			rounds = [int(u) for u in df.loc[df['season']==year,'round'].unique() ]
			min_round = min(rounds)
			max_round = max(rounds)
			for r in list(range(min_round,max_round+1)):
				test = df.index[(df['season']==year) & (df['round']==r)].tolist()
				train = list(range(0,test[0]))

				self.idx_train.append(train)
				self.idx_test.append(test)




	def validate(self,model_class,X,Y):
		
		pred = []
		true = []
		for y in zip(self.idx_test,self.idx_train):
			#clone_clf = clone(model_class)

			model_class.fit(X[y[1]],Y[y[1]])
			pred.extend(model_class.predict(X[y[0]]))
			true.extend(Y[y[0]])

		acc = np.sum(np.array(pred) == np.array(true))

		return acc


	def get_iterable(self):

		cv = [(self.idx_train[i], self.idx_test[i]) for i in range(len(self.idx_test))]
		return cv







