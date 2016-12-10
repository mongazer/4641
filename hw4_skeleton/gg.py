

class Tfidf():
	def __init__(self):
		self.sublinear_tf= True




	def fit_transform(self,X):
		
		d, t = X.shape


		tf = np.copy(X)
		tf[tf!=0] = np.log(tf[tf!=0])+1	
		

		idf = X.sum(axis=0)
		idf[idf<d-1] = np.log(d/(idf[idf<d-1]+1))
		self.idf = idf

		tfidf = tf*idf

		normalizee(tfidf)

		return sparse.csr_matrix(tfidf)


	def transform(self, X):
		d, t = X.shape
		tf = np.copy(X)
		tf[tf!=0] = np.log(tf[tf!=0])+1	

		tfidf = tf*self.idf
		
		normalizee(tfidf)

		return sparse.csr_matrix(tfidf)



