
from flask import Flask, render_template,request
import os
import numpy as np
import itertools
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np

#csv containing mapping of logs and solutions 
solved_log=pd.read_csv('data/full_solved_log.csv')

#csv containing solutions
solutions=pd.read_csv('data/solutions.csv')


porter_stemmer = PorterStemmer()

def preprocess(s):
    words=word_tokenize(s)
    words=map(lambda w:porter_stemmer.stem(w),words)
    return " ".join(words)

#vectorizer to convert the sentences into vectors having tf-idf weights
vectorizer = TfidfVectorizer(stop_words='english',lowercase="True",preprocessor=preprocess)


from sklearn.decomposition import TruncatedSVD
#SVD model to perform latent semantic analysis on tf-idf vectors
svd_model = TruncatedSVD(n_components=18, 
                         algorithm='arpack',
                         n_iter=10, random_state=42)


#list of sentences having (user description+solution+notes+category) 
document_corpus=list(solved_log.knowledge.values)

from sklearn.pipeline import Pipeline
svd_transformer = Pipeline([('tfidf', vectorizer), 
                            ('svd', svd_model)])


#fitting  model on training data 
svd_matrix = svd_transformer.fit_transform(document_corpus)


#combining solutionname ,category, description into new field "complete" in solutions csv
solutions['complete']=solutions.solutionname+' '+solutions.category+' '+ solutions.description


#sol is list of tuple (solutionname ,description)
sol=zip(solutions.solutionname.values,solutions.description.values)
labels=solved_log.sol_no.values


sol_complete=solutions.complete.values

#using fitted model on all solutions 
sol_vectors=svd_transformer.transform(sol_complete)

sol_vectors=zip(solutions['Unnamed: 0'].values,sol_vectors)



app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html',answer=None)

@app.route('/response',methods=['POST'])
def response():

	#getting query from user
    query=request.form['query']
    answer=[]
    prediction=[]

    #using fitted model on queries
    query_vector=svd_transformer.transform([query])

    #running loop for each query to find best matching solution
    for query_vec in query_vector:
        
        similarity=[]
        for sol_vec in sol_vectors:
            
            similarity.append(np.dot(query_vec,sol_vec[1]))
            

        sol_vectors_list=list(sol_vectors)
        sol_list=list(sol)
        for i in range(4):
            index=similarity.index(max(similarity))
            prediction=sol_vectors_list[index][0]
            answer.append((prediction,sol_list[index],max(similarity)))


            similarity.remove(max(similarity))
            sol_vectors_list.pop(index)
            sol_list.pop(index)
            
    return render_template('index.html',answer=answer)



if __name__ == '__main__':
    app.run(host='0.0.0.0')

