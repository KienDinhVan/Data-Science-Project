import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

def preprocess_data(input_file='reviews.csv'):
    # Load data
    data = pd.read_csv(input_file)
    
    # Initialize CountVectorizer and TfidfTransformer
    vec = CountVectorizer()
    tfidf = TfidfTransformer()
    
    # Fit the vectorizer and transformer on the training data
    X = data['Reviews']
    vec_X = vec.fit_transform(X)
    vec_X = tfidf.fit_transform(vec_X)
    
    # Extract target labels
    Y = data['Rating'].tolist()

    # Save the vectorizer and transformer to files
    with open('vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(vec, vec_file)
    
    with open('tfidf.pkl', 'wb') as tfidf_file:
        pickle.dump(tfidf, tfidf_file)

    return vec_X, Y

if __name__ == "__main__":
    preprocess_data()
