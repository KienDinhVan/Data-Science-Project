import pickle

def predict():
    # Load the vectorizer, transformer, and model
    with open('vectorizer.pkl', 'rb') as vec_file:
        vec = pickle.load(vec_file)
    
    with open('tfidf.pkl', 'rb') as tfidf_file:
        tfidf = pickle.load(tfidf_file)
    
    with open('rating_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Get user input for review
    txt = input('Your review: ')
    X = [txt]
    
    # Apply vectorizer and transformer
    X = vec.transform(X)
    X = tfidf.transform(X)

    # Predict rating
    y_pred = model.predict(X)
    print('Rating: ', y_pred[0])

if __name__ == "__main__":
    predict()
