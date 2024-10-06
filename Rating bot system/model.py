import pickle
from sklearn.tree import DecisionTreeClassifier
from preprocessing import preprocess_data

def train_model():
    # Preprocess data
    vec_X, Y = preprocess_data()

    # Train Decision Tree model
    model = DecisionTreeClassifier()
    model.fit(vec_X, Y)

    # Save the model
    with open('rating_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

if __name__ == "__main__":
    train_model()
