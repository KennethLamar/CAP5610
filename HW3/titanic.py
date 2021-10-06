import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt

def main():
    # Import csv into Pandas dataframe.
    df = pd.read_csv("./data/train.csv")
    
    # Preprocess data.
    # Remove the Survived column. This is what we want to predict.
    X = df.drop(columns="Survived")
    # Remove the PassengerId column. It serves no predictive purpose.
    X.drop(columns="PassengerId", inplace=True)
    # Keep last names only.
    X["Name"] = X["Name"].str.rsplit(",", expand=True)[0]
    # Encode last names as numbers.
    le = LabelEncoder()
    X["Name"] = le.fit_transform(X["Name"])
    # Convert sex to numbers. Scikit cannot work with strings natively.
    X["Sex"].replace(to_replace="male", value="0", inplace=True)
    X["Sex"].replace(to_replace="female", value="1", inplace=True)
    # Assume an unknown age is the average.
    X.loc[X["Age"].isnull(),"Age"] = X["Age"].mean()
    # Remove non-digits from ticket IDs.
    X["Ticket"] = X["Ticket"].str.rsplit(" ", expand=True)[0]
    X["Ticket"] = X["Ticket"].str.replace(r'\D', '')
    X.loc[X["Ticket"] == "","Ticket"] = 0
    # Assume an unknown cabin is it's own fictional cabin.
    X.loc[X["Cabin"].isnull(),"Cabin"] = "N100"
    # Ensure only one cabin per person is considered.
    X["Cabin"] = X["Cabin"].str.rsplit(" ", expand=True)[0]
    # Split the cabin by room and deck.
    X["Deck"] = X.apply(lambda row: ord(row["Cabin"][0]) - ord("A"), axis=1)
    X["Room"] = X["Cabin"].str[1:]
    X.loc[X["Room"] == "","Room"] = 0
    # Remove the Cabin column. It is now redundant.
    X.drop(columns="Cabin", inplace=True)
    # Convert Embarked to numbers. Scikit cannot work with strings natively.
    X["Embarked"].replace(to_replace="S", value="1", inplace=True)
    X["Embarked"].replace(to_replace="C", value="2", inplace=True)
    X["Embarked"].replace(to_replace="Q", value="3", inplace=True)
    X.loc[X["Embarked"].isnull(),"Embarked"] = 0
    # Ensure all numeric fields are actually numbers.
    X["Pclass"] = X["Pclass"].astype(int)
    X["Name"] = X["Name"].astype(int)
    X["Sex"] = X["Sex"].astype(int)
    X["Age"] = X["Age"].astype(float)
    X["SibSp"] = X["SibSp"].astype(int)
    X["Parch"] = X["Parch"].astype(int)
    X["Ticket"] = X["Ticket"].astype(int)
    X["Fare"] = X["Fare"].astype(float)
    X["Deck"] = X["Deck"].astype(int)
    X["Room"] = X["Room"].astype(int)
    X["Embarked"] = X["Embarked"].astype(int)
    # Carefully select a subset of our features.
    features = ['Pclass', 'Sex', 'Age']#, 'Fare', 'Deck', 'Embarked']
    X = X.loc[:, features]
    # Keep our known values as training outputs.
    Y = df["Survived"].astype(int)

    # DEBUG: Print out the contents of our filtered dataset.
    # with open('output.txt', 'w') as f:
    #     original_stdout = sys.stdout
    #     sys.stdout = f
    #     pd.set_option("display.max_rows", None, \
    #                   "display.max_columns", None, \
    #                   "display.max_colwidth", None)
    #     print(X)
    #     sys.stdout = original_stdout

    # Train various SVM kernel functions.
    # Linear
    linearClassifier = svm.SVC(kernel='linear')
    linearClassifier = linearClassifier.fit(X, Y)
    print("Computed linear classifier")
    # Quadratic
    quadraticClassifier = svm.SVC(kernel='poly', degree=2)
    quadraticClassifier = quadraticClassifier.fit(X, Y)
    print("Computed quadratic classifier")
    # RBF
    rbfClassifier = svm.SVC(kernel='rbf')
    rbfClassifier = rbfClassifier.fit(X, Y)
    print("Computed RBF classifier")

    # Use five-fold cross validation on each.
    print("Performing 5-fold cross validation.")
    linearScores = cross_val_score(linearClassifier, X, Y, cv=5)
    print("Computed linear score")
    quadraticScores = cross_val_score(quadraticClassifier, X, Y, cv=5)
    print("Computed quadratic score")
    rbfScores = cross_val_score(rbfClassifier, X, Y, cv=5)
    print("Computed RBF score")
    # Report average classification accuracy.
    print("Linear    accuracy: " + str(linearScores.mean()))
    print("Quadratic accuracy: " + str(quadraticScores.mean()))
    print("RBF       accuracy: " + str(rbfScores.mean()))


if __name__ == "__main__":
    main()
