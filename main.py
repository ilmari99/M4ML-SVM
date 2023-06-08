from SentenceToVec import SentenceToVec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SVM classifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, cohen_kappa_score

SENTIMENT_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}
INVERSE_SENTIMENT_MAP = {v: k for k, v in SENTIMENT_MAP.items()}

def load_df1():
    df1 = pd.read_csv('headline_sentiments/all-data.csv', encoding='latin-1', header = None, names = ['sentiment', 'headline'])
    # Map the sentiment values to integers
    df1['sentiment'] = df1['sentiment'].map(SENTIMENT_MAP)
    return df1

def load_data(balance = False):
    df = load_df1()
    class_counts = df['sentiment'].value_counts()
    if balance:
        # Balance the data
        df = df.groupby('sentiment').head(class_counts.min()).reset_index(drop=True)
    print(f"DF class counts: {class_counts}")
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def train_test_split(df1, test_size=0.2):
    # Shuffle the data
    df1 = df1.sample(frac=1).reset_index(drop=True)
    # Split the data
    split_index = int(test_size * len(df1))
    test_df = df1[:split_index]
    train_df = df1[split_index:]
    return train_df, test_df

def svm_grid_search(train_df, stv):
    # Create a list of sentence vectors
    X_train = [stv.get_sentence_vector(headline) for headline in train_df['headline']]
    # Create a list of labels
    y_train = train_df['sentiment'].tolist()
    # Train a classifier with grid search and cross validation
    parameters = {'kernel': ("rbf",),
                  'C': [10],
                  # The neutral class is not very important, so we can give it a lower weight
                  "class_weight": [{0:1, 1:0.5, 2:1}],
                  "decision_function_shape": ["ovo"],
                  }
    # Best on unbalanced, no smote: {'C': 10, 'class_weight': {0: 1, 1: 0.5, 2: 1}, 'kernel': 'rbf'}
    # Best on balanced, no smote: {'C': 10, 'class_weight': {0: 1, 1: 0.5, 2: 1}, 'kernel': 'rbf'}
    # Best on unbalanced, smote: {'C': 200, 'class_weight': {0: 1, 1: 1, 2: 1}, 'kernel': 'rbf'}
    svc = SVC(verbose=0)
    scorer = lambda clf, X, y: f1_score(y, clf.predict(X), average='macro')

    clf = GridSearchCV(svc, parameters, cv=10, scoring=scorer, verbose=1, n_jobs=-1)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    print(f"Grid search score: ",clf.best_score_)
    return clf.best_estimator_

def train_model(train_df,stv):
    # Create a list of sentence vectors
    X_train = [stv.get_sentence_vector(headline) for headline in train_df['headline']]
    # Create a list of labels
    y_train = train_df['sentiment'].tolist()
    # Train a classifier
    classifier = SVC(C=10, kernel='rbf', class_weight={0: 1, 1: 0.5, 2: 1}, verbose=0)
    classifier.fit(X_train, y_train)
    return classifier


if __name__ == "__main__":
    np.random.seed(42)
    # Load the data
    df1 = load_data(balance = False)
    # Split the data
    train_df, test_df = train_test_split(df1)
    # Create a SentenceToVec object
    stv = SentenceToVec.load("word2vec-google-news-300.model")
    # Train the model
    classifier = train_model(train_df, stv)
    # Predict the sentiment of the test data
    X_test = [stv.get_sentence_vector(headline) for headline in test_df['headline']]
    y_test = test_df['sentiment'].tolist()
    y_pred = classifier.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=INVERSE_SENTIMENT_MAP.values(), output_dict=True)
    for key, value in report.items():
        if isinstance(value, dict):
            for k, v in value.items():
                print(f"{key} {k}: {v}")
            print()
        else:
            print(f"{key}: {value}")
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
    kappa = cohen_kappa_score(y_test, y_pred)
    print(f"Cohen's kappa: {kappa}")
    f1_score = f1_score(y_test, y_pred, average='macro')
    print(f"F1 score: {f1_score}")
    with open("test_headlines.txt", "r") as f:
        test_sentences = f.readlines()
        test_sentences = [sentence.strip() for sentence in test_sentences]
    for sentence in test_sentences:
        print(sentence, end=": ")
        pred = classifier.predict([stv.get_sentence_vector(sentence)])
        print(INVERSE_SENTIMENT_MAP[pred[0]])
    # plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Normalize the confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.round(cm, 2)
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    # Set the ticks to negative, positive, and neutral
    inverted_map = {v: k for k, v in SENTIMENT_MAP.items()}
    keys = list(inverted_map.keys())
    ax.set_xticks(keys)
    ax.set_yticks(keys)
    ax.set_xticklabels([inverted_map[i] for i in keys])
    ax.set_yticklabels([inverted_map[i] for i in keys])
    # Show counts on tiles
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, cm[i,j], ha='center', va='center', color='w')
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    plt.show()




