import os
import sys
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

class CLIApplet:
    def __init__(self, text_classifier, text_preprocessor, graph_util, evaluator):
        self.text_classifier = text_classifier
        self.text_preprocessor = text_preprocessor
        self.graph_util = graph_util
        self.evaluator = evaluator

    def run(self):
        print("Welcome to Text Classifier CLI Applet!")
        print("Available actions:")
        print("1. Evaluation of Graph Based Method")
        print("2. Visualization of a graph")
        print("3. Visualization of MCS of two graphs")
        print("4. Vector-Based Classification (Naive Bayes) Results")
        print("5. Vector-Based Classification (SVM) Results")
        
        print("6. Exit")

        while True:
            action = input("Enter the action number: ")
            
            if action == '1':
                self.evaluate()
            elif action == '2':
                self.visualize_graph()
            elif action == '3':
                self.visualize_mcs()
            elif action == '4':
                self.nb()
            elif action == '5':
                self.sv()
            elif action == '6':
                print("Exiting the program...")
                sys.exit()
            else:
                print("Invalid action. Please enter a valid action number.")
    
    def vectorize_text(self, texts, method='count'):
        if method == 'count':
            vectorizer = CountVectorizer()
        elif method == 'tfidf':
            vectorizer = TfidfVectorizer()
        else:
            raise ValueError("Invalid method. Please choose 'count' or 'tfidf'.")

        X = vectorizer.fit_transform(texts)
        return X, vectorizer

    def classify_vectorized(self, X_train, y_train, X_test, model='nb'):
        if model == 'nb':
            clf = MultinomialNB()
        elif model == 'svm':
            clf = SVC(kernel='linear')
        else:
            raise ValueError("Invalid model. Please choose 'nb' for Naive Bayes or 'svm' for Support Vector Machine.")

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred
    
    def nb(self):
        files = {
            'sports': [os.path.join('sports data', f'{i}.txt') for i in range(1, 16)],
            'sci_ed': [os.path.join('sci_ed data', f'{i}.txt') for i in range(1, 16)],
            'sl_mar': [os.path.join('sl_mar data', f'{i}.txt') for i in range(1, 16)]
        }
        raw_data = {
            'sports': [open(file, 'r', encoding='utf-8').read() for file in files['sports']],
            'sci_ed': [open(file, 'r', encoding='utf-8').read() for file in files['sci_ed']],
            'sl_mar': [open(file, 'r', encoding='utf-8').read() for file in files['sl_mar']],
        }

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(raw_data['sports'] + raw_data['sci_ed'] + raw_data['sl_mar'])

        
        # Combine training data into a single list for vector-based approach
        train_data = raw_data['sports'][:12] + raw_data['sci_ed'][:12] + raw_data['sl_mar'][:12]
        train_labels = ['sports'] * 12 + ['sci_ed'] * 12 + ['sl_mar'] * 12

        # Combine testing data into a single list for vector-based approach
        test_data = raw_data['sports'][12:] + raw_data['sci_ed'][12:] + raw_data['sl_mar'][12:]
        true_labels = ['sports'] * 3 + ['sci_ed'] * 3 + ['sl_mar'] * 3

        # Vectorize training and testing data
        X_train_vectorized, vectorizer = self.vectorize_text(train_data)
        X_test_vectorized = vectorizer.transform(test_data)

        # Vector-based classification with Naive Bayes
        y_pred_nb = self.classify_vectorized(X_train_vectorized, train_labels, X_test_vectorized, model='nb')

        # Evaluate vector-based classification with Naive Bayes
        accuracy_nb = accuracy_score(true_labels, y_pred_nb)
        precision_nb = precision_score(true_labels, y_pred_nb, average='weighted')
        recall_nb = recall_score(true_labels, y_pred_nb, average='weighted')
        f1_nb = f1_score(true_labels, y_pred_nb, average='weighted')

        print("Vector-Based Classification (Naive Bayes) Results:")
        print("Accuracy:", accuracy_nb)
        print("Precision:", precision_nb)
        print("Recall:", recall_nb)
        print("F1 Score:", f1_nb)
        print()

    def sv(self):
        files = {
            'sports': [os.path.join('sports data', f'{i}.txt') for i in range(1, 16)],
            'sci_ed': [os.path.join('sci_ed data', f'{i}.txt') for i in range(1, 16)],
            'sl_mar': [os.path.join('sl_mar data', f'{i}.txt') for i in range(1, 16)]
        }
        raw_data = {
            'sports': [open(file, 'r', encoding='utf-8').read() for file in files['sports']],
            'sci_ed': [open(file, 'r', encoding='utf-8').read() for file in files['sci_ed']],
            'sl_mar': [open(file, 'r', encoding='utf-8').read() for file in files['sl_mar']],
        }

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(raw_data['sports'] + raw_data['sci_ed'] + raw_data['sl_mar'])

        
        # Combine training data into a single list for vector-based approach
        train_data = raw_data['sports'][:12] + raw_data['sci_ed'][:12] + raw_data['sl_mar'][:12]
        train_labels = ['sports'] * 12 + ['sci_ed'] * 12 + ['sl_mar'] * 12

        # Combine testing data into a single list for vector-based approach
        test_data = raw_data['sports'][12:] + raw_data['sci_ed'][12:] + raw_data['sl_mar'][12:]
        true_labels = ['sports'] * 3 + ['sci_ed'] * 3 + ['sl_mar'] * 3

        # Vectorize training and testing data
        X_train_vectorized, vectorizer = self.vectorize_text(train_data)
        X_test_vectorized = vectorizer.transform(test_data)

        # Vector-based classification with SVM
        y_pred_svm = self.classify_vectorized(X_train_vectorized, train_labels, X_test_vectorized, model='svm')

        # Evaluate vector-based classification with SVM
        accuracy_svm = accuracy_score(true_labels, y_pred_svm)
        precision_svm = precision_score(true_labels, y_pred_svm, average='weighted')
        recall_svm = recall_score(true_labels, y_pred_svm, average='weighted')
        f1_svm = f1_score(true_labels, y_pred_svm, average='weighted')

        print("Vector-Based Classification (SVM) Results:")
        print("Accuracy:", accuracy_svm)
        print("Precision:", precision_svm)
        print("Recall:", recall_svm)
        print("F1 Score:", f1_svm)
    def evaluate(self):
        # Perform evaluation
        files = {
            'sports': [os.path.join('sports data', f'{i}.txt') for i in range(1, 16)],
            'sci_ed': [os.path.join('sci_ed data', f'{i}.txt') for i in range(1, 16)],
            'sl_mar': [os.path.join('sl_mar data', f'{i}.txt') for i in range(1, 16)]
        }
        raw_data = {
            'sports': [open(file, 'r', encoding='utf-8').read() for file in files['sports']],
            'sci_ed': [open(file, 'r', encoding='utf-8').read() for file in files['sci_ed']],
            'sl_mar': [open(file, 'r', encoding='utf-8').read() for file in files['sl_mar']],
        }

        tokens = {
            'sports': [self.text_preprocessor.preprocess(text) for text in raw_data['sports']],
            'sci_ed': [self.text_preprocessor.preprocess(text) for text in raw_data['sci_ed']],
            'sl_mar': [self.text_preprocessor.preprocess(text) for text in raw_data['sl_mar']],
        }

        graphs = {
            'sports': [self.graph_util.build_graph(token) for token in tokens['sports']],
            'sci_ed': [self.graph_util.build_graph(token) for token in tokens['sci_ed']],
            'sl_mar': [self.graph_util.build_graph(token) for token in tokens['sl_mar']],
        }

        labels = ['sports', 'sci_ed', 'sl_mar']
        train_graphs = {category: graphs[category][:12] for category in graphs}
        test_graphs = {category: graphs[category][12:] for category in graphs}


        y_pred, y_test = self.text_classifier.knn_classification(train_graphs, test_graphs)
        accuracy, precision, recall, f1 = self.evaluator.evaluate_classification(y_test, y_pred)
        self.evaluator.plot_confusion_matrix(y_test, y_pred, labels)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

    def visualize_graph(self):
        category = input("Enter the category (sports, sci_ed, sl_mar): ")
        index = int(input("Enter the index of the graph: "))
        if category in ['sports', 'sci_ed', 'sl_mar'] and 0 <= index < 15:
            graph = self.graph_util.build_graph(
                self.text_preprocessor.preprocess(
                    open(f"{category} data/{index+1}.txt", 'r', encoding='utf-8').read()))
            self.graph_util.visualize_graph(graph)
        else:
            print("Invalid category or index.")

    def visualize_mcs(self):
        category = input("Enter the category (sports, sci_ed, sl_mar): ")
        index = int(input("Enter the index of the graph: "))
        if category in ['sports', 'sci_ed', 'sl_mar'] and 0 <= index < 15:
            category2 = input("Enter the category (sports, sci_ed, sl_mar): ")
            index2 = int(input("Enter the index of the graph: "))
            if category2 in ['sports', 'sci_ed', 'sl_mar'] and 0 <= index2 < 15:
                graph1 = self.graph_util.build_graph(
                    self.text_preprocessor.preprocess(
                        open(f"{category} data/{index+1}.txt", 'r', encoding='utf-8').read()))
                graph2 = self.graph_util.build_graph(
                    self.text_preprocessor.preprocess(
                        open(f"{category2} data/{index2+1}.txt", 'r', encoding='utf-8').read()))
                mcs = self.graph_util.get_mcs(graph1, graph2)
                self.graph_util.visualize_graph(mcs)
            else:
                print("Invalid category or index.")
        else:
            print("Invalid category or index.")
