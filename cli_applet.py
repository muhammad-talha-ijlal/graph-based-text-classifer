import os
import sys

class CLIApplet:
    def __init__(self, text_classifier, text_preprocessor, graph_util, evaluator):
        self.text_classifier = text_classifier
        self.text_preprocessor = text_preprocessor
        self.graph_util = graph_util
        self.evaluator = evaluator

    def run(self):
        print("Welcome to Text Classifier CLI Applet!")
        print("Available actions:")
        print("1. Evaluation")
        print("2. Visualization of a graph")
        print("3. Visualization of MCS of a graph")
        print("4. Exit")

        while True:
            action = input("Enter the action number: ")
            
            if action == '1':
                self.evaluate()
            elif action == '2':
                self.visualize_graph()
            elif action == '3':
                self.visualize_mcs()
            elif action == '4':
                print("Exiting the program...")
                sys.exit()
            else:
                print("Invalid action. Please enter a valid action number.")

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
        print("Accuracy:", accuracy, '%')
        print("Precision:", precision, '%')
        print("Recall:", recall, '%')
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
