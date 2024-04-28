from collections import Counter

class TextClassifier:
    def __init__(self, graph_util):
        self.graph_util = graph_util

    def custom_knn_predict(self, train_graphs, test_graph):
        scores = []
        for category, graphs in train_graphs.items():
            for train_graph in graphs:
                score = self.graph_util.get_score(test_graph, train_graph)
                scores.append((category, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_30 = [category for category, _ in scores[:30]]
        majority_label = Counter(top_30).most_common(1)[0][0]
        return majority_label

    def knn_classification(self, train_graphs, test_graphs):
        y_pred = []
        y_test = []
        for category, graphs in test_graphs.items():
            for test_graph in graphs:
                prediction = self.custom_knn_predict(train_graphs, test_graph)
                y_pred.append(prediction)
                y_test.append(category)
        return y_pred, y_test
