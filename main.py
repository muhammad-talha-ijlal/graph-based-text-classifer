from text_classifier import TextClassifier
from text_processor import TextPreprocessor
from cli_applet import CLIApplet
from graph_util import GraphUtil
from evaluator import Evaluator

if __name__ == "__main__":
    
    text_preprocessor = TextPreprocessor()
    graph_util = GraphUtil()
    evaluator = Evaluator()
    text_classifier = TextClassifier(graph_util)

    applet = CLIApplet(text_classifier, text_preprocessor, graph_util, evaluator)
    applet.run()
