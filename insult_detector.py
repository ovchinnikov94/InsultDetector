__author__ = 'Dmitriy Ovchinnikov'

from sklearn.preprocessing import LabelEncoder
from time import time
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PassiveAggressiveClassifier
import re
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV


class InsultDetector:

    def __init__(self):
        """
        it is constructor. Place the initialization here. Do not place train of the model here.
        :return: None
        """
        self.labelEncoder = LabelEncoder()
        self.classifier = None

    def train(self, labeled_discussions, research=False):
        """
        This method train the model.
        :param discussions: the list of discussions. See description of the discussion in the manual.
        :return: None
        """
        # TODO put your code here
        print("Preparing corpus...")
        t = time()
        data = [[], []]
        for root in labeled_discussions:
        	for node in root.get('root').get('children'):
        		new_data = self.getData(node)
        		data[0] = data[0] + new_data[0]
        		data[1] = data[1] + new_data[1]
        print("Preparing finished! %0.3fs" % (time() - t))
        l = self.labelEncoder.fit_transform(data[1])
        
        cross_validating = cross_validation.StratifiedKFold(l, n_folds=8)

        if (research):
        	text_clf = Pipeline([
        		('vect', CountVectorizer()),
        		('tf_idf', TfidfTransformer()),
        		('clf', PassiveAggressiveClassifier(n_jobs=-1))
        		])
        	param = {'vect__ngram_range' : ((1,5), (1, 6), (1, 7), (1, 8)),
        		'vect__max_df' : (0.4, 0.5, 0.6, 0.8, 1.0),
        		'vect__analyzer' : ('char_wb', 'word', 'char'),
        		'tf_idf__norm' : ('l1', 'l2'),
        		'clf__n_iter' : (10, 50, 70),
        		'clf__C' : (0.6, 0.8, 1.0, 1.1) } 
        	print ("GridSearch running...")
        	t_gs = time()
        	gs_clf = GridSearchCV(text_clf, param, n_jobs=-1, verbose=1)
        	gs_clf.fit(data[0][:200], l[:200])
        	print("GridSearch finished! %0.3fs" % (time()-t_gs))
        	best_parametres, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
        	for param_name in sorted(param.keys()):
        		print("%s: %r" % (param_name, best_parametres[param_name]))
        	print("Score: %0.4f" % score)
        else:
        	self.classifier = Pipeline([
        		('vect', CountVectorizer(analyzer='char_wb', max_df=0.4, max_features=None, ngram_range=(1, 5))),
        		('tf_idf', TfidfTransformer(norm='l1')),
        		('clf', PassiveAggressiveClassifier(C=0.6, n_iter=10))
        		])
        	print("Training...")
        	self.classifier.fit(data[0], l)
        	print ("Training succesfully finished! %0.3fs" % (time() - t))


    def classify(self, unlabeled_discussions):
        """
        This method take the list of discussions as input. You should predict for every message in every
        discussion (except root) if the message insult or not. Than you should replace the value of field "insult"
        for True if the method is insult and False otherwise.
        :param discussion: list of discussion. The field insult would be replaced by False.
        :return: None
        """
        # TODO put your code here
        if (unlabeled_discussions != None):
        	for root in unlabeled_discussions:
        		for child in root['root']['children']:
        			self.recursion_clf(child)
        return unlabeled_discussions

    def recursion_clf(self, discussion):
    	if (discussion != None):
    		#print(discussion.get('id'))
    		text = discussion.get('text')
    		if (text != None):
    			if (len(text) > 0):
    				discussion['insult'] = self.labelEncoder.inverse_transform(self.classifier.predict([text]))[0]
    		if (discussion.get('children') != None):
    			for child in discussion['children']:
    				self.recursion_clf(child)
    
    def prepare(self, text):
    	text = text.lower()
    	text = re.sub(r"[\)]+", ')', text)
    	text = re.sub(r"[\(]+", '(', text) 
    	text = re.sub(
        	r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        	'', text)
    	return text

    def getData(self, discussion):
    	result1 = []
    	result2 = []
    	if (discussion != None):
    		childs = discussion.get('children')
    		if (childs != None):
    			for child in childs:
    				insult = child.get('insult')
    				text = child.get('text')
    				if (text != None and insult != None):
    					result1.append(text)
    					result2.append(insult)
    				data_child = self.getData(child)
    				if (len(data_child) == 2):
    					result1 = result1 + data_child[0]
    					result2 = result2 + data_child[1]
    		return [result1, result2]
    	else:
    		return result


if __name__ == '__main__':
	detector = InsultDetector()
	t0 = time()
	print("Loading corpus...")
	corpus = json.load(open('./discussions.json'))
	print('Corpus loaded %0.3fs' % (time() - t0))
	detector.train(corpus)
	for root in detector.classify(corpus):
		for item in root.get('root').get('children')[:10]:
			print(item.get('text'))
			print(item.get('insult'))
