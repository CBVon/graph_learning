import sys
import cPickle


from keras.models import load_model  

from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

timeStamp_model = "1529836375_model"

testX = cPickle.load(open(timeStamp_model + "/testX.pkl", "rb"))
testY = cPickle.load(open(timeStamp_model + "/testY.pkl", "rb"))

model = load_model(timeStamp_model + '/lstmModel.h5')
predY = model.predict_classes(testX)
print "precisionScore : " + str(precision_score(testY, predY))
print str(classification_report(testY, predY))










