## evaluate learners on data
import scipy.io as sio
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
import logging
from sklearn.externals import joblib
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
import numpy
numpy.random.seed()

def load_labels(label_pickle,test_labels):

    type_train = []
    gender_train = []

    
    type_test = []
    gender_test = []

    with open(label_pickle, "rb") as input_file:
        e = pickle.load(input_file)
        
    with open(test_labels, "rb") as input_file:
        et = pickle.load(input_file)

    for el in e:
        type_train.append(el['type'])
        gender_train.append(el['gender'])

        
    for el in et:
        type_test.append(el['type'])
        gender_test.append(el['gender'])


    encoder_type = preprocessing.LabelEncoder().fit(type_train+type_test)
    encoder_gender = preprocessing.LabelEncoder().fit(gender_train+gender_test)

        
    label_object = {}
    label_object['gender'] = (encoder_gender.transform(gender_train),encoder_gender.transform(gender_test))
    label_object['type'] = (encoder_type.transform(type_train),encoder_type.transform(type_test))

    encoders = (encoder_type,encoder_gender)
    return label_object,encoders
    
    
        
if __name__ == "__main__":

    for lang in ['en', 'es']:
        for task in ['bot', 'gender']:

            fname = "../train_data/train_instances_" + task + "_" + lang + ".mat"
            labels_train = "../train_data/train_labels_" + task + "_" + lang + ".pickle"
            labels_test = "../train_data/test_labels_" + task + "_" + lang + ".pickle"

            dmat = sio.loadmat(fname)
            train_features = dmat['train_features']
            test_features = dmat['test_features']
            label_vectors,encoders = load_labels(labels_train,labels_test)
            encoder_type,encoder_gender = encoders

            outfile = open("../train_data/encoder_" + task + "_" + lang + ".pickle",'wb')
            pickle.dump(encoder_type,outfile)
            outfile.close()

            outfile = open("../train_data/encoder_" + task + "_" + lang + ".pickle",'wb')
            pickle.dump(encoder_gender,outfile)
            outfile.close()

            preds = {}
            print('Evaluation on task and language: ', task, lang )
            for target, vals in label_vectors.items():

                if target == task or (task=='bot' and target=="type"):
                    train_labels = vals[0]
                    test_labels = vals[1]

                    clf = LogisticRegression(C=1e2, fit_intercept=False)
                    clf.fit(train_features,train_labels)
                    joblib.dump(clf, '../train_data/trained_LR_' + task + "_" + lang + '.pkl')
                    predictions = clf.predict(test_features)

                    accuracy = accuracy_score(predictions,test_labels)
                    logging.info("{} Performed with {}".format(target, accuracy))
                    preds[target] = accuracy
            total_score = 1/np.sum([1/sc for sc in preds.values()])
            logging.info("Total score {}".format(total_score))

#to beat
'''
Evaluation on task and language:  bot en
26-Apr-19 15:49:34 - type Performed with 0.9016129032258065
26-Apr-19 15:49:34 - Total score 0.9016129032258065
Evaluation on task and language:  gender en
26-Apr-19 15:49:36 - gender Performed with 0.7951612903225806
26-Apr-19 15:49:36 - Total score 0.7951612903225806

Evaluation on task and language:  bot es
26-Apr-19 15:49:37 - type Performed with 0.8804347826086957
26-Apr-19 15:49:37 - Total score 0.8804347826086957
Evaluation on task and language:  gender es
26-Apr-19 15:49:38 - gender Performed with 0.6695652173913044
26-Apr-19 15:49:38 - Total score 0.6695652173913044'''
