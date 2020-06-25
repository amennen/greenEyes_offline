trainingLabels = np.array([paranoidLabel,cheatingLabel ]).flatten()
clf.fit(training_data_classifier_SRM,trainingLabels)
print(clf.predict(training_data_classifier_SRM))
print(clf.predict_proba(training_data_classifier_SRM))
print(clf.classes_)

# original training Labels: [1,1,1,1,1,1,] from 0:20 and [00000000] from 20:40
# output: predictions are 1's and 0's 
# array predictions: 1st col is prob(class1) and second col is prob(class 0)
# classes: [0 1]
############################################################################
trainingLabels2 = trainingLabels
trainingLabels2[0:19] = 0
trainingLabels2[19:] = 1
print(trainingLabels2)

clf.fit(training_data_classifier_SRM,trainingLabels2)
print(clf.predict(training_data_classifier_SRM))
print(clf.predict_proba(training_data_classifier_SRM))
print(clf.classes_)

# output: predictions are 0 and 1
# array predictions: 1st col is prob(class 1) and 2nd col is prob(class 0)
# classes [0 1]


############################################################################
trainingLabels3 = trainingLabels
trainingLabels3[0:19] = 1
trainingLabels3[19:] = 2

clf.fit(training_data_classifier_SRM,trainingLabels3)
print(clf.predict(training_data_classifier_SRM))
print(clf.predict_proba(training_data_classifier_SRM))
print(clf.classes_)

# output: predictions are 1 and 2
# array predictions: 1st col is prob(class 2) and 2nd col is prob(class 1)
# classes [1 2]

############################################################################
trainingLabels4 = trainingLabels
trainingLabels4[0:19] =2
trainingLabels4[19:] = 1

clf = SVC(kernel='linear', probability=True)
clf.fit(training_data_classifier_SRM,trainingLabels4)
print(clf.predict(training_data_classifier_SRM))
print(clf.predict_proba(training_data_classifier_SRM))
print(clf.classes_)

# output: predictions are 2 and 1
# array predictions: 1st col is prob(class 2) and 2nd col is prob(class 1)
# classes [1 2 ]


############################################################################
trainingLabels5 = trainingLabels
trainingLabels5 = ['k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','p','p','p','p','p','p','p','p','p','p','p','p','p','p','p','p','p','p','p']
print(trainingLabels5) 

clf = SVC(kernel='linear', probability=True)
clf.fit(training_data_classifier_SRM,trainingLabels5)
print(clf.predict(training_data_classifier_SRM))
print(clf.predict_proba(training_data_classifier_SRM))
print(clf.classes_)

# output: predictions are k and p
# array predictions: 1st col is prob(class p) and 2nd col is prob(class k)
# classes [k p]

############################################################################
trainingLabels6 = [  'kw','kw','kw','kw','kw','kw','kw','kw','kw','kw','kw','kw','kw','kw','kw','kw','kw','kw','kw','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k',]
print(trainingLabels6) 

clf = SVC(kernel='linear', probability=True)
clf.fit(training_data_classifier_SRM,trainingLabels6)
print(clf.predict(training_data_classifier_SRM))
print(clf.predict_proba(training_data_classifier_SRM))
print(clf.classes_)

# output: predictions are kw and k
# array predictions: 1st col is prob(class kw) and 2nd col is prob(class k)
# classes [k kw]