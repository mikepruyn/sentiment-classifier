from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from parse_data import parse_review


#initialize SVM model
SVM = SVC(C=10000, gamma='auto')

#format data for training
X, Y, vocab = parse_review('sorted_data/apparel/all.review')
X_baby, Y_baby, vocab = parse_review('sorted_data/baby/all.review', vocab)

#Split data into test and train groups
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


#train the model and score it's accuracy on heldout test data 
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)

#do the same for a different category of review
y_pred_baby = SVM.predict(X_baby)
baby_score = accuracy_score(Y_baby, y_pred_baby)
print(baby_score)
