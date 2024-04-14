import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("....Reading DataSet and Creating Pandas DataFrame....")
alphabet_data = pd.read_csv("A_Z Handwritten Data.csv")
print("...DataFrame Created...")
print("...Slicing and creating initial training and testing set...")
# Dataset of letter A containing features
X_Train_A = alphabet_data.iloc[:13869, 1:]
# Dataset of letter A containing labels
Y_Train_A = alphabet_data.iloc[:13869, 0]

# Dataset of letter C containing features
X_Train_C = alphabet_data.iloc[22537:45946, 1:]
# Dataset of letter C containing labels
Y_Train_C = alphabet_data.iloc[22537:45946, 0]

# Dataset of letter L containing features
X_Train_P = alphabet_data.iloc[222789:234355, 1:]
# Dataset of letter L containing labels
Y_Train_P = alphabet_data.iloc[222789:234355, 0]

# Dataset of letter O containing features
X_Train_N = alphabet_data.iloc[120801:139811, 1:]
# Dataset of letter O containing labels
Y_Train_N = alphabet_data.iloc[120801:139811, 0]

# Joining the Datasets of all letters
X_Train = pd.concat([X_Train_N, X_Train_P,X_Train_A, X_Train_C], ignore_index=True)
Y_Train = pd.concat([Y_Train_N, Y_Train_P,Y_Train_A, Y_Train_C], ignore_index=True)

print("...X_Train and Y_Train created...")# Train-test split
train_features, test_features, train_labels, test_labels = train_test_split(X_Train, Y_Train, test_size=0.25, random_state=0, shuffle=True)

# Random Forest classifier created
clf = RandomForestClassifier(n_estimators=100, random_state=0)  # Adjust parameters as needed
print("...Training the Model...")
clf.fit(train_features, train_labels)
print("...Model Trained...")

# Predictions
labels_predicted = clf.predict(test_features)

# Accuracy
accuracy = accuracy_score(test_labels, labels_predicted)
print("Accuracy of the model:", accuracy)

# Saving the trained model
print("...Saving the trained model...")
joblib.dump(clf, "NPAC_rf.pkl", compress=3)
print("...Model Saved...")