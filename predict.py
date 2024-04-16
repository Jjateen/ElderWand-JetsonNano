import warnings
from PIL import Image
import joblib
import numpy as np
warnings.filterwarnings("ignore", message='UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names"X does not have valid feature names, but"')
# Loading the processed last frame form Desktop
img = Image.open("lastframe.jpg")

# Loading the SVM classifier
clf = joblib.load("NPAC_rf.pkl")

# Converting image to numpy array
img = np.array(img)
# Converting the numpy array to 1-Dimensional array
img = img.reshape(1, -1)


prediction = clf.predict(img)
print(prediction)