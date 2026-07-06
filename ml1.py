from sklearn.datasets import load_digits

import pandas as pd
import seaborn as sns

digits = load_digits()

print(digits.DESCR)     # tells us details that is located in load_digits (1797 sample each with 64 features and the targets are numbers between 0-9)

print(digits.data[:2])  # previewing the first two rows, numbers that show the pixel intensity on darkness scale (0-16) 16 being darkest

print(digits.data.shape)    # 1794 rows(sample) and 64 columns(features)

print(digits.target[:2])    # each row contains: 64 features describing a target 0 [0,1], second row: 64 features describing target 1 [0,1]

print(digits.target.shape)      # only one column (1794,0)

import matplotlib.pyplot as plt

figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6,4))

    # zip() combines three iterables together:
    # 1. axes.ravel() -> flattens the 4x6 grid of subplot axes into a 1D sequence
    # 2. digits.images -> the actual 8x8 images of handwritten digits
    # 3. digits.target -> the correct digit label (0-9) for each image
    
    # On each iteration, item contains:
    # (one subplot axis, one image, one target label)


for item in zip(axes.ravel(), digits.images, digits.target):        # Explanation provided above
    axes, image, target = item                                          # Unpack the tuple stored in item: axes, image, target
    axes.imshow(image, cmap=plt.cm.gray_r)                              # Display the digit image on the current subplot. gray_r means reversed grayscale: darker pixels appear black, lighter appear white.
    axes.set_xticks([])                                                 # Remove tick marks and labels from the x-axis, so images look cleaner
    axes.set_yticks([])                                                 # Remove tick marks and labels from the y-axis
    axes.set_title(target)                                              # Set the title above the image. This shows the actual digit (0-9) represented by the image.

plt.tight_layout()                                                      # automatically adjusts the spacing between subplots

#plt.show()                                                              # tells Matplotlib to display the figure window containing all the plots

# Split the dataset into two parts: training model and testing model 
    # does it in random fashion but can be changed to not be random 
    # splits dataset in to 75% training and 25% Testing
    # Function listed below:

from sklearn.model_selection import train_test_split

# Specifying with both data & target (train) and data & target (test)
data_train, data_test, target_train, target_test = train_test_split(digits.data, digits.target, random_state=11)

print(data_train.shape)         # 1347/1797 = 75% which is why the training output is 1347 records
print(target_train.shape)       # (1347,0)
print(data_test.shape)          # 450/1797 = 25% which is why the test output is 450 records
print(target_test.shape)        # (450,0)

# Training of the model, using the kneighborsclassifier estimator 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X=data_train, y=target_train)       # Training puporse: X uppercase and y lowercase does matter. Method called fit runs the data and learns 

predicted = knn.predict(X=data_test)        # Predict what the y is going to be

expected = target_test                      # changing name so it is easier to understand that target is the expected 

# Printing out the 1st 20 elements of predicited and expected
print(predicted[:20])       # from the result, did pretty good besides the second to last item 5 
print(expected[:20])

# Result: 
# [0 4 9 9 3 1 4 1 5 0 4 9 4 1 5 3 3 8 5 6]
# [0 4 9 9 3 1 4 1 5 0 4 9 4 1 5 3 3 8 3 6]


wrong = [(p,e) for (p, e) in zip(predicted, expected) if p != e]

print(wrong)        # it did okay, next is to determine how well it actually did

print(f"{knn.score(data_test, target_test):.2%}")
# result: 97.78% is the accurancy rate


#################################################### CONFUSION MATRIX ####################################################

from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true=expected, y_pred=predicted)

print(confusion)

# result: 
#    0  1  2  3  4  5  6  7  8  9

# [[45  0  0  0  0  0  0  0  0  0]  predicted good
#  [ 0 45  0  0  0  0  0  0  0  0]  predicted good
#  [ 0  0 54  0  0  0  0  0  0  0]  predicted good
#  [ 0  0  0 42  0  1  0  1  0  0]  wrong: predicted 5 & 7, predicted wrong twice
#  [ 0  0  0  0 49  0  0  1  0  0]  wrong
#  [ 0  0  0  0  0 38  0  0  0  0]  right
#  [ 0  0  0  0  0  0 42  0  0  0]  right
#  [ 0  0  0  0  0  0  0 45  0  0]  right
#  [ 0  1  1  2  0  0  0  0 39  1]  wrong
#  [ 0  0  0  0  1  0  0  0  1 41]] wrong


# make sure to install pandas and seaborn before this next step

################################################ VISUALIZE CONFUSION MATRIX AS HEATMAP ####################################################

import matplotlib.pyplot as plt2

confusion_df = pd.DataFrame(confusion,index=range(10), columns=range(10))

figure = plt2.figure(figsize=(7,6))

axes = sns.heatmap(confusion_df, annot=True, cmap=plt2.cm.nipy_spectral_r)

plt2.show()

print("done")














