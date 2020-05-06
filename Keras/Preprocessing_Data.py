import numpy as np 
from random import randint
from sklearn.preprocessing import MinMaxScaler

'''
        TRAINING DATA
'''

train_samples = []
train_labels =  []

'''
Example data:
- An experimental drug was tested on individuals from ages 13 to 100
- The trial had 2100 participants. Half were under 65 years old, half were over 65 years old.
- 95% of patients who were 65 or older experienced side effects
- 95% of patientes under 65 experiencied no side effects.
'''

for i in range(1000):
    # 1000 younger individuals
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # 100 older individuals
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

for i in range(50):
    # 50 younger individuals
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # 50 older individuals
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

# Print raw data
#for i in train_samples:
#    print(i)

#for i in train_labels:
#    print(i)

# Keras expects our samples to be in a numpy array or in a list of numpy arrays
train_samples = np.array(train_samples)
train_labels  = np.array(train_labels)

# Scikit-learn MinMaxScaler to scale all the data in a range (0,1)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_sample = scaler.fit_transform((train_samples).reshape(-1,1))

# Print scaled data
#for i in scaled_train_sample:
#    print(i)

'''
        TESTING DATA
'''

test_samples = []
test_labels =  []

for i in range(10):
    # The 5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)

    # The 5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)

for i in range(200):
    # The 95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)

    # The 95% of older individuals who did experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)

scaled_test_sample = scaler.fit_transform((test_samples).reshape(-1,1))
