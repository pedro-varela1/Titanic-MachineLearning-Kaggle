# Titanic-MachineLearning-Kaggle
My submit to Titanic Machine Learning competition. It was done by creating a neural network with a hidden layer whose activation function was "relu" and an activation function output "sigmoid". It was use the "Adam" optimizer. Pedro Varela, Electrical Engineering - UFRN. (https://www.kaggle.com/competitions/titanic)

## Feature Engineering
First, some feature engineering were done in order to better understand the type of data we are dealing with. For example, describing the main statistical parameters of the training dataset:
~~~python
dftrain.describe() 
~~~
In addition, data considered relevant to survival on the Titanic were selected:
~~~python
features = ['Sex']+['Pclass']+['Age']+['SibSp']+['Parch']+['Fare']+['Embarked']
~~~
The non-numeric parameters have been converted to numeric:
~~~python
train = dftrain[['Survived']+features].copy()
train['Sex'] = train['Sex'].map({'male': 1, 'female': 0}).astype(int)
train['Age'] = train['Age'].fillna(age_mean)
train = train.dropna()
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
~~~
And this data have been normalized:
~~~python
for f in features:
  max_value = pd.concat([train[f], test[f]], ignore_index=True).max()
  min_value = pd.concat([train[f], test[f]], ignore_index=True).min()
  train[f] = (train[f] - min_value) / (max_value - min_value)
  test[f] = (test[f] - min_value) / (max_value - min_value)
~~~

## The Model:
### Building the model:
A shallow neural network model was built, there is a hidden layer with 4 neurons and activation function "relu" and, for the output, the activation function was "sigmoid".
~~~python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(7,)),
    keras.layers.Dense(4, activation='relu', use_bias=True),
    keras.layers.Dense(2, activation='sigmoid', use_bias=True)
])
~~~
### Compile the model:
It was use the "Adam" optimizer because it'a computationally efficient, has little memory requirement, invariant to diagonal rescaling of gradients, and is well suited for problems that are large in terms of data/parameters
~~~python
model.compile(optimizer='Adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
~~~
### Training the model:
~~~python
model.fit(train, y_train, epochs=28)
~~~

## Save the submission file:
After creating a pandas dataset to save the survival binary outputs for the test file, it is necessary to create a commit .csv file.
~~~python
dfsubmission.to_csv('submission_pedro.csv', index=False)
~~~
