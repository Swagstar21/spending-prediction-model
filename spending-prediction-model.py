import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.pyplot as plt


def sine(x):
	return np.sine(x)

def spike(x):
	if int(x) % 4 == 0:
		return 100.0
	else:
		return 0.0

def constant(x):
	return np.array([100 * len(x)])

model = Sequential()
model.add(LSTM(20, activation='tanh'))
model.add(Dense(1, activation='tanh'))
model.compile(optimizer='adam', loss='mse')

def train_function(x):
    return np.sin(x * 3.14 / 8)
    # return np.array([spike(elem) for elem in x])
    # return np.array([100.0 for elem in x])
    # return np.array([elem * 3.0 for elem in x])

def split_sequence(sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
                end_ix = i + n_steps
                if end_ix > len(sequence)-1:
                        break
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                X.append(seq_x)
                y.append(seq_y)
        return np.array(X), np.array(y)

xaxis = np.arange(0, 100, 1)
train_seq = train_function(xaxis)
n_steps = 20
X, y = split_sequence(train_seq, n_steps)

n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
print("X.shape = {}".format(X.shape))
print("y.shape = {}".format(y.shape))

model.fit(X, y, epochs=20, verbose=1)

X_predicted = [x[0][0] + 20 for x in X]
y_predicted = y.reshape(-1)
plt.plot(y_predicted)
plt.ylabel("Money")
plt.xlabel("Time")
plt.show()

X_predicted = X

y_predicted = model.predict(X_predicted)
print(y_predicted.shape)
print(X_predicted.shape)
X_predicted = [x[0][0] + 20 for x in X_predicted]
y_predicted = y_predicted.reshape(-1)
plt.plot(y_predicted)
plt.ylabel("Money")
plt.xlabel("Time")
plt.show()