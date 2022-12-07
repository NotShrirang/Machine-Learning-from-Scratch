import neuralnetwork as nn
import numpy as np
import pandas as pd
import random

def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

def vertical_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    data = []
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        X[ix] = np.c_[np.random.randn(samples)*.1 + (class_number)/3, np.random.randn(samples)*.1 + 0.5]
        y[ix] = class_number
    return X, y

def linear_data(hm, variance, step=2, correlation=True):
    val = 1
    ys = []
    for _ in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation or correlation == 'pos':
            val += step
        elif correlation or correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

if __name__ == '__main__':
    np.random.seed(0)
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split


    df = pd.read_csv('../Datasets/breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], axis=1, inplace=True)

    df = df.astype(float)

    X = np.array(df.drop(['class'], axis=1))
    y = np.array(df['class'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = nn.models.Sequential([
                    nn.layers.Layer_Dense(9, 128, activation=nn.activations.ReLU),
                    nn.layers.Layer_Dense(128, 128, activation=nn.activations.ReLU),
                    nn.layers.Layer_Dense(128, 128, activation=nn.activations.ReLU),
                    nn.layers.Layer_Dense(128, 2, activation=nn.activations.Softmax),
                ])
    y_train: pd.DataFrame = y_train.astype(np.uint8)
    y_train[y_train==2] = 0
    y_train[y_train==4] = 1

    y_test: pd.DataFrame = y_test.astype(np.uint8)
    y_test[y_test==2] = 0
    y_test[y_test==4] = 1

    history = model.fit(X_train, y_train.astype(np.uint8), epoch=4, print_output=True)
    model.save("MyModel1")
    # model = nn.models.load_model(path_to_model="MyModel1")
    # print(history)
    # predictions = model.predict(X_test)
    # print([np.argmax(prediction[0]) for prediction in predictions])
    # print(np.mean([np.argmax(prediction[0]) for prediction in predictions]==y_test))
    predictions = model.predict([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
    print([np.argmax(prediction[0]) for prediction in predictions])
