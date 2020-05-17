""" Method to read the csv file using Pandas 
and later use this data for linear regression. """

# Library to read csv file effectively
import pandas
import matplotlib.pyplot as plt
import numpy as np


# Method to read the csv file
def load_data(file_name):
    print(file_name)
    column_names = ['area', 'price']
    # To read columns
    io = pandas.read_csv(file_name, names=column_names, header=None)
    x_val = (io.values[1:, 0])
    y_val = (io.values[1:, 1])
    size_array = len(y_val)
    for i in range(size_array):
        x_val[i] = float(x_val[i])
        y_val[i] = float(y_val[i])
    return x_val, y_val


def feature_normalize(train_X):
    global mean, std
    mean = np.mean(train_X, axis=0)
    std = np.std(train_X, axis=0)
    print(mean, std)
    return (train_X - mean) / std


def main_linear_regression(file_name):
    # Call the method for a specific file
    x_raw, y_raw = load_data(file_name)
    x = feature_normalize(x_raw)
    y = y_raw

    linear_regression_method(x, y)


def linear_regression_method(x, y):
    # Modeling
    w, b = 0.0, 0.0
    num_epoch = 100
    converge_rate = np.zeros([num_epoch, 1], dtype=float)
    learning_rate = 1e-3

    for e in range(num_epoch):
        # Calculate the gradient of the loss function with respect to arguments (model parameters) manually.
        y_predicted = w * x + b
        grad_w, grad_b = (y_predicted - y).dot(x), (y_predicted - y).sum()

        # Update parameters.
        w, b = w - learning_rate * grad_w, b - learning_rate * grad_b
        converge_rate[e] = np.mean(np.square(y_predicted - y))
    print(w, b)

    y_estimated = w * x + b

    # Plot the data
    plt.rcParams.update({'font.size': 20})
    fig = plt.figure()
    fig.set_size_inches(18, 8)
    ax = fig.add_subplot(121)

    ax.scatter(x, y, color='green', s=70)
    ax.set_xlabel('Apartment area (square meters)')
    ax.set_ylabel('Apartment price (1000 Euros)')
    ax.set_title('Apartment price vs area in Berlin')
    ax.plot(x, y_estimated, linewidth=4.0, color='red')

    ax.spines['bottom'].set_color('orange')
    ax.spines['bottom'].set_linewidth(3.0)
    ax.spines['top'].set_color('orange')
    ax.spines['top'].set_linewidth(3.0)
    ax.spines['right'].set_color('orange')
    ax.spines['right'].set_linewidth(3.0)
    ax.spines['left'].set_color('orange')
    ax.spines['left'].set_linewidth(3.0)

    ax.tick_params(axis='x', colors='orange')
    ax.tick_params(axis='y', colors='orange')

    # ticks issue
    locs, labels = plt.xticks()
    plt.xticks(np.arange(locs[0], locs[-1], step=0.2))

    ax.yaxis.label.set_color('orange')
    ax.xaxis.label.set_color('orange')
    ax.title.set_color('orange')

#     ax2 = fig.add_subplot(122)

#     ax2.set_xlabel('Number of iterations')
#     ax2.set_ylabel('Error')
#     ax2.set_title('Convergence')
#     ax2.plot(converge_rate, linewidth=4.0, color='red')
#     # ax.spines['bottom'].set_color('orange')
    # ax.spines['bottom'].set_linewidth(3.0)
    # ax.spines['top'].set_color('orange')
    # ax.spines['top'].set_linewidth(3.0)
    # ax.spines['right'].set_color('orange')
    # ax.spines['right'].set_linewidth(3.0)
    # ax.spines['left'].set_color('orange')
    # ax.spines['left'].set_linewidth(3.0)
    #
    # ax.tick_params(axis='x', colors='orange')
    # ax.tick_params(axis='y', colors='orange')
    #
    # ax.yaxis.label.set_color('orange')
    # ax.xaxis.label.set_color('orange')
    # ax.title.set_color('orange')

    plt.savefig('numpy_regression_points_norm.png', transparent=True)
    plt.show()


# main_linear_regression('area_price.csv')
