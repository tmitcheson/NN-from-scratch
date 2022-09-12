import random
import pickle
import numpy as np
import pandas as pd
import part1_nn_lib as nn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(
        self,
        x,
        nb_epoch=30000,
        neurons=[50, 50, 50, 1],
        activations=["relu", "relu",  "relu", "identity"],
        batch_size=100,
        dropout_rate=0.4,
        learning_rate=0.2,
        loss_fun="mse",
    ):
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.
            - neurons {list} -- number of neurons per layer
            - activations {list} -- activation functions to use at each layer
            - batch_size {int} -- size of batches in training
            - dropout_rate {float} -- fraction of neurons to dropout per epoch
                                      in training
            - learning_rate {float} -- initial leanring rate in training
            - loss_fun {string} -- functions for the calculation of loss in 
                                   training

        """

        self.input_size = x.shape[1] + 4
        self.output_size = 1
        self.x = x
        self.neurons = neurons
        self.activations = activations
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.net = nn.MultiLayerNetwork(self.input_size, neurons, activations)
        return


    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        # set up one hot maker
        if training:
            self._lb = LabelBinarizer()
            self._lb.fit(
                ["<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"])

        # fix text entries
        X = x.copy()  # Copy the dataframe
        X.fillna(random.uniform(0, 1), inplace=True)
        one_hots = self._lb.transform(
            X["ocean_proximity"])  # Form one-hot vectors
        X = X.drop(labels="ocean_proximity", axis=1)

        if training:
            self.min_X = X.min(skipna=True)
            self.max_X = X.max(skipna=True)

        # x normalisation
        X_norm = (X - self.min_X) / (self.max_X - self.min_X)
        X_numpy = X_norm.copy().to_numpy().astype(float)
        X_numpy = np.concatenate((X_numpy, one_hots), axis=1)

        # y normalisation
        Y_numpy = None
        if isinstance(y, pd.DataFrame):
            Y_numpy = y.copy().to_numpy().astype(float)
            if training:
                self.min_y = np.amin(Y_numpy)
                self.max_y = np.amax(Y_numpy)
            Y_numpy = (Y_numpy - self.min_y) / (self.max_y - self.min_y)

        return X_numpy, Y_numpy


    def fit(self, x_train, y_train, x_dev=None, y_dev=None):
        """
        Regressor training function

        Arguments:
            - x_train {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y_train {pd.DataFrame} -- Raw output array of shape 
                                        (batch_size, 1).
            - x_dev {pd.DataFrame} -- Raw development array of shape
                (batch_size, input_size).
            - y_dev {pd.DataFrame} -- Raw development array of shape 
                                      (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        X, Y = self._preprocessor(x_train, y=y_train, training=True)
        X_dev = None
        Y_dev = None
        if type(x_dev) != type(None) and type(y_dev) != type(None):
            X_dev, Y_dev = self._preprocessor(x_dev, y=y_dev, training=False)
        trainer = nn.Trainer(
            network=self.net,
            batch_size=self.batch_size,
            nb_epoch=self.nb_epoch,
            learning_rate=self.learning_rate,
            loss_fun=self.loss_fun,
            shuffle_flag=True,
            generate_plot_data=True,
        )
        trainer.train(X, Y, X_dev, Y_dev, self.min_y, self.max_y)
        return self


    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        X, _ = self._preprocessor(x, training=False)  # Do not forget
        return self.net(X, training=False) * (self.max_y - self.min_y) + self.min_y


    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        _, Y = self._preprocessor(x, y=y, training=False)  # Do not forget
        predictions = self.predict(x)
        return np.sqrt(mean_squared_error(y.to_numpy(), predictions))



def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open("part2_model.pickle", "wb") as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open("part2_model.pickle", "rb") as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(x, y):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
        - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

    Returns:
        {Regressor} -- Trained model with optimal parameters.

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # params to investigate
    x_in = [x]
    learning_rate = [0.2]
    neurons = [[5, 20, 20, 1],
               [50, 50, 50, 1], [150, 150, 150, 1]]
    batch_size = [50, 100, 250, 500]
    nb_epoch = [10, 50, 200, 500]
    dropout_rate = [0.0, 0.3, 0.4, 0.5]

    # model to test
    regressor = Regressor(x)

    # establish search grid
    grid = dict(
        x=x_in,
        neurons=neurons,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )

    grid_search = GridSearchCV(
        estimator=regressor,
        param_grid=grid,
        scoring=["neg_mean_squared_error"],
        verbose=4,
        error_score="raise",
        refit="neg_mean_squared_error"
    )

    result = grid_search.fit(x, y)
    with open("result.pickle", "wb") as target:
        pickle.dump(result, target)

    return result.best_estimator_
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():

    output_label = "median_house_value"

    data = pd.read_csv("housing.csv")

    # Spliting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    split_idx1 = int(0.6 * len(x))
    split_idx2 = int(0.8 * len(x))

    x_train = x.iloc[:split_idx1]
    y_train = y.iloc[:split_idx1]
    x_dev = x.iloc[split_idx1:split_idx2]
    y_dev = y.iloc[split_idx1:split_idx2]
    x_test = x.iloc[split_idx2:]
    y_test = y.iloc[split_idx2:]

    # Get best regressor
    regressor = RegressorHyperParameterSearch(x, y)
    save_regressor(regressor)

    # Loading
    regressor = load_regressor()

    # Error
    error = regressor.score(x, y)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
