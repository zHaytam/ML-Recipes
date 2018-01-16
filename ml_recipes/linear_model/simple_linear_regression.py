from ..ml_utils import mean, sd


class SimpleLinearRegression:
    """ y = mx + b """

    def __init__(self, slope=None, intercept=None):
        self.slope = slope
        self.intercept = intercept

    def fit(self, train_points, sample=True):
        """
        Calculates and sets the slope and the intercept for the simple linear regression equation
            :param self: ~
            :param train_points: A 2d numpy array containing the training points [x,y]
            :param sample: A boolean indicating if we're dealing with a sample
        """

        if not train_points.any():
            raise ValueError('Cannot work with an empty training data.')

        if train_points.ndim != 2:
            raise ValueError('The training data array must be a 2d ndarray.')

        train_points_t = train_points.T
        x_points = train_points_t[0]
        y_points = train_points_t[1]
        x_mean = mean(x_points)
        y_mean = mean(y_points)
        x_sd = sd(x_points, sample)
        y_sd = sd(y_points, sample)
        corr = self.__calculate_correlation(
            x_points, y_points, x_mean, y_mean, x_sd, y_sd)
        self.slope = (y_sd / x_sd) * corr
        self.intercept = y_mean - (self.slope * x_mean)

    def predict(self, x_value):
        """
        Returns a predicted value (y) of a x value using the simple linear regression equation
            :param self: ~
            :param x_value: The x value to feed to the equation
        """

        if self.slope is None or self.intercept is None:
            raise ValueError('No training data, call fit first.')

        return self.intercept + (self.slope * x_value)

    def __calculate_correlation(self, x_points, y_points, x_mean, y_mean, x_sd, y_sd):
        """
        Returns the correlation coefficient (R)
            :param self: ~
            :param x_points: A numpy array of all the x values
            :param y_points: A numpy array of all the y values
            :param x_mean: The mean of x_points
            :param y_mean: The mean of y_points
            :param x_sd: The (sample) standard deviation of x_points
            :param y_sd: The (sample) standard deviation of y_points
        """

        N = float(len(x_points))
        return sum((x_points - x_mean) * (y_points - y_mean)) / (x_sd * y_sd * N)
