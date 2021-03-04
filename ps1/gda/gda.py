import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)
    
    # *** START CODE HERE ***
    # Train a GDA classifier
    clf = GDA()
    clf.fit(x_train, y_train)
    preds = clf.predict(x_eval)

    # Plot decision boundary on validation set
    theta_ = np.insert(clf.theta, 0, clf.theta_zero)
    save_path_ = save_path.strip('.txt')
    util.plot(x_eval, y_eval, theta_, save_path_)

    # Use np.savetxt to save outputs from validation set to save_path
    np.savetxt(save_path, preds)
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        phi = y.mean()
        
        target_classes = np.unique(y)
        mean_vectors = []
        
        for cls in target_classes:
            mean_vectors.append(np.mean(x[y == cls], axis=0))
            
        mu_0 = mean_vectors[0]
        mu_1 = mean_vectors[1]
        
        arr = np.concatenate((x[y == 0] - mu_0, x[y == 1] - mu_1), axis = 0)
        sigma = (1/arr.shape[0])*np.dot(arr.T,arr)
        
        # Write theta in terms of the parameters
        sigma_inv = np.linalg.inv(sigma)
        mu0_sigma = np.dot(mu_0.T, np.dot(sigma_inv, mu_0))
        mu1_sigma = np.dot(mu_1.T, np.dot(sigma_inv, mu_1))
        self.theta = np.dot(sigma_inv, mu_1 - mu_0)
        self.theta_zero = 1/2*(mu0_sigma - mu1_sigma) + np.log(phi/(1-phi))

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        theta = self.theta.reshape((2,1))
        preds = 1/(1 + np.exp(-(np.dot(x, self.theta) + self.theta_zero)))
        return(preds)
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
