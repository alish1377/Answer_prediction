from .utils import *
from scipy.linalg import sqrtm

import numpy as np


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.

    # I implement 3 initial methods
    # first : fill NaN value with 0.5
    matrix[np.isnan(matrix)] = 0.5

    # second : fill Nan with mean of existed questions of the specific user
    # for c in range(matrix.shape[1]):
    #     matrix[:,c][np.isnan(matrix[:,c])] = np.nansum(matrix[:,c]) / \
    #                                          (matrix.shape[1]- np.sum(np.isnan(matrix[:,c])))

    # third : fill Nan with mean of existed user of the specific question
    # for r in range(matrix.shape[0]):
    #     matrix[r,:][np.isnan(matrix[r,:])] = np.nansum(matrix[r,:]) / \
    #                                          (matrix.shape[0]- np.sum(np.isnan(matrix[r,:])))
    #####################################################################
    # TODO:                                                             #
    # Part A:                                                           #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # apply svd
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)
    S_diag = np.diag(S)
    reconst_matrix = U[:,:k] @ S_diag[:k, :k] @ VT[:k, :]
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Part C:                                                           #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # we want update row <n> of u and column <m> of z

    # derivative formula to update u and z
    u[n,:] = u[n,:] + lr * ((c - (u[n,:] @ z[q,:].T)) * z[q,:])
    z[q,:] = z[q,:] + lr * ((c - (u[n,:] @ z[q,:].T)) * u[n,:])

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Part C:                                                           #
    # Implement the function as described in the docstring.             #
    #####################################################################


    u_arr, z_arr = np.zeros((10,u.shape[0],u.shape[1])), \
                    np.zeros((10,z.shape[0],z.shape[1]))
    for i in range(num_iteration):
        if i % 50000 ==0:
            idx = int(i / 50000)
            u_arr[idx] = u
            z_arr[idx] = z
        u,z = update_u_z(train_data, lr, u, z)
    mat = u @ z.T
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    #
    return mat, u, z, list(u_arr), list(z_arr)


def matrix_factorization_main():
    train_matrix = load_train_sparse("data").toarray()
    train_data = load_train_csv("data")
    val_data = load_valid_csv("data")
    try:
        test_data = load_public_test_csv("data")
    except:
        pass

    #####################################################################
    # TODO:                                                             #
    # Part A:                                                           #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    np.random.seed(42)
    # try 7 different k
    k_list = [1,5,10,50,100,200,500]
    best_k_svd:int=0
    best_val_acc_svd:float=0
    for k in k_list:
        recons_mat = svd_reconstruct(train_matrix, k)
        valid_acc = sparse_matrix_evaluate(val_data, recons_mat, threshold=0.5)
        print(f"Accuracy value by SVD method with k={k} is :{valid_acc}")
        if valid_acc > best_val_acc_svd:
            best_val_acc_svd = valid_acc
            best_k_svd = k
    print(f"Best accuracy value by SVD method with k={best_k_svd} is :{best_val_acc_svd}")
    #test_acc_svd:float=None
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Part D and E:                                                     #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # Results of part D


    best_k_als:int=0
    best_val_acc_als:float=0

    num_iteration = 500000
    lr = 0.02
    best_recons_mat = 0
    best_u, best_z = 0, 0

    # save best u_list and z_list based on the val_accuracy
    temp_u_list, temp_z_list = [], []
    best_u_list, best_z_list = [], []
    best_val_loss = 10000000  # select high value for initial loss to ensure that the first loss will be saved in this variable
    for k in k_list:
        recons_mat, u, z, temp_u_list, temp_z_list = als(train_data, k, lr, num_iteration)

        val_acc = sparse_matrix_evaluate(val_data, recons_mat, threshold=0.5)
        val_loss = squared_error_loss(val_data, u, z)
        print(f"Accuracy value by ALS method with k = {k} is: {val_acc}")
        print(f"Loss value by ALS method with k = {k} is {val_loss}")
        # select best model based on the minimum loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc_als = val_acc
            best_k_als = k
            # best reconstruction matrix
            best_recons_mat = recons_mat
            # save temporary list in best_lists for plotting purpose
            best_u_list = temp_u_list
            best_z_list = temp_z_list

            # best u and z based on the best val_acc according to different k
            best_u, best_z = u, z
        # Important note : The ALS update just one train data in each iteration, so after each iteration, the difference
        # is not considerable. so,for plotting purpose, I consider each point iteration as the point where main iteration
        # reaches the 1/10 num_iteration value, for example in my model the num_iteration is 50,000, so after 5,000 iteration update
        # , I add one point to plot list.

    train_loss_lst, val_loss_lst = [], []
    plt_list = [50000 * i for i in range(10)]
    for i in range(10):  # x_points, 10 points with 5,000 intervals  y_point : the loss in each 5,000 points iteration
        train_loss_lst.append(squared_error_loss(train_data, best_u_list[i], best_z_list[i]))
        val_loss_lst.append(squared_error_loss(val_data, best_u_list[i], best_z_list[i]))


    print(f"Best accuracy by ALS method with k={best_k_als} is :{best_val_acc_als}")
    print('Best reconstructed matrix in term of validation loss:...')
    print(best_recons_mat)
    print('The best u...')
    print(best_u)
    print('The best_z...')
    print(best_z)



    # Results of part E
    # Save the line chart

    import matplotlib.pyplot as plt
    plt.plot(plt_list, train_loss_lst, marker = 'o', color = 'red')
    plt.plot(plt_list, val_loss_lst, marker = '*', color = 'green')
    plt.title('validation and train losses based on the iteration')
    plt.xlabel('iteration')
    plt.ylabel('loss value')
    plt.legend(['train_loss', 'val_loss'])
    plt.savefig('plots/matrix_factorization/part_e.png')
    plt.close()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    results={
    'best_k_svd':best_k_svd,
    'test_acc_svd':None,    # change this variable to None
    'best_val_acc_svd':best_val_acc_svd,
    'best_val_acc_als':best_val_acc_als,
    'best_k_als':best_k_als

    }

    return results

if __name__ == "__main__":
    matrix_factorization_main()
