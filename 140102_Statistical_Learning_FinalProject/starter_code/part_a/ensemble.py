import numpy as np

#####################################################################
# TODO:                                                             #                                                          
# Import packages you need here                                     #
from .utils import *
# I use IRT model as the base model
from .item_response import irt, sigmoid
import random
#####################################################################


#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################  




#####################################################################
# Define and implement functions here                               #
#####################################################################
np.random.seed(42)

# this function defined to compute accuracy from predictions of the 3 methods by voting
def compute_acc(train_data, val_data, train_pred_lst, val_pred_lst):
    val_pred_array = np.sum(np.array(val_pred_lst), axis=0)
    train_pred_array = np.sum(np.array(train_pred_lst), axis=0)
    # if summation value of methods that predict one is equal or more than 2,
    # then it means frequent label of the 3 predictions is one.
    val_pred_array = np.where(val_pred_array >= 2, 1, 0)
    train_pred_array = np.where(train_pred_array >= 2, 1, 0)
    # utils.evaluate to prevent mis_understanding between evaluate function in
    # item_response.py and utils.py
    return evaluate(train_data, train_pred_array, threshold=0.5),\
        evaluate(val_data, val_pred_array, threshold=0.5)


def bagging_ensemble(train_data, val_data, sparse_matrix, bagging_size = 3,
                     lr = 0.01, num_iteration = 50):
    # create 3 matrices as the output_matrix of each model(first value of each matrix is sparse_matrix)
    mat_lst = []
    for b in range(bagging_size):
        mat_lst.append(sparse_matrix)
    # zero initialization
    theta = np.zeros((542,))
    beta = np.zeros((1774,))

    val_pred_lst = []
    train_pred_lst = []
    for b in range(bagging_size):
        # select all of the train data with replacement
        size = len(train_data['is_correct'])
        train_lst = list(np.random.choice(size, size, replace = True))

        # select the train_lst index samples
        train_sample = {}
        train_sample['user_id'] = list(np.array(train_data['user_id'])[train_lst])
        train_sample['question_id'] = list(np.array(train_data['question_id'])[train_lst])
        train_sample['is_correct'] = list(np.array(train_data['is_correct'])[train_lst])
        learned_theta, learned_beta, _, _, _, _ = irt(train_sample, val_data, lr, num_iteration)
        for i in range(542):
            for j in range(1774):
                # fill the b_th output_matrix
                mat_lst[b][i][j] = sigmoid(learned_theta[i] - learned_beta[j])
        # now obtain validation prediction list with b_th sparse matrix
        train_pred = sparse_matrix_predictions(train_data, mat_lst[b], threshold=0.5)
        val_pred = sparse_matrix_predictions(val_data, mat_lst[b], threshold=0.5)
        train_pred_lst.append(train_pred)
        val_pred_lst.append(val_pred)

    train_acc, val_acc = compute_acc(train_data, val_data, train_pred_lst, val_pred_lst)
    return mat_lst, train_acc, val_acc

#####################################################################
#                       END OF YOUR CODE                            #
##################################################################### 




def ensemble_main():
    #####################################################################
    # Compute the finall validation and test accuracy                   #
    #####################################################################

    # define the train_matrix, train_data and val_data
    train_matrix = load_train_sparse("data").toarray()
    train_data = load_train_csv("data")
    val_data = load_valid_csv("data")
    # Best parameters of the IRT for lr and iteration
    lr = 0.01
    iteration = 100
    bagging_size = 3
    # it takes about a few minutes
    sp_mat, train_acc, val_acc = bagging_ensemble(train_data, val_data, train_matrix,
                     bagging_size = bagging_size, lr = lr, num_iteration = iteration)
    val_acc_ensemble:float = val_acc
    # test_acc_ensemble:float = None
    method1_output_matrix:np.array = sp_mat[0]
    method2_output_matrix:np.array = sp_mat[1]
    method3_output_matrix:np.array = sp_mat[2]

    print("The train accuracy of bagging_ensemble is: ", train_acc)
    print("The validation accuracy of bagging_ensemble is: ", val_acc)
    # iter_list = [i for i in range(iteration)]
    # plt.plot()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    results={
    'val_acc_ensemble':val_acc_ensemble,
    'test_acc_ensemble':None,    # change this variable to None
    'method1_output_matrix':method1_output_matrix,
    'method2_output_matrix':method2_output_matrix,
    'method3_output_matrix':method3_output_matrix
    }

    return results


if __name__ == "__main__":
    ensemble_main()

