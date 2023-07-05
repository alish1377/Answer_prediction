from sklearn.impute import KNNImputer
from .utils import *  # I change the .utils to utils
#####################################################################
# TODO:                                                             #
# Import packages you need here                                     #
#####################################################################
import matplotlib.pyplot as plt
#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################  

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    completed_mat = KNNImputer(n_neighbors=k).fit_transform(matrix)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################    
    acc = sparse_matrix_evaluate(valid_data, completed_mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    acc = sparse_matrix_evaluate(valid_data, (KNNImputer(n_neighbors=k).fit_transform(matrix.T)).T)
    print("Validation Accuracy: {}".format(acc))  # add this print line to determine the item results
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def knn_main():
    sparse_matrix = load_train_sparse("data").toarray()
    val_data = load_valid_csv("data")
    # I change this test_data loading to try_except form, because test data is not available
    try:
        test_data = load_public_test_csv("data")
    except:
        pass

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # Part B&C:                                                         #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*. do all these using knn_impute_by_user().                                                       #
    #####################################################################

    user_best_k:float = None                    # :float means that this variable should be a float number
    user_test_acc:float = None   # except test acc, I save the best acc result on val_data in this variable
    user_valid_acc:list = []

    user_best_acc = 0
    print("Impute by user..")   # add this print line to split the users results
    k_list = [1,6,11,16,21,26] # k_list as the first axis of plot function
    for k in k_list:
        # apply model for different k
        acc = knn_impute_by_user(sparse_matrix, val_data, k = k)
        user_valid_acc.append(acc)
        # save best acc and equivalent k
        if acc > user_best_acc:
            user_best_acc = acc
            user_best_k = k

    print("Best accuracy by user..")
    user_test_acc = knn_impute_by_user(sparse_matrix, val_data, k = user_best_k)
    plt.plot(k_list, user_valid_acc, color = 'red', marker = 'o')
    plt.title('KNN impute by user')
    plt.xlabel('k')
    plt.ylabel('validation acc')
    plt.savefig('plots/knn/knn_impute_by_user.png')
    plt.close()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # Part D:                                                           #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*. do all these using knn_impute_by_item().                                                        #
    #####################################################################
    question_best_k:float = None
    question_test_acc:float = None   # except test acc, I save the best acc result on val data in this variable
    question_valid_acc:list = []

    question_best_acc = 0
    print("Impute by item..")   # add this print line to split the items results from users
    for k in k_list:
        # Apply model for different k
        acc = knn_impute_by_item(sparse_matrix, val_data, k = k)
        question_valid_acc.append(acc)
        # Save best acc and equivalent k
        if acc > question_best_acc:
            question_best_acc = acc
            question_best_k = k

    print("Best accuracy by item..")
    question_test_acc = knn_impute_by_item(sparse_matrix, val_data, k = question_best_k)
    plt.plot(k_list, question_valid_acc, color = 'red', marker = 'o')
    plt.title('KNN impute by question')
    plt.xlabel('k')
    plt.ylabel('validation acc')
    plt.savefig('plots/knn/knn_impute_by_item.png')
    plt.close()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    results = {
    'user_best_k':user_best_k,
    'user_test_acc':user_test_acc,
    'user_valid_accs':user_valid_acc,
    'question_best_k':question_best_k,
    'question_test_acc':question_test_acc,
    'question_valid_acc':question_valid_acc,
    }
    
    
    return results

if __name__ == "__main__":
    knn_main()
