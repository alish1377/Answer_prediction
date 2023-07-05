from .utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    # number of questions is equal to number of students is equal to length of data['is_correct']
    for e,label in enumerate(data['is_correct']):
        c_ij = label
        theta_i = theta[data['user_id'][e]]
        beta_j = beta[data['question_id'][e]]
        # compute likelihood based on the computation
        log_lklihood += c_ij * (theta_i - beta_j) - np.log(1 / (1-sigmoid(theta_i - beta_j)))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    # copies variable from theta and beta to save summation
    copy_theta = np.zeros((542,))
    copy_beta = np.zeros((1774,))
    for e in range(len(data['is_correct'])):
        # update copy variables based on the computation results
        label = data['is_correct'][e]
        copy_theta[data['user_id'][e]] += label - sigmoid(theta[data['user_id'][e]] - beta[data['question_id'][e]])
        copy_beta[data['question_id'][e]] += -label + sigmoid(theta[data['user_id'][e]] - beta[data['question_id'][e]])
    # update theta and beta by copy variables
    theta = theta + lr * copy_theta
    beta = beta + lr * copy_beta


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    # zero initialization
    # theta = np.zeros((542,))
    # beta = np.zeros((1774,))
    theta = np.zeros((542,))
    beta = np.zeros((1774,))

    val_acc_lst = []
    neg_lld_lst = []
    # I add these 2 list variables for train set
    train_acc_lst = []
    val_neg_lld_lst = []

    for i in range(iterations):
    #####################################################################
    # TODO:Complete the code                                            #
    #####################################################################
    # append likelihood and accuracy values
        neg_lld_lst.append(neg_log_likelihood(data, theta, beta))
        val_neg_lld_lst.append(neg_log_likelihood(val_data, theta, beta))
        val_acc_lst.append(evaluate(val_data, theta, beta))
        train_acc_lst.append(evaluate(data, theta, beta))
        theta, beta = update_theta_beta(data, lr, theta, beta)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################       
    
    # TODO: You may change the return values to achieve what you want.
    # add train_acc_lst for train accuracy, val_neg_lld_lst for validation log_likelihood
    return theta, beta, val_acc_lst, train_acc_lst, neg_lld_lst, val_neg_lld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def item_response_main():
    train_data = load_train_csv("data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("data")
    val_data = load_valid_csv("data")
    try:
        test_data = load_public_test_csv("data")
    except:
        pass

    num_iterations = 100
    lr = 0.001

    #####################################################################
    # Part B:                                                           #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    # -> Important Note: save plots instead of showing them!            #
    #####################################################################
    learned_theta, learned_beta, val_acc_list, train_acc_list, neg_lld_lst, val_neg_lld_lst = \
        irt(train_data, val_data, lr, num_iterations)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # Part C:                                                           #
    # Best Results                                                      #
    # -> Important Note: save plots instead of showing them!            #
    #####################################################################
    final_validation_acc = max(val_acc_list)
    final_neg_lld_value = max(val_neg_lld_lst)  # we want lld instad of neg_lld
    print("Best accuracy of validation in IRT method :", final_validation_acc)
    print("Best neg_lld of validation in IRT method : ", final_neg_lld_value)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # Part D:                                                           #
    # Plots                                                             #
    # -> Important Note: save plots instead of showing them!            #
    #####################################################################
    # first plot : save train and validation accuracies based on the iteration
    iter_list = [i for i in range(num_iterations)]

    plt.plot(iter_list, train_acc_list, marker = 'o', color = 'red')
    plt.plot(iter_list, val_acc_list, marker = '*', color = 'green')
    plt.title('validation and train accuracies based on the iteration')
    plt.xlabel('iteration')
    plt.ylabel('acc value')
    plt.legend(['train_acc', 'val_acc'])
    plt.savefig(f'plots/IRT/lr={lr},iteration={num_iterations},accuracies_plots.png')
    plt.close()

    # because we want log_likelihood plot, I use negative values of neg_lld_list as the lld_list
    plt.plot(iter_list, neg_lld_lst, marker = 'o', color = 'orange')
    plt.plot(iter_list, val_neg_lld_lst, marker = '*', color = 'blue')
    plt.title('validation and train neg_log_likelihood values based on the iteration')
    plt.xlabel('iteration')
    plt.ylabel('neg_lld value')
    plt.legend(['neg_train_lld', 'neg_val_lld'])
    plt.savefig(f'plots/IRT/lr={lr},iteration={num_iterations},neg_logLikelihood_plots.png')
    plt.close()

    # test 5 arbitrary questions
    j_list = [22,333,654,1276,1700]
    color = ['red', 'orange', 'cyan', 'yellow', 'blue']
    # sort learned_theta to prevent distortion in plot figure
    sort_theta = np.sort(learned_theta)
    for i,q in enumerate(j_list):
        # The truth probability of the question
        pc = sigmoid(sort_theta - learned_beta[q])
        plt.plot(sort_theta, pc, color = color[i], marker = '*', label = f'Question {j_list[i]} with beta = {np.round(learned_beta[q],2)}')
    plt.legend()
    plt.title('The truth probability of five questions')
    plt.xlabel('theta')
    plt.ylabel('probability')
    plt.savefig('plots/IRT/The plot of the truth probability of five question.png')
    plt.close()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


    results = {
        'lr':lr,
        'num_iterations':num_iterations,
        'theta':learned_theta,
        'beta':learned_beta,
        'val_acc_list':val_acc_list,
        'neg_lld_lst':neg_lld_lst,
        'final_validation_acc':final_validation_acc,
        }
    return results

if __name__ == "__main__":
    item_response_main()
