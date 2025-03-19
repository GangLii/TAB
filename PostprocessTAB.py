from free_support_barycenters.bary import *

import numpy as np
import scipy as sp
import ot
from sklearn.metrics import roc_auc_score, accuracy_score



class ProcessorTAB():

  def generate_new_score_train(self, train_scores, train_sensitive_labels, seed):
    randommap = 1  # 1: Random; 0: Average
    NUM_LABELS = len(train_scores[0])
    supports = train_scores
    groupsize = train_sensitive_labels.value_counts()
    masses = [None] * len(groupsize)
    for i in range(len(groupsize)):
        group = groupsize.index[i]
        temp = np.zeros(np.sum(groupsize))
        temp[train_sensitive_labels == group] = 1.0 / groupsize.iloc[i]
        masses[i] = temp

    # Compute Barycenter
    pairwise_supp, pairwise_masses, err_bound = pairwise_bary(
        supports, masses, compute_err_bound=True
    )

    # Transportation Plan to Barycenter
    T = [None] * len(groupsize)
    for i in range(len(groupsize)):
        M = ot.dist(
            supports[train_sensitive_labels == groupsize.index[i]], pairwise_supp
        )
        a = [1 / groupsize.iloc[i]] * groupsize.iloc[i]
        b = pairwise_masses
        T[i] = ot.emd(a, b, M, numItermax=1_000_000)

    train_proba_temp = np.zeros_like(train_scores)
    np.random.seed(seed)
    for i in range(len(groupsize)):
        group = groupsize.index[i]
        group_indices = np.where(train_sensitive_labels == group)[0]
        
        for j in range(T[i].shape[0]):
            current_index = group_indices[j]
            
            if randommap == 1:
                index = np.random.choice(
                    np.nonzero(T[i][j])[0],
                    1,
                    p=T[i][j][np.nonzero(T[i][j])[0]] * T[i].shape[0],
                )
                train_proba_temp[current_index] = pairwise_supp[index]
            else:
                train_proba_temp[current_index] = np.matmul(
                    T[i][j] * T[i].shape[0], pairwise_supp
                )

    return train_proba_temp, T, supports, masses, pairwise_supp, groupsize

  

  def generate_new_score_test(self, test_scores, train_sensitive_labels, test_sensitive_labels, T, supports, pairwise_supp, groupsize, seed, sigma=0.1):
    randommap = 1  # 1: Random; 0: Average
    NUM_LABELS = len(test_scores[0])
    supports_test = test_scores
    groupsize_test = test_sensitive_labels.value_counts()
    masses_test = [None] * len(groupsize)
    for i in range(len(groupsize_test)):
        group = groupsize_test.index[i]
        temp = np.zeros(np.sum(groupsize_test))
        temp[test_sensitive_labels == group] = 1.0 / groupsize_test.iloc[i]
        masses_test[i] = temp

    test_proba_temp = np.zeros_like(test_scores)
    np.random.seed(seed)
    for i in range(len(groupsize)):
        group_test = groupsize_test.index[i]
        group = groupsize.index[i]
        
        group_indices = np.where(train_sensitive_labels == group)[0]
        group_test_indices = np.where(test_sensitive_labels == group_test)[0]
        
        M_test_to_train = ot.dist(
            supports_test[test_sensitive_labels == group_test, :],
            supports[train_sensitive_labels == group, :],
        )
        kernel_density_test = (
            np.exp(-M_test_to_train / (2 * sigma * sigma)) / sigma / np.sqrt(2 * np.pi)
        )
        kernel_density_test_sum = np.sum(kernel_density_test, 1)
        T_test = (
            np.matmul(kernel_density_test, T[i])
            / kernel_density_test_sum[:, None]
            * groupsize.iloc[i]
        )
        for j in range(groupsize_test.iloc[i]):
            current_index = group_test_indices[j]
            if randommap == 1:
                index = np.random.choice(range(supports.shape[0]), 1, p=T_test[j, :])
                test_proba_temp[current_index] = pairwise_supp[index]
            else:
                test_proba_temp[current_index] = np.matmul(T_test[j, :], pairwise_supp)

    return test_proba_temp, supports_test, masses_test


  def evaluate_performance(self, supports, masses, new_scores, sensitive_labels, y, class_list=None, alpha_list = np.linspace(0, 1, 6), task_type='multi_class'):
    """
    task_type: 'multi_class' or 'multi_label'
    """

    proba_temp = new_scores
    if task_type == 'multi_class':
        accuracy_list = np.zeros((len(alpha_list), 1))
    elif task_type == 'multi_label':
       accuracy_list = np.zeros((len(alpha_list), supports.shape[1]))
    unfairness_list = np.zeros((len(alpha_list), supports.shape[1]))

    proba_new = supports * 0
    alpha_index = 0
    groupsize = sensitive_labels.value_counts()
    wasserstein_list = np.zeros((len(alpha_list), len(groupsize)))
    
    for alpha in alpha_list:
        for i in range(len(groupsize)):
            group = groupsize.index[i]
            proba_new[sensitive_labels == group] = (1 - alpha) * proba_temp[sensitive_labels == group] + alpha * supports[sensitive_labels == group]
        
        if task_type == 'multi_class':
            y_pred = np.array([class_list[s] for s in np.argmax(proba_new, axis=1)])
            accuracy_list[alpha_index, 0] = accuracy_score(y, y_pred)
            for i in range(supports.shape[1]):
                # auc_list[index, i] = roc_auc_score(np.where(y == classes[i], 1, 0), proba_new[:, i])
                temp = np.zeros(len(groupsize))
                for j in range(len(groupsize)):
                    group = groupsize.index[j]
                    temp[j] = np.sum(np.where(y_pred[sensitive_labels == group] == class_list[i], 1, 0)) / groupsize.iloc[j]
                unfairness_list[alpha_index, i] = np.max(temp) - np.min(temp)
        
        elif task_type == 'multi_label':
            y_pred = np.where(proba_new >= 0.5, 1, 0)
            for i in range(supports.shape[1]):
                accuracy_list[alpha_index, i] = accuracy_score(y[:, i], y_pred[:, i])
                # auc_list[index, i] = roc_auc_score(y[:, i], proba_new[:, i])
                temp = np.zeros(len(groupsize))
                for j in range(len(groupsize)):
                    group = groupsize.index[j]
                    temp[j] = np.sum(y_pred[sensitive_labels == group, i]) / groupsize.iloc[j]
                unfairness_list[alpha_index, i] = np.max(temp) - np.min(temp)
        print ("alpha", alpha)
        print ("accuracy_score", np.mean(accuracy_list[alpha_index]))

        pairwise_supp_temp, pairwise_masses_temp = ref_bary(proba_new, masses)
        temp = np.zeros(len(groupsize))
        for i in range(len(groupsize)):
            group = groupsize.index[i]
            M = ot.dist(proba_new[sensitive_labels == group, :], pairwise_supp_temp)
            a = [1 / groupsize.iloc[i]] * groupsize.iloc[i]
            b = pairwise_masses_temp
            temp[i] = ot.emd2(a, b, M)
            wasserstein_list[alpha_index, i] = temp[i]
        
        alpha_index += 1
    
    return accuracy_list, unfairness_list, wasserstein_list
  
  def display_graphs(self, accuracy_list, unfairness_list, wasserstein_list, line_type, groupsize, alpha_list = np.linspace(0, 1, 6)):
    
    unfairness=np.zeros(len(alpha_list))
    for i in range(len(alpha_list)):
        unfairness[i]=np.max(unfairness_list[i,:])

    accuracy=np.zeros(len(alpha_list))
    for i in range(len(alpha_list)):
        accuracy[i]=np.mean(accuracy_list[i,:])

    weights = np.array([s/np.sum(groupsize) for s in groupsize])
    wasserstein=np.zeros(len(alpha_list))
    for i in range(len(alpha_list)):
      wasserstein[i]=np.inner(wasserstein_list[i,:], weights)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].errorbar(unfairness, accuracy, label = line_type, linewidth=3.0)
    axes[0].set_xlabel("Unfairness on Predicted Classes")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].tick_params(axis='both', which='major', labelsize=15)


    axes[1].errorbar(wasserstein, accuracy, label = line_type, linewidth=3.0)
    axes[1].set_xlabel("Unfairness on Wasserstein Distance to Barycenter")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].tick_params(axis='both', which='major', labelsize=15)


    plt.subplots_adjust(wspace=0.2)

    plt.show()
  

  def display_graphs_seed(self, accuracy_list_seed, unfairness_list_seed, wasserstein_list_seed, line_type, groupsize, alpha_list = np.linspace(0, 1, 6)):
    

    temp=np.zeros((len(alpha_list),len(accuracy_list_seed)))
    for i in range(len(alpha_list)):
      for j in range(len(accuracy_list_seed)):
        temp[i,j]=np.max(unfairness_list_seed[j][i,:])
    unfairness_error=sp.stats.sem(temp,axis=1)
    unfairness_mean=np.mean(temp,axis=1)

    temp=np.zeros((len(alpha_list),len(accuracy_list_seed)))
    for i in range(len(alpha_list)):
      for j in range(len(accuracy_list_seed)):
        temp[i,j]=np.mean(accuracy_list_seed[j][i,:])
    accuracy_error=sp.stats.sem(temp,axis=1)
    accuracy_mean=np.mean(temp,axis=1)

    weights = np.array([s/np.sum(groupsize) for s in groupsize])
    temp=np.zeros((len(alpha_list),len(accuracy_list_seed)))
    for i in range(len(alpha_list)):
      for j in range(len(accuracy_list_seed)):
        temp[i,j]=np.inner(wasserstein_list_seed[j][i,:], weights)
    wasserstein_error=sp.stats.sem(temp,axis=1)
    wasserstein_mean=np.mean(temp,axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].errorbar(unfairness_mean, accuracy_mean, label = line_type, xerr=unfairness_error, yerr=accuracy_error, linewidth=3.0)
    axes[0].set_xlabel("Unfairness on Predicted Classes")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].tick_params(axis='both', which='major', labelsize=15)

    axes[1].errorbar(wasserstein_mean, accuracy_mean, label = line_type, xerr=wasserstein_error, yerr=accuracy_error, linewidth=3.0)
    axes[1].set_xlabel("Unfairness on Wasserstein Distance to Barycenter")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].tick_params(axis='both', which='major', labelsize=15)

    plt.subplots_adjust(wspace=0.2)

    plt.show()