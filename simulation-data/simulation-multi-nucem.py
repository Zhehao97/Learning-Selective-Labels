# -*- coding: utf-8 -*-
"""
Created on Wed May 18 0:38:01 2024

@author: Zhehao
"""

import pickle
import numpy as np
import pandas as pd

import logging
import causalml as cml


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def generate_data(n_samples, n_observed, n_unobserved, num_iv_levels, num_y_classes, confounding_type, confounding_strength_Y, confounding_strength_D, random_seed=100):

    np.random.seed(random_seed)

    # Step 1: Generate observed and unobserved variables X, U
    X = 2 * np.random.normal(size=(n_samples, n_observed))
    U = 2 * np.random.normal(size=(n_samples, n_unobserved))
    
    # Step 2: Generate instrumental variable Z
    Z = np.random.choice(range(1, num_iv_levels + 1), size=n_samples)

    # Step 3: Generate the coefficients of X on Y and U on Y 
    coefficients_X_Y = np.array([[i + j for j in range(num_y_classes)] for i in range(n_observed)])  # Fixed coefficients for Y on X
    coefficients_U_Y = np.array([[i + j for j in range(num_y_classes)] for i in range(n_unobserved)])  # Fixed coefficients for Y on U

    # Step 4: Generate the outcome Y via conditioinal probabilities 
    linear_combination = (1 - confounding_strength_Y) * X @ coefficients_X_Y + confounding_strength_Y * 2 * (U @ coefficients_U_Y)
    probabilities = softmax(linear_combination)
    Y = np.array([np.random.choice(range(1, num_y_classes + 1), p=p) for p in probabilities])

    # Step 5: Generate decision D as a function of X, U, and Z
    coefficients_X_D = np.array([2-j for j in range(n_observed)])
    coefficients_U_D = np.array([1+j for j in range(n_unobserved)])
    
    # Step 6: Directly generate probabilities for each action (sigmoid)
    logits_1 = 2 * Z * (X @ coefficients_X_D)
    logits_2 = (U @ coefficients_U_D)
    if confounding_type == 'nucem':
        D_prob = (1 - confounding_strength_D) * sigmoid(logits_1) + confounding_strength_D * sigmoid(logits_2)
    elif confounding_type == 'uc':
        D_prob = sigmoid( (1 - confounding_strength_D) * logits_1 + confounding_strength_D * logits_2 )
    else:
        raise ValueError("Invalid ztype. Must be 'nucem' or 'uc'.")
    D = np.random.binomial(1, D_prob)

    # Step 7: Aggregate data with augmented variables
    df1 = pd.DataFrame(X, columns=[f'X{i}' for i in range(n_observed)])
    df2 = pd.DataFrame(U, columns=[f'U{i}' for i in range(n_unobserved)])
    data = pd.concat([df1, df2], axis=1).assign(Z=Z, Y=Y, D=D)

    # data['DZ'] = data['D'] * data['Z']

    # # Generate dummy variables for Y
    # dummies_Y = pd.get_dummies(data['Y'], dtype=int)

    # # Add DY and DZY variables
    # for k in dummies_Y.columns:
    #     data[f'DY{k}'] = data['D'] * dummies_Y[k]
    #     data[f'DZY{k}'] = data['DZ'] * dummies_Y[k]

    return data


def experiment(data):

    # Step 1: Initialize parameters from data
    parameters = dict()
    parameters['outcome'] = 'Y'
    parameters['decision'] = 'D'
    parameters['instrument'] = 'Z'
    parameters['observables'] = data.filter(like='X').columns.tolist()
    parameters['unobservables'] = data.filter(like='U').columns.tolist()

    # Step 2: Initialize lists to store results
    acc_selected_lst = []
    acc_selected_ipw_lst = []
    acc_full_lst = []
    acc_partial_lst = []
    acc_point_lst = []

    for r_state in [100, 200, 300, 400, 500, 600]:

        # Step 3: Experiment with various methods
        acc_selected = cml.ClassificationNN(Sample=data, Params=parameters, 
                                            Selective=True, IPW=False, 
                                            random_state=r_state)
        acc_selected_lst.append( acc_selected['accuracy'] )

        acc_selected_ipw = cml.ClassificationNN(Sample=data, Params=parameters, 
                                                Selective=True, IPW=True, 
                                                random_state=r_state)
        acc_selected_ipw_lst.append( acc_selected_ipw['accuracy'] )

        acc_full = cml.ClassificationNN(Sample=data, Params=parameters, 
                                        Selective=False, IPW=False, 
                                        random_state=r_state)
        acc_full_lst.append( acc_full['accuracy'] )

        acc_partial = cml.CostSensitiveClassificationNN(Sample=data, 
                                                        Params=parameters, 
                                                        WeightFunc=cml.IVPartialWeightMulti,
                                                        surrogate='exploss', 
                                                        k_folds=5, random_state=r_state)
        acc_partial_lst.append( acc_partial['accuracy'] )
        
        acc_point = cml.CostSensitiveClassificationNN(Sample=data, 
                                                        Params=parameters, 
                                                        WeightFunc=cml.IVPointWeightMulti, 
                                                        surrogate='exploss',
                                                        k_folds=5, random_state=r_state)
        acc_point_lst.append( acc_point['accuracy'] )
        

    # Step 4: Aggregate results
    accuracy = dict()

    accuracy['Selected Sample'] = np.max( acc_selected_lst )
    accuracy['Selected Sample (IPW)'] = np.max( acc_selected_ipw_lst )
    accuracy['Full Sample'] = np.max( acc_full_lst )
    accuracy['Partial Learning'] = np.max( acc_partial_lst )
    accuracy['Point Learning'] = np.max( acc_point_lst )

    return accuracy
    


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Confounding parameters
    confounding_strength_values = [0.3, 0.5, 0.7, 0.9]

    # Number of iterations for the experiment
    num_iterations = 100

    # Iterate over all combinations of confounding_strength_Y and confounding_strength_D
    for confounding_strength_Y in confounding_strength_values:
        for confounding_strength_D in confounding_strength_values:
            
            # Record the results for each combination of confounding strengths
            data_batches = []
            accuracy_batches = []

            # Repeated experiments
            for iter in range(num_iterations):
                logging.info(f'Round {iter} with confounding_strength_Y={confounding_strength_Y} and confounding_strength_D={confounding_strength_D} begins')

                # Generate data and run the experiment
                # data = generate_data(n_samples=10000, num_iv_levels=5, 
                #                      confounding_type='nucem', 
                #                      confounding_strength_D=confounding_strength_D, 
                #                      confounding_strength_Y=confounding_strength_Y, 
                #                      random_seed=20*iter)
                
                data = generate_data(n_samples=10000, n_observed=6, n_unobserved=4, num_iv_levels=5, num_y_classes=3, 
                                     confounding_type='nucem', confounding_strength_Y=confounding_strength_Y, confounding_strength_D=confounding_strength_D, 
                                     random_seed=20*iter)

                accuracy = experiment(data)

                # Aggregate data and simulation result
                # data_batches.append(data)
                accuracy_batches.append(accuracy)

            # Dump results to pickles
            # with open(f'NUCEM-Data-Iterations-{num_iterations}-Confounding-Y{confounding_strength_Y}-D{confounding_strength_D}.pickle', 'wb') as handle:
            #     pickle.dump(data_batches, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(f'NUCEM-Accuracy-Iterations-{num_iterations}-Confounding-Y{confounding_strength_Y}-D{confounding_strength_D}.pickle', 'wb') as handle:
                pickle.dump(accuracy_batches, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info('All experiments completed')

