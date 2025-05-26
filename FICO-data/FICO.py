# -*- coding: utf-8 -*-
"""
Created on Wed May 18 0:38:01 2024

@author: Zhehao
"""

import pickle
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier

import logging
# from joblib import Parallel, delayed
import Experiment.FICO.causalml as cml


def generate_data(heloc_data, num_iv_levels, confounding_type='nucem', confounding_strength=0.5, random_seed=666):

    # Step 1: Factorize the binary-valued label and standardization
    data = heloc_data.copy()
    def standardization(df):
        return ( df - df.mean(axis=0) ) / df.std(axis=0)
        
    features = data.columns[1:].to_list()
    data[features] = standardization(data[features])
    data['RiskPerformance'] = pd.factorize(data['RiskPerformance'])[0]

    # Step 2: Specify the observed and unobserved features
    X = data[features]
    y = data['RiskPerformance']

    reg = HistGradientBoostingClassifier(max_iter=200)
    reg.fit(X, y)
    # print('Classification error: ', reg.score(X, y))

    U = y - pd.DataFrame(reg.predict_proba(X))[1]

    # Step 3: Generate instrumental variable Z
    Z = np.random.choice(range(1, num_iv_levels + 1), size=data.shape[0])

    # Step 4: Generate missing decision
    np.random.seed(random_seed)
    X1 = data['ExternalRiskEstimate']

    if confounding_type == 'nucem':
        D_prob = confounding_strength * expit(5 * U) + (1 - confounding_strength) * expit((1 + Z) * X1)
    elif confounding_type == 'uc':
        D_prob = expit(confounding_strength * (5 * U) + (1 - confounding_strength) * (1 + Z) * X1)
    else:
        raise ValueError("Invalid ztype. Must be 'nucem' or 'uc'.")
    D = np.random.binomial(1, D_prob)

    # Step 5: Return the aggregate data and parameters
    data = data.assign(U=U, Z=Z, D=D, DZ=D * Z, DY=D * y, DYZ=D * y * Z)
    data_params = {
        'observables': features,
        'instrument': 'Z',
        'decision': 'D',
        'outcome': 'RiskPerformance'
    }

    return data, data_params



def experiment(data, data_params):

    # Step 1: Initialize lists to store results
    acc_selected_df     = pd.DataFrame()
    acc_selected_ipw_df = pd.DataFrame()
    acc_full_df    = pd.DataFrame()
    acc_partial_df = pd.DataFrame()
    acc_point_df   = pd.DataFrame()
    acc_dr_df      = pd.DataFrame()


    for r_state in [100, 200, 300, 500, 600, 800]:

        # Step 2: Experiment with various methods
        acc_selected = cml.Classification(Sample=data, Params=data_params, Selective=True, IPW=False, k_folds=5, random_state=r_state)
        # print('Classification with Selective labels completes')

        acc_selected_ipw = cml.Classification(Sample=data, Params=data_params, Selective=True, IPW=True, k_folds=5, random_state=r_state)
        # print('Classification (IPW) with Selective labels completes')


        acc_full = cml.Classification(Sample=data, Params=data_params, Selective=False, IPW=False, k_folds=5, random_state=r_state)
        # print('Classification with Full labels completes')

        acc_partial = cml.WeightedClassification(Sample=data, Params=data_params, WeightFunc=cml.IVPartialWeightBinary, k_folds=5, random_state=r_state)
        # print('Partial Learning with Selective labels completes')

        acc_point = cml.WeightedClassification(Sample=data, Params=data_params, WeightFunc=cml.IVPointWeightBinary, k_folds=5, random_state=r_state)
        # print('Point Learning with Selective labels completes')

        acc_dr = cml.DRLearner(Sample=data, Params=data_params, k_folds=5, random_state=r_state)

        # Record the results
        acc_selected_df[r_state]     = acc_selected
        acc_selected_ipw_df[r_state] = acc_selected_ipw
        acc_full_df[r_state]         = acc_full
        acc_partial_df[r_state]      = acc_partial
        acc_point_df[r_state]        = acc_point
        acc_dr_df[r_state]           = acc_dr


    # Step 3: Aggregate results
    accuracy = pd.DataFrame(index=acc_full.keys())

    accuracy['Selected Sample']       = acc_selected_df.max(axis=1)
    accuracy['Selected Sample (IPW)'] = acc_selected_ipw_df.max(axis=1)
    accuracy['Full Sample']           = acc_full_df.max(axis=1)
    accuracy['Partial Learning']      = acc_partial_df.max(axis=1)
    accuracy['Point Learning']        = acc_point_df.max(axis=1)
    accuracy['DR Learner']            = acc_dr_df.max(axis=1)

    return accuracy
    


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    HelocData = pd.read_csv('HELOC/heloc_dataset_v1.csv')

    # Base parameters for generating data
    base_params = {
        'heloc_data': HelocData,
        'num_iv_levels': 5,
    }

    # Confounding parameters
    confounding_strength_values = [0.5, 0.7, 0.9]

    # Number of iterations for the experiment
    num_iterations = 100

    # Module for single iteration
    def run_single_iteration(iter, params):
        # Generate semi-synthetic data and run the experiment
        data, data_params = generate_data(**params, random_seed=68*iter)
        accuracy = experiment(data, data_params)
        return accuracy


    # Iterate over all combinations of confounding_strength_Y and confounding_strength_D
    for confounding_type in ['nucem', 'uc']:
        for confounding_strength in confounding_strength_values:

            # Add confounding strength to parameters collection
            params = base_params.copy()
            params['confounding_type'] = confounding_type
            params['confounding_strength'] = confounding_strength
                
            # Initialize containers    
            # data_batches = []
            accuracy_batches = []

            # Run iterations in parallel using joblib.Parallel
            # accuracy_batches = Parallel(n_jobs=num_iterations)(delayed(run_single_iteration)(iter, params) for iter in range(num_iterations))

            # Repeated experiments
            for iter in range(num_iterations):
                logging.info(f'Round {iter} with type={confounding_type} confounding_strength={confounding_strength} begins')

                # Generate data and run the experiment
                data, data_params = generate_data(**params, random_seed=68*iter)
                accuracy = experiment(data, data_params)

                accuracy_batches.append(accuracy)

            # Dump results to pickles
            # with open(f'FICO-Data-Iterations-{num_iterations}-Confounding-{confounding_strength}-UC.pickle', 'wb') as handle:
            #     pickle.dump(data_batches, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(f'FICO-Accuracy-Type-{confounding_type}-Strength-{confounding_strength}.pickle', 'wb') as handle:
                pickle.dump(accuracy_batches, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info('All experiments completed')

