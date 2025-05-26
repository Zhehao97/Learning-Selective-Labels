# Unified Cost-sensitive Learning
import numpy as np
import pandas as pd

import random
import torch
import nnmodel 
import torch.nn as nn
import torch.optim as optim

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

import warnings
warnings.filterwarnings('ignore')


PARAMS_LOGISTIC = {'C': [0.1, 1, 10], 'max_iter': [500]}
PARAMS_SVM = {'C': [0.1, 1, 10], 'kernel': ['sigmoid']}
PARAMS_RFOREST = {'max_depth':[2, 3, 5], 'n_estimators': [300]}
PARAMS_ABOOST = {'n_estimators': [100, 300], 'learning_rate': [0.01]}
PARAMS_HGBOOST = {'learning_rate': [0.01, 0.1], 'max_iter': [300]}

DEVICE = torch.device("mps")


def set_random_seed(seed=0):
    """
    Set random seeds for reproducibility in CPU-only environments.
    """
    # Python random
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False



# --------------------------------------------------------------------------------
# I. Nuisances Estimation
# --------------------------------------------------------------------------------

# Estimation module for nuisance parameters
def NuisanceEstimation(Sample, features, label, instrument=None, ztype='weighted'):
    """
    Inputs:
        - Sample: pd.DataFrame
        - features: list of str
        - label: str 
        - instrument: str (optional)
        - ztype: str
        - n_jobs: int, number of jobs to run in parallel (default: -1 for all cores)

    Output: 
        sklearn.model or dict of sklearn.models
    """ 
    # Prepare the sample for predicition
    X = Sample[features].values
    Y = Sample[label].values.reshape(-1)

    # Select the correct mode for prediction
    if ztype == 'weighted':

        # Fitting
        # model = LogisticRegression()
        model = HistGradientBoostingClassifier(max_iter=500)
        model.fit(X, Y)
        # gbm = HistGradientBoostingClassifier(max_iter=500, random_state=42)
        # calibrated_model = CalibratedClassifierCV(gbm, method='sigmoid')
        # calibrated_model.fit(X, Y)

        # print(model.score(X, Y))

        return model

    elif ztype == 'pointwise':
        if instrument is None:
            raise ValueError("Instrument must be provided for pointwise ztype")

        # Unique value of instruments
        z_vals = np.unique(Sample[instrument]).astype(int)
        
        # Initiate a containter for the estimator of each possible z
        models = dict() 
        # calibrated_models = dict()

        # Fit estimators for each possible z
        for z in z_vals:

            # Covariates and labels
            X = np.array( Sample[Sample[instrument] == z][features] )
            Y = np.array( Sample[Sample[instrument] == z][label]    ).reshape(-1)

            # Fitting
            # model = LogisticRegression()
            model = HistGradientBoostingClassifier(max_iter=500)
            model.fit(X, Y)
            # gbm = HistGradientBoostingClassifier(random_state=42)
            # calibrated_model = CalibratedClassifierCV(gbm, method='sigmoid')
            # calibrated_model.fit(X, Y)

            # Adding the estimator to the container
            models[f'z={z}'] = model
            # calibrated_models[f'z={z}'] = calibrated_model
            
        return models

    elif ztype == 'robust':
        if instrument is None:
            raise ValueError("Instrument must be provided for robust ztype")

        # Incoporate the insturmental variables
        features_z = features + [instrument]
        Xz = Sample[features_z].values
        
        # Fitting
        # model = LogisticRegression()
        model = HistGradientBoostingClassifier(max_iter=500)
        model.fit(Xz, Y)
        # gbm = HistGradientBoostingClassifier(random_state=42)
        # calibrated_model = CalibratedClassifierCV(gbm, method='sigmoid')
        # calibrated_model.fit(Xz, Y)

        # print(model.score(Xz, Y))

        return model

    else:
        raise ValueError("Invalid ztype. Must be 'weighted', 'pointwise', or 'robust'.")



# --------------------------------------------------------------------------------
# II. Weight Function Estimation
# --------------------------------------------------------------------------------


# Compute the inverse propensity scores
def InversePropensityWeight(Sample_1, Sample_2, Params):
    """
    Inputs:
        - Sample_1: pd.DataFrame
        - Sample_2: pd.DataFrame
        - Params: dictionary

    Outputs:
        pd.DataFrame
    """
    # Specify the parameters
    features = Params['observables']
    decision = Params['decision']
    outcome  = Params['outcome']

    # Regress on the propensity score P(D = 1 | X = x)
    Model_D = NuisanceEstimation(Sample=Sample_1, features=features, label=decision, instrument=None, ztype='weighted')

    # Estimate the inverse propensity weight function given X = x of Sample 2
    X = Sample_2[features].values
    W = 1 / Model_D.predict_proba(X)[:, 1]
    
    # Construct new sample
    Sample_IPW = Sample_2.copy()
    Sample_IPW['W'] = W

    # Cutoff the extreme value
    cutoff_value = Sample_IPW['W'].quantile(q=0.9)
    Sample_IPW['W'] = Sample_IPW['W'].clip(upper=cutoff_value)

    return Sample_IPW


def DoublyRobustBound(Sample_1, Sample_2, Params, ztype='robust'):
    """
    Inputs:
        - Sample_1: pd.DataFrame
        - Sample_2: pd.DataFrame
        - Params: dictionary

    Outputs:
        pd.DataFrame
    """
    # Specify the parameters
    instrument = Params['instrument']
    features = Params['observables']
    decision = Params['decision']
    outcome  = Params['outcome']

    # Regress on the propensity score P(D = 1 | X = x) with Sample_1
    Model_D = NuisanceEstimation(Sample=Sample_1, features=features, label=decision, instrument=None, ztype='weighted')

    # Regress on the propensity score P(D = 1 | X = x, Z = z) with Sample_1
    Model_Dz = NuisanceEstimation(Sample=Sample_1, features=features, label=decision, instrument=instrument, ztype='robust')

    # Regress on the conditional probability function P(Y = 1 | D = 1, X = x) with Sample_1
    Sample_1_selected = Sample_1.copy()
    Sample_1_selected = Sample_1_selected[Sample_1_selected[decision] == 1]
    Model_Y1 = NuisanceEstimation(Sample=Sample_1_selected, features=features, label=decision, instrument=None, ztype='weighted')

    # Regress on the conditional probability function P(DY = 1 | X = x, Z = z) with Sample_1
    Sample_1_augmented = Sample_1.copy()
    Sample_1_augmented['DY'] = Sample_1_augmented[decision] * Sample_1_augmented[outcome]
    Model_DYz = NuisanceEstimation(Sample=Sample_1_augmented, features=features, label='DY', instrument=instrument, ztype='robust')

    # Estimate the nuisance functions for Sample_2
    X = Sample_2[features].values
    
    # Doubly Robust Estimator
    DREstimate = Model_Y1.predict_proba(X)[:, 1] + Sample_2[decision] * (Sample_2[outcome] - Model_Y1.predict_proba(X)[:, 1]) / Model_D.predict_proba(X)[:, 1]
    

    # Construct IV upper bounds
    UBounds = pd.DataFrame()
    Sample_2_augmented = Sample_2.copy() # Construct the augmented sample for z-values 
    
    for z in np.unique(Sample_2[instrument]):
        # Augmented sample for z-values
        Sample_2_augmented[f'z={z}'] = z
        features_z = features + [f'z={z}']
        Xz = Sample_2_augmented[features_z].values

        # Compute the IV upper bound
        UBounds[f'z={z}'] = Model_Dz.predict_proba(Xz)[:, 0] + Model_DYz.predict_proba(Xz)[:, 1] - Model_Y1.predict_proba(X)[:, 1]
    
    # Compute the tight upper bound
    U = UBounds.min(axis=1).values

    # Construct the bound of true conditional mean as pseudo outcome
    Sample_DR = Sample_2.copy()
    Sample_DR['Y_pseudo'] = DREstimate + U

    return Sample_DR


# Compute the weight function for point-learning
def IVPointWeightBinary(Sample_1, Sample_2, Params):
    """
    Inputs:
        - Sample_1: pd.DataFrame
        - Sample_2: pd.DataFrame
        - Params: dictionary

    Outputs:
        pd.DataFrame
    """
    # Specify the parameters
    instrument = Params['instrument']
    features = Params['observables']
    decision = Params['decision']
    outcome = Params['outcome'] 
    
    # Prepare the sample for prediction
    X = Sample_2[features].values

    # Construct the augmented sample for nuisance estimation
    Sample_1_aumented = Sample_1.copy()
    Sample_1_aumented['DZ']  = Sample_1_aumented[decision] * Sample_1_aumented[instrument]
    Sample_1_aumented['DY']  = Sample_1_aumented[decision] * Sample_1_aumented[outcome]
    Sample_1_aumented['DZY'] = Sample_1_aumented['DZ'] * Sample_1_aumented[outcome]        

    # Estimate the conditional probability functions
    Models = {
        'Z': NuisanceEstimation(Sample_1, features, instrument, instrument, 'weighted'),
        'D': NuisanceEstimation(Sample_1, features, decision, instrument, 'weighted')
    }
    Models.update({
        'DZ':  NuisanceEstimation(Sample_1_aumented, features, 'DZ', instrument, 'weighted'),
        'DY':  NuisanceEstimation(Sample_1_aumented, features, 'DY', instrument, 'weighted'),
        'DZY': NuisanceEstimation(Sample_1_aumented, features, 'DZY', instrument, 'weighted')
    })

    # Compute the numerator: Cov(DY, Z | X) = E[DZY | X] - E[DY | X] * E[Z | X]
    numerator_1 = Models['DZY'].predict_proba(X) @ np.unique(Sample_1_aumented['DZY'])
    numerator_0 = (Models['DY'].predict_proba(X) @ np.unique(Sample_1_aumented['DY'])) * (Models['Z'].predict_proba(X) @ np.unique(Sample_1_aumented[instrument]))

    # Compute the denominator: Cov(D, Z | X) = E[DZ | X] - E[D | X] * E[Z | X]
    denominator_1 = Models['DZ'].predict_proba(X) @ np.unique(Sample_1_aumented['DZ'])
    denominator_0 = (Models['D'].predict_proba(X) @ np.unique(Sample_1_aumented[decision])) * (Models['Z'].predict_proba(X) @ np.unique(Sample_1_aumented[instrument]))

    # Compute the point weight w_point = Cov(Y, Z | X)/Cov(D, Z | X)
    W_point = (numerator_1 - numerator_0) / (denominator_1 - denominator_0) - 0.5

    # Construct new sample
    PointSample = Sample_2.copy()
    PointSample['W'] = (W_point > 0).astype(int)
    PointSample['Weight'] = np.abs(W_point)

    upper_cutoff = PointSample['Weight'].quantile(q=0.9)
    PointSample['Weight'] = PointSample['Weight'].clip(upper=upper_cutoff)

    return PointSample


# Compute the weight function for multi-class point-learning
def IVPointWeightMulti(Sample_1, Sample_2, Params):
    """
    Inputs:
        - Sample_1: pd.DataFrame
        - Sample_2: pd.DataFrame
        - Params: dictionary

    Compute:
        - Eta_k = P(Y=k | X) = Cov(DY_k, Z | X) / Cov(D, Z | X) for any k

    Outputs:
        pd.DataFrame
    """
    # Specify the parameters
    instrument = Params['instrument']
    features = Params['observables']
    decision = Params['decision']
    outcome = Params['outcome'] 

    if len(np.unique(Sample_1[outcome])) != len(np.unique(Sample_2[outcome])):
        raise ValueError("The number of labels in Sample_1 and Sample_2 must be the same")

    # Prepare the sample for prediction
    X = Sample_2[features].values
    Y_vals = np.unique(Sample_1[outcome]).astype(int)

    # Construct the augmented label for sample_1 to help nuisance estimation
    Sample_1_augmented = Sample_1.copy()
    Sample_1_augmented['DZ'] = Sample_1_augmented[decision] * Sample_1_augmented[instrument]

    Sample_1_dummy_outcome = pd.get_dummies(Sample_1[outcome], dtype=int)
    for k in Y_vals:
        Sample_1_augmented[f'DY{k}']  = Sample_1_augmented[decision] * Sample_1_dummy_outcome[k]
        Sample_1_augmented[f'DZY{k}'] = Sample_1_augmented['DZ'] * Sample_1_dummy_outcome[k]

    # Estimate the conditional probability functions
    Models = {
        'Z':  NuisanceEstimation(Sample_1, features, instrument, instrument, 'weighted'),
        'D':  NuisanceEstimation(Sample_1, features, decision, instrument, 'weighted'),
        'DZ': NuisanceEstimation(Sample_1_augmented, features, 'DZ', instrument, 'weighted')
    }
    for k in Y_vals:
        Models.update({
            f'DZY{k}': NuisanceEstimation(Sample_1_augmented, features, f'DZY{k}', instrument, 'weighted'),
            f'DY{k}':  NuisanceEstimation(Sample_1_augmented, features, f'DY{k}', instrument, 'weighted')
        })

    # Compute the denominators of eta_k: Cov(D, Z | X) = E[DZ | X] - E[D | X] * E[Z | X]
    denominator_1 = Models['DZ'].predict_proba(X) @ np.unique(Sample_1_augmented['DZ'])
    denominator_0 = (Models['D'].predict_proba(X) @ np.unique(Sample_1_augmented[decision])) * (Models['Z'].predict_proba(X) @ np.unique(Sample_1_augmented[instrument]))

    # Iteratively estimate eta_k for k = 1, ..., K
    eta_lst = []
    for k in Y_vals:

        # Compute the numerator: Cov(DY_k, Z | X) = E[DZY_k | X] - E[DY_k | X] * E[Z | X]
        numerator_1 = Models[f'DZY{k}'].predict_proba(X) @ np.unique(Sample_1_augmented[f'DZY{k}'])
        numerator_0 = (Models[f'DY{k}'].predict_proba(X) @ np.unique(Sample_1_augmented[f'DY{k}'])) * (Models['Z'].predict_proba(X) @ np.unique(Sample_1_augmented[instrument]))
            
        # Compute the of eta_k = Cov(DY_k, Z | X) / Cov(D, Z | X)
        eta_k = (numerator_1 - numerator_0) / (denominator_1 - denominator_0)
        eta_lst.append(eta_k.reshape(-1, 1))

    # Aggregate the conditional probabilities eta_k for k = 1, ..., K
    eta_df = pd.DataFrame(np.concatenate(eta_lst, axis=1), columns=Y_vals)
    eta_max = np.max(eta_df.values, axis=1)

    # Construct cost-sensitive weight based on Sample 2
    PointSample = Sample_2.copy()

    # Compute w_k = max_p eta_p - eta_k
    for k in Y_vals:
        PointSample[f'W_{k}'] = eta_max - eta_df[k].values
        upper_cutoff = PointSample[f'W_{k}'].quantile(q=0.9)
        PointSample[f'W_{k}'] = PointSample[f'W_{k}'].clip(upper=upper_cutoff)
    
    return PointSample



# Compute the weight function for partial-learning
def IVPartialWeightBinary(Sample_1, Sample_2, Params, ztype='robust', alph=0.0, beta=1.0):
    """
    Inputs:
        - Sample_1: pd.DataFrame
        - Sample_2: pd.DataFrame
        - Params: dictionary
        - alph: float (lower bound adjustment)
        - beta: float (upper bound adjustment)

    Outputs:
        pd.DataFrame
    """
    
    # Specify the parameters
    instrument = Params['instrument']
    features = Params['observables']
    decision = Params['decision']
    outcome  = Params['outcome']

    # --------------------------------------------------------------------
    # Part 1: Fit the nuisance functions from Sample_1
    # --------------------------------------------------------------------

    # Construct the augmented sample for nuisance functions estimation
    Sample_1_augmented = Sample_1.copy()
    Sample_1_augmented['DY'] = Sample_1_augmented[decision] * Sample_1_augmented[outcome]

    # Check the type utilizing the instrumental variable z
    if ztype == 'pointwise':

        # Estimates the conditional probability functions
        Models = {
            'D': NuisanceEstimation(Sample_1_augmented, features, decision, instrument, 'pointwise'),
            'DY': NuisanceEstimation(Sample_1_augmented, features, 'DY', instrument, 'pointwise')
        }

    elif ztype == 'robust':

        # Estimates the conditional probability functions
        Models = {
            'D': NuisanceEstimation(Sample_1_augmented, features, decision, instrument, 'robust'),
            'DY': NuisanceEstimation(Sample_1_augmented, features, 'DY', instrument, 'robust')
        }

        # Construct the augmented sample for z-values 
        Sample_2_augmented = Sample_2.copy()
        for z in z_vals:
            Sample_2_augmented[f'z={z}'] = z

    else:
        raise ValueError("Invalid ztype. Must be 'pointwise' or 'robust'.")


    # --------------------------------------------------------------------
    # Part 2: Compute the nuisance functions based on Sample_2
    # --------------------------------------------------------------------

    # Prepare the data for predictions
    X = Sample_2[features].values
    z_vals =  np.unique(Sample_2[instrument]).astype(int)

    # Compute the sequence of lower and upper bounds
    Lbounds = pd.DataFrame()
    Ubounds = pd.DataFrame()
    
    # Enumerate all possible values of z for eta_z(X) = E[Y^{\star} \mid X, Z = z]
    for z in z_vals:
        
        if ztype == 'pointwise':
            # Compute lower- and upper-bounds, l_z(X) = E[DY | X, Z = z] + alpha * P(D = 0 | X, Z = z)
            Additive = Models['D'][f'z={z}'].predict_proba(X)[:, 0]
            Lbounds[f'z={z}'] = (Models['DY'][f'z={z}'].predict_proba(X) @ np.unique(Sample_1_augmented['DY'])) + alph * Additive
            Ubounds[f'z={z}'] = Lbounds[f'z={z}'] + (beta - alph) * Additive

        elif ztype == 'robust':
            features_z = features + [f'z={z}']
            Xz = Sample_2_augmented[features_z].values
        
            # Additive = 1 - (Models['D'].predict_proba(Xz) @ np.unique(Sample_1_aumented[decision]))
            Additive = Models['D'].predict_proba(Xz)[:, 0]
            Lbounds[f'z={z}'] = (Models[f'DY'].predict_proba(Xz) @ np.unique(Sample_1_augmented[f'DY'])) + alph * Additive
            Ubounds[f'z={z}'] = Lbounds[f'z={z}'] + (beta - alph) * Additive

        else:
            raise ValueError("Invalid ztype. Must be 'pointwise' or 'robust'.")

    
    # --------------------------------------------------------------------
    # Part 3: Compute the weight functions based on Sample_2
    # --------------------------------------------------------------------

    # Construct the lower and upper bounds of P(Y* = 1 | X)
    L = Lbounds.max(axis=1).values - 0.5
    U = Ubounds.min(axis=1).values - 0.5

    # Compute the weight function
    W_partial = np.array(np.abs(U) * (U > 0).astype(int) - np.abs(L) * (L < 0).astype(int))

    # Construct new sample
    PartialSample = Sample_2.copy()
    PartialSample['W'] = (W_partial > 0).astype(int)
    PartialSample['Weight'] = np.abs(W_partial)

    # PartialSample['Lower_1'] = Lbounds.max(axis=1).values
    # PartialSample['Upper_1'] = Ubounds.min(axis=1).values

    return PartialSample



# Compute the weight function for multi-class partial-learning
def IVPartialWeightMulti(Sample_1, Sample_2, Params, ztype='robust', alph=0.0, beta=1.0):
    """
    Inputs:
        - Sample_1: pd.DataFrame
        - Sample_2: pd.DataFrame
        - Params: dictionary
        - alph: float (lower bound adjustment)
        - beta: float (upper bound adjustment)

    Outputs:
        pd.DataFrame
    """
    # Specify the parameters
    instrument = Params['instrument']
    features = Params['observables']
    decision = Params['decision']
    outcome  = Params['outcome']

    if len(np.unique(Sample_1[outcome])) != len(np.unique(Sample_2[outcome])):
        raise ValueError("The number of labels in Sample_1 and Sample_2 must be the same")
        
    # Retrieve the value of variables
    z_vals = np.unique(Sample_1[instrument]).astype(int)
    y_vals = np.unique(Sample_1[outcome]).astype(int)

    # --------------------------------------------------------------------
    # Part 1: Fit the nuisance functions from Sample_1
    # --------------------------------------------------------------------

    # Construct the augmented sample for nuisance functions estimation
    Sample_1_augmented = Sample_1.copy()
    Sample_1_dummy_outcome = pd.get_dummies(Sample_1[outcome], dtype=int)
    for k in y_vals:
        Sample_1_augmented[f'DY{k}'] = Sample_1_augmented[decision] * Sample_1_dummy_outcome[k]

    # Check the type utilizing the instrumental variable z
    if ztype == 'pointwise':

        # Estimate the conditional probability functions
        Models = {'D': NuisanceEstimation(Sample_1, features, decision, instrument, 'pointwise')}
        for k in y_vals:
            Models.update({f'DY{k}': NuisanceEstimation(Sample_1_augmented, features, f'DY{k}' , instrument, 'pointwise')})

    elif ztype == 'robust':

        # Estimate the conditional probability functions
        Models = {'D': NuisanceEstimation(Sample_1, features, decision, instrument, 'robust')}
        for k in y_vals:
            Models.update({f'DY{k}': NuisanceEstimation(Sample_1_augmented, features, f'DY{k}' , instrument, 'robust')})

        # Construct the augmented sample for z-values 
        Sample_2_augmented = Sample_2.copy()
        for z in z_vals:
            Sample_2_augmented[f'z={z}'] = z

    else:
        raise ValueError("Invalid ztype. Must be 'pointwise' or 'robust'.")

    # --------------------------------------------------------------------
    # Part 2: Predict the nuisance functions for Sample_2
    # --------------------------------------------------------------------

    # Estimate the lower and upper bounds in Sample 2 for eta_k, k = 1, ..., K
    PartialSample = Sample_2.copy()
    X = Sample_2[features].values

    for k in y_vals:
        # Compute the sequence of lower and upper bounds
        Lbounds = pd.DataFrame()
        Ubounds = pd.DataFrame()
        
        # Enumerate all possible values of z
        for z in z_vals:

            # Compute the lower and upper bounds: l_z(X) = E[DYk | X, Z = z] + alpha * P(D = 0 | X, Z = z), Yk = 0,1
            # predict_proba returns (n_samples, n_classes) array, calsses are order as np.unique(y)
            if ztype == 'pointwise':
                Additive = Models['D'][f'z={z}'].predict_proba(X)[:, 0]
                Lbounds[f'z={z}'] = (Models[f'DY{k}'][f'z={z}'].predict_proba(X) @ np.unique(Sample_1_augmented[f'DY{k}'])) + alph * Additive
                Ubounds[f'z={z}'] = Lbounds[f'z={z}'] + (beta - alph) * Additive

            elif ztype == 'robust':
                features_z = features + [f'z={z}']
                Xz = Sample_2_augmented[features_z].values
            
                # Additive = 1 - (Models['D'].predict_proba(Xz) @ np.unique(Sample_1_aumented[decision]))
                Additive = Models['D'].predict_proba(Xz)[:, 0]
                Lbounds[f'z={z}'] = (Models[f'DY{k}'].predict_proba(Xz) @ np.unique(Sample_1_augmented[f'DY{k}'])) + alph * Additive
                Ubounds[f'z={z}'] = Lbounds[f'z={z}'] + (beta - alph) * Additive

            else:
                raise ValueError("Invalid ztype. Must be 'pointwise' or 'robust'.")

        # Construct the lower and upper bounds of eta_k
        PartialSample[f'L_{k}'] = Lbounds.quantile(0.8, axis=1).values
        PartialSample[f'U_{k}'] = Ubounds.quantile(0.2, axis=1).values

    # --------------------------------------------------------------------
    # Part 3: Compute the realizable partial bounds and the weight function for Sample_2
    # --------------------------------------------------------------------

    # Compute the realizable lower and upper bound for eta_k, k = 1, ..., K
    for k in y_vals:
        # Initialize the realizable upper-bound of eta_k
        Lk_Realized = 1
        Uk_Realized = 1

        # Handle the constraints that sum_k eta_k = 1
        for p in y_vals:
            if p != k:
                Lk_Realized -= PartialSample[f'U_{p}']
                Uk_Realized -= PartialSample[f'L_{p}']
        
        # Element-wise comparison
        PartialSample[f'LR_{k}'] = np.maximum(Lk_Realized, PartialSample[f'L_{k}'])
        PartialSample[f'UR_{k}'] = np.minimum(Uk_Realized, PartialSample[f'U_{k}'])  

    # Compute w_k = max_p [tilde{u}_p - tilde{l}_k]^{+}
    for k in y_vals:
        # Width_k = UpperBounds_Realized.values - LowerBounds_Realized[k].values.reshape(-1, 1)
        Width_k = PartialSample.filter(like='U_').values - PartialSample[f'L_{k}'].values.reshape(-1, 1)
        PartialSample[f'W_{k}'] = np.maximum(np.max(Width_k, axis=1), 0)

    return PartialSample



# ##########################################################################################
# B. Learning Algorithm
# ##########################################################################################

# Cross-fitting for Nuisance Estimations
def CrossFitting(Sample, Params, WeightFunc, k_folds=5, random_state=0):
    """
        Inputs:
            - Train: pd.DataFrame
            - Parms: dictionary
            - WeightFunc: function obejct
            - k_folds: int

        Output:     
            - Weighted_Train: pd.DataFrame
    """
    # KFold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    # Training sample list
    Sample_Lst = []

    # Sample splitting and computing weight function
    for major_idx, minor_idx in kf.split(Sample):

        # Sample splitting, K-1 folds and 1 fold
        Sample_major = Sample.iloc[major_idx, :].copy()
        Sample_minor = Sample.iloc[minor_idx, :].copy()

        # New sample with weights
        Sample_Lst.append( WeightFunc(Sample_1=Sample_major, Sample_2=Sample_minor, Params=Params) )

    # Concat the K-folds weighted sample
    return pd.concat(Sample_Lst, axis=0)



# Multiclass/Binary Classification on Selective/Full Labels
def Classification(Sample, Params, Selective=True, IPW=False, k_folds=5, random_state=0):
    """
    Inputs:
        - Sample: pd.DataFrame
        - Params: dictionary
        - Selective: bool
        - IPW: bool

    Outputs:
        - Result: dictionary containing accuracy and best estimator
    """

    # Specify the parameters  
    features = Params['observables']
    decision = Params['decision']
    outcome  = Params['outcome']

    # Split into training and testing set with ratio 70-30
    Train, Test = train_test_split(Sample, test_size=0.3, random_state=random_state)

    # Determine training sample based on conditions
    if IPW:
        Sample_train = CrossFitting(Sample=Train, Params=Params, WeightFunc=InversePropensityWeight, k_folds=k_folds, random_state=random_state)
        if Selective:
            Sample_train = Sample_train[Sample_train[decision] == 1]
    else:
        Sample_train = Train.copy()
        Sample_train['W'] = 1
        if Selective:
            Sample_train = Sample_train[Sample_train[decision] == 1]

    # Define a helper function for model training and scoring
    def train_and_score(model):
        # grid_search = GridSearchCV(estimator=model, param_grid=params, cv=k_folds)
        model.fit(X=Sample_train[features], y=Sample_train[outcome], sample_weight=Sample_train['W'])
        score = model.score(Test[features], Test[outcome])
        return score

    # Train models and get accuracy scores and best estimators
    scores_and_estimators = {
        "LogisticRegression": train_and_score(LogisticRegression()),
        "SVM": train_and_score(SVC()),
        "RandomForest": train_and_score(RandomForestClassifier()),
        "GradientBoosting": train_and_score(HistGradientBoostingClassifier()),
        "AdaBoost": train_and_score(AdaBoostClassifier(algorithm='SAMME'))
    }

    # Extract accuracy scores and best estimators
    accuracy = {model: np.around(score, decimals=3) for model, score in scores_and_estimators.items()}
    # best_estimators = {model: estimator for model, (_, estimator) in scores_and_estimators.items()}

    return accuracy


def DRLearner(Sample, Params, k_folds=5, random_state=0):
    """
    Methodology:
        mu^{star}(X) = P(Y(1) = 1 | X), 
        mu_{DR}(X) = (mu_1(X) + D(Y - mu_1(X)) / pi_1(X)) + delta_{upper}(X)
        delta_{upper}(X) = min_{z} {pi_p(X, Z) + E[DY | X = x, Z = z] - mu_1(X) }
        Regress on mu_{DR}(X) using logistisc regression with l1 penalty.
        If mu_{DR}(X) > 0.5 then Y = 1, otherwise Y = 0.
    Inputs:
        - Sample: pd.DataFrame
        - Params: dictionary
        - k_folds: int

    Outputs:
        - Result: dictionary containing accuracy and best estimator
    """
    # Specify the parameters
    features = Params['observables']
    # decision = Params['decision']
    outcome = Params['outcome']

    # Split the sample into train and test set
    Train, Test = train_test_split(Sample, test_size=0.3, random_state=random_state)

    # Estimate the nuisance functions on training set and evaluate on test set
    Test_DR = DoublyRobustBound(Sample_1=Train, Sample_2=Test, Params=Params, ztype='robust')

    # Evaluate the plug-in classifier on test data
    Y_test = (Test_DR['Y_pseudo'] > 0.5).astype(int)
    accuracy = np.around(accuracy_score(Test[outcome], Y_test), decimals=3)

    return accuracy



# Weighted Binary Classification on Selective Labels
def WeightedClassification(Sample, Params, WeightFunc, k_folds=5, random_state=0):
    """
    Inputs:
        - Sample: pd.DataFrame
        - Params: dictionary
        - WeightFunc: function object
        - k_folds: int

    Outputs:
        - Result: dictionary containing accuracy and best estimator
    """

    # Specify the parameters
    features = Params['observables']
    outcome = Params['outcome']

    # Split the sample into train and test set
    Train, Test = train_test_split(Sample, test_size=0.3, random_state=random_state)

    # Construct the weighted training set via cross-fitting
    Train_weighted = CrossFitting(Sample=Train, Params=Params, WeightFunc=WeightFunc, k_folds=k_folds, random_state=random_state)

    # Define a helper function for model training and scoring
    def train_and_score(model):
        # grid_search = GridSearchCV(estimator=model, param_grid=params, cv=k_folds)
        model.fit(X=Train_weighted[features], y=Train_weighted['W'], sample_weight=Train_weighted['Weight'])
        score = model.score(Test[features], Test[outcome])
        return score

    # Train models and get accuracy scores and best estimators
    scores_and_estimators = {
        "LogisticRegression": train_and_score(LogisticRegression()),
        "SVM": train_and_score(SVC()),
        "RandomForest": train_and_score(RandomForestClassifier()),
        "GradientBoosting": train_and_score(HistGradientBoostingClassifier()),
        "AdaBoost": train_and_score(AdaBoostClassifier(algorithm='SAMME'))
    }

    # Extract accuracy scores and best estimators
    accuracy = {model: np.around(score, decimals=3) for model, score in scores_and_estimators.items()}
    # best_estimators = {model: estimator for model, (_, estimator) in scores_and_estimators.items()}

    return accuracy
    


# Multiclass Classification using Neural Networks
def ClassificationNN(Sample, Params, Selective=True, IPW=False, k_folds=5, random_state=0):
    """
    Inputs:
        - Sample: pd.DataFrame
        - Params: dictionary
        - Selective: bool
        - IPW: bool

    Outputs:
        - Result: dictionary containing accuracy and trained model
    """
    # Set random seed for reproducibility
    set_random_seed(random_state)

    # Specify the parameters  
    features = Params['observables']
    decision = Params['decision']
    outcome = Params['outcome']

    # Split into training and testing set with ratio 70-30
    Train, Test = train_test_split(Sample, test_size=0.3, random_state=random_state)

    # Determine training sample based on conditions
    if IPW:
        Sample_train = CrossFitting(Sample=Train, Params=Params, WeightFunc=InversePropensityWeight, k_folds=k_folds, random_state=random_state)
        if Selective:
            Sample_train = Sample_train[Sample_train[decision] == 1]
    else:
        Sample_train = Train.copy()
        Sample_train['W'] = 1
        if Selective:
            Sample_train = Sample_train[Sample_train[decision] == 1]

    # Prepare data for PyTorch
    X_train = torch.tensor(Sample_train[features].values, dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(Sample_train[outcome].values - 1, dtype=torch.long, device=DEVICE)  # Shift labels to start at 0
    sample_weights = torch.tensor(Sample_train['W'].values, dtype=torch.float32, device=DEVICE)

    X_test = torch.tensor(Test[features].values, dtype=torch.float32, device=DEVICE)
    y_test = torch.tensor(Test[outcome].values - 1, dtype=torch.long, device=DEVICE)  # Shift labels to start at 0

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(Sample[outcome]))

    # Initialize the model
    model = nnmodel.SimpleClassifier(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction='none')  # Unreduced to apply sample weights
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        logits = model(X_train)
        loss = criterion(logits, y_train)
        weighted_loss = (loss * sample_weights).mean()
        
        # Backward pass and optimization
        weighted_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            # print(f"Epoch [{epoch}/{num_epochs}], Loss: {weighted_loss.item():.4f}")
            pass

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y_test).sum().item() / y_test.size(0)

    return {"accuracy": round(accuracy, 3), "trained_model": model}




# Cost-sensitive classification using neural networks
def CostSensitiveClassificationNN(Sample, Params, WeightFunc, surrogate='exploss', k_folds=5, random_state=0):

    """
    Inputs:
        - Sample: pd.DataFrame
        - Params: dictionary
        - WeightFunc: function object
        - k_folds: int

    Outputs:
        - Result: dictionary containing accuracy and best estimator
    """
    # Set random seed for reproducibility
    set_random_seed(random_state)

    # Specify the parameters
    features = Params['observables']
    outcome = Params['outcome']

    # Split the sample into train and test set
    Train, Test = train_test_split(Sample, test_size=0.3, random_state=random_state)

    # Construct the weighted training set via cross-fitting
    Train_weighted = CrossFitting(Sample=Train, Params=Params, WeightFunc=WeightFunc, k_folds=k_folds, random_state=random_state)

    # Specify the dimensions of the neural network
    features_dim = len(Params['observables'])
    num_classes  = len(np.unique(Sample[Params['outcome']]))

    # Specify the features data and weights
    X = torch.tensor(Train_weighted[features].values, dtype=torch.float32, device=DEVICE)
    W = torch.tensor(Train_weighted.filter(like='W_').values, dtype=torch.float32, device=DEVICE)

    # Initialize neural-network model
    model = nnmodel.SimpleClassifier(features_dim, num_classes).to(DEVICE)

    # Determines the optimizer 
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # Minimize cost-sensitive entropy-loss function
    num_epochs = 200
    for epoch in range(num_epochs):
        # Forward pass
        logits = model(X)  # Raw scores for each class

        # Compute the cost-sensitive loss
        if surrogate == 'exploss':
            predictions = torch.softmax(logits, dim=1)  # Convert to probabilities    
            weighted_loss = torch.sum(W * predictions, dim=1)  # Apply weights
            loss = torch.mean(weighted_loss)  # Average over all samples

        elif surrogate == 'logloss':
            predictions = torch.log_softmax(logits, dim=1)  # Convert to log-probabilities
            weighted_loss = torch.sum(W * predictions, dim=1)  # Apply weights
            loss = torch.mean(weighted_loss)  # Average over all samples

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss for every 10 epochs
        if epoch % 10 == 0:
            # print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")
            pass

    # Move model to evaluation mode
    model.eval()

    # Make predictions on the test set
    with torch.no_grad():
        inputs = torch.tensor(Test[features].values, dtype=torch.float, device=DEVICE)
        labels = torch.tensor(Test[outcome].values, device=DEVICE)
        outputs = model(inputs)
        predicted = torch.argmax(outputs, dim=1) + 1.0

    # Compute the accuracy
    total_samples = labels.shape[0]
    accuracy = (predicted == labels).sum().item() / total_samples

    return {"accuracy": round(accuracy, 3), "trained_model": model}





# ##########################################################################################
# C. Evaluation on Selective Labelled Data
# ##########################################################################################

# def ExactEvaluation(Sample, Params, Result, binary=True, k_folds=5):
#     """
#     Inputs:
#         - Sample: pd.DataFrame
#         - Params: dictionary
#         - Result: dictionary containing accuracy and best estimator
#         - binary: bool
#         - k_folds: int

#     Outputs:
#         - PartialRisk: dictionary 
#     """
#     # Specify the parameters
#     features = Params['observables']

#     # Select the features data
#     X = Sample[features]

#     # Compute the lower and upper bounds for conditional probabilities 
#     Sample_valid = CrossFitting(Sample=Sample, Params=Params, WeightFunc=IVPointWeight, k_folds=k_folds)
#     # print(Sample_valid)

#     # Compute the partial bounds of misclassification risk for a given estimator
#     def ComputeRiskBounds(Sample, X, Model=None, binary=True, k_folds=5):

#         if Model is None:
#             Model = LogisticRegression()

#         # Predict the outcome
#         y_pred = Model.predict(X)

#         if binary:
#             # worstcase risk = 1/n \sum_{i=1}^{n} |W(X_i)| * 1(sign[w(X_i)] != t(X_i) ) 
#             WorstcaseRisk = Sample['Weight'] * (Sample['W'].values != y_pred).astype(int)

#         else:
#             # worstcase risk = 1/n \sum_{i=1}^{n} \sum_{k=1}^{K} W(X_i) \cdot I{t(X_i) = k}
#             WorstcaseRisk = []
#             for i in range(len(y_pred)):
#                 y = y_pred[i]
#                 idx = Sample.index[i]
#                 WorstcaseRisk.append( Sample[f'W_{y}'][idx] )
            
#         return np.mean(WorstcaseRisk)

#     # Retrieve the best estimators (a dictionary)
#     BestEstimators = Result['best_estimators']

#     # Partial bounds of misclassification risk
#     PartialRisk = dict()

#     # Define a function to run ComputeRiskBounds and return the result
#     def process_estimator(name, model):
#         return name, ComputeRiskBounds(Sample=Sample_valid, X=X, Model=model, binary=binary, k_folds=k_folds)

#     # Use joblib.Parallel to run the tasks in parallel
#     results = Parallel(n_jobs=N_JOBS)(
#         delayed(process_estimator)(name, model) for name, model in BestEstimators.items()
#     )

#     # Collect the results into a dictionary
#     PartialRisk = {name: risk_bounds for name, risk_bounds in results}

#     return PartialRisk




# def WorstcaseEvaluation(Sample, Params, Result, binary=True, k_folds=5):
#     """
#     Inputs:
#         - Sample: pd.DataFrame
#         - Params: dictionary
#         - Result: dictionary containing accuracy and best estimator
#         - binary: bool
#         - k_folds: int

#     Outputs:
#         - PartialRisk: dictionary 
#     """
#     # Specify the parameters
#     features = Params['observables']

#     # Select the features data
#     X = Sample[features]

#     # Compute the lower and upper bounds for conditional probabilities 
#     Sample_valid = CrossFitting(Sample=Sample, Params=Params, WeightFunc=IVPartialWeight, k_folds=k_folds)
#     # print(Sample_valid)

#     # Compute the partial bounds of misclassification risk for a given estimator
#     def ComputeRiskBounds(Sample, X, Model=None, binary=True, k_folds=5):

#         if Model is None:
#             Model = LogisticRegression()

#         # Predict the outcome
#         y_pred = Model.predict(X)

#         if binary:
#             # worstcase risk = 1/n \sum_{i=1}^{n} |W(X_i)| * 1(sign[w(X_i)] != t(X_i) ) 
#             WorstcaseRisk = Sample['Weight'] * (Sample['W'].values != y_pred).astype(int)

#         else:
#             # worstcase risk = 1/n \sum_{i=1}^{n} \sum_{k=1}^{K} W(X_i) \cdot I{t(X_i) = k}
#             WorstcaseRisk = []
#             for i in range(len(y_pred)):
#                 y = y_pred[i]
#                 idx = Sample.index[i]
#                 WorstcaseRisk.append( Sample[f'W_{y}'][idx] )
            
#         return np.mean(WorstcaseRisk)

#     # Retrieve the best estimators (a dictionary)
#     BestEstimators = Result['best_estimators']

#     # Partial bounds of misclassification risk
#     PartialRisk = dict()

#     # Define a function to run ComputeRiskBounds and return the result
#     def process_estimator(name, model):
#         return name, ComputeRiskBounds(Sample=Sample_valid, X=X, Model=model, binary=binary, k_folds=k_folds)

#     # Use joblib.Parallel to run the tasks in parallel
#     results = Parallel(n_jobs=N_JOBS)(
#         delayed(process_estimator)(name, model) for name, model in BestEstimators.items()
#     )

#     # Collect the results into a dictionary
#     PartialRisk = {name: risk_bounds for name, risk_bounds in results}

#     return PartialRisk




# def PartialEvaluation(Sample, Params, Result, binary=True, k_folds=5):
#     """
#     Inputs:
#         - Sample: pd.DataFrame
#         - Params: dictionary
#         - Result: dictionary containing accuracy and best estimator
#         - binary: bool
#         - k_folds: int

#     Outputs:
#         - PartialRisk: dictionary 
#     """
#     # Specify the parameters
#     features = Params['observables']

#     # Select the features data
#     X = Sample[features]

#     # Compute the lower and upper bounds for conditional probabilities 
#     Sample_valid = CrossFitting(Sample=Sample, Params=Params, WeightFunc=IVPartialWeight, k_folds=k_folds)
#     # print(Sample_valid) 

#     # Compute the partial bounds of misclassification risk for a given estimator
#     def ComputeRiskBounds(Sample, X, Model=None, binary=True, k_folds=5):

#         if Model is None:
#             Model = LogisticRegression()

#         # Predict the outcome
#         y_pred = Model.predict(X)

#         if binary:
#             # lower risk = 1/n \sum_{i=1}^{n} l_1(X_i) + (1 - u_1(X_i) - l_1(X_i)) * I{t(X_i) = 1}
#             # upper risk = 1/n \sum_{i=1}^{n} u_1(X_i) + (1 - u_1(X_i) - l_1(X_i)) * I{t(X_i) = 1}
#             LowerRisk = Sample['Lower_1'] + (1 - Sample['Lower_1'] - Sample['Upper_1']) * (y_pred > 0).astype(int)
#             UpperRisk = Sample['Upper_1'] + (1 - Sample['Lower_1'] - Sample['Upper_1']) * (y_pred > 0).astype(int)

#             lower_risk = np.mean(LowerRisk)
#             upper_risk = np.mean(UpperRisk)

#         else:
#             # lower_risk = 1 - 1/n \sum_{i=1}^{n} \sum_{k=1}^{K} u(X_i) \cdot I{t(X_i) = k}
#             # upper_risk = 1 - 1/n \sum_{i=1}^{n} \sum_{k=1}^{K} l(X_i) \cdot I{t(X_i) = k}
#             LowerRisk = []
#             UpperRisk = []
#             for i in range(len(y_pred)):
#                 y = y_pred[i]
#                 idx = Sample.index[i]
#                 LowerRisk.append( Sample['Upper_'+str(y)][idx] )
#                 UpperRisk.append( Sample['Lower_'+str(y)][idx] )
            
#             lower_risk = 1 - np.mean(LowerRisk)
#             upper_risk = 1 - np.mean(UpperRisk)

#         return lower_risk, upper_risk

#     # Retrieve the best estimators (a dictionary)
#     BestEstimators = Result['best_estimators']

#     # Partial bounds of misclassification risk
#     PartialRisk = dict()

#     # Define a function to run ComputeRiskBounds and return the result
#     def process_estimator(name, model):
#         return name, ComputeRiskBounds(Sample=Sample_valid, X=X, Model=model, binary=binary, k_folds=k_folds)

#     # Use joblib.Parallel to run the tasks in parallel
#     results = Parallel(n_jobs=N_JOBS)(
#         delayed(process_estimator)(name, model) for name, model in BestEstimators.items()
#     )

#     # Collect the results into a dictionary
#     PartialRisk = {name: risk_bounds for name, risk_bounds in results}

#     return PartialRisk




