
import warnings
warnings.filterwarnings('ignore')

import pickle
import numpy as np
import pandas as pd
import itertools as it
import multiprocessing as mp 
from scipy.special import expit

import datetime
import CounterfactualML as cml


## Experiment repeated time
REPEATED = 50

## Global Parameters
MISS_TYPE = {'NUCEM', 'UC'}
ALPHA = {0.1, 0.3, 0.5, 0.7, 0.9}
BETA = {1.0, 0.5, 0.25}


## Generate Synthetic Dataset
def SamplingFICO(data, mtype='NUCEM', alpha=0.5, beta=1.0, seed=0):

    if mtype not in MISS_TYPE:
        raise ValueError("Missing types must be one of %r." % MISS_TYPE)
    
    if (alpha <= 0.0) or (alpha >= 1.0):
        raise ValueError("Alpha must take value within (0, 1)")    
        
    if (beta <= 0.0) or (beta > 1.0):
        raise ValueError("Beta must take value within (0, 1]")
        
    # Parameters setting
    alpha_lst = np.array(list(ALPHA))
    
    # Sampling with replacement
    df = data.copy()
    nn = df.shape[0]

    # Observable covariates and unmeasured covariates
    np.random.seed(seed)
    Z  = df['Z'].values
    U = df['Residuals'].values
    X = df['ExternalRiskEstimate'].values
    
    # Probability distribution
    if (mtype == 'NUCEM'):
        ProbD = beta * (alpha * expit(8 * U) + (1 - alpha) * expit((1 + Z) * X))
        
    elif (mtype == 'UC'):
        ProbD = beta * (expit(alpha * 6 * U + (1 - alpha) * (1 + Z) * X))
        
    else:
        ProbD = 1.0
    
    # Aggegrate the data
    df['ProbD'] = ProbD
    df['D'] = np.random.binomial(n=1, p=ProbD, size=nn)
    df['DZ'] = df['D'] * df['Z']
    df['DY'] = df['D'] * df['RiskPerformance']
    df['DYZ'] = df['D'] * df['RiskPerformance'] * df['Z']

    return df


## Conduct the experiment with specific hyperparameter tuple
def Experiment(value_tuple):

    # 1. Input dataset and create recording
    with open('Data/Heloc_IV.pickle', 'rb') as handle:
        HELOC = pickle.load(handle)
        
    ACC_DF = pd.DataFrame(columns=["Selected Sample", "Point Learning", "Partial Learning", "Partial Learning Robust", "Full Sample"])
    AUC_DF = pd.DataFrame(columns=["Selected Sample", "Point Learning", "Partial Learning", "Partial Learning Robust", "Full Sample"])


    # 2. Parameters setting
    n_repeat = REPEATED
    missing, alpha, beta = value_tuple
    
    labels = dict()
    labels["Z"] = "Z"
    labels["D"] = "D"
    labels["DZ"] = "DZ"
    labels["DY"] = "DY"
    labels["DYZ"] = "DYZ"

    params = dict()
    params["labels"] = labels
    params["outcome"] = "RiskPerformance"
    params["decision"] = "D"
    params["instrument"] = "Z"
    params["features"] = list( HELOC.columns[2:-2] )


    # 3. Data sampling
    Sample_Lst = []
    for i in range(n_repeat):
        
        # sampling data with distinct seed
        sample = SamplingFICO(data=HELOC, mtype=missing, alpha=alpha, beta=beta, seed=i*5)
        Sample_Lst.append(sample)


    # 4. Repeated experiments
    for i in range(n_repeat):

        print(value_tuple, " repeated round "+str(i))

        ## Select samples (dict) with distinct alpha values
        sample = Sample_Lst[i]

        ## 1. Supervised Learning
        acc_super, auc_super = cml.SupervisedLearning(Data=sample, Params=params, Selected=True)

        ## 2. Full Sample Supversied Learning
        acc_full, auc_full = cml.SupervisedLearning(Data=sample, Params=params, Selected=False)

        ## 3. Point Identification Learning
        acc_point, auc_point = cml.CounterLearning(Data=sample, Params=params, WeightFunc=cml.PointWeight, k_folds=5)
        
        ## 4. Partial Identification Learning
        acc_partial, auc_partial = cml.CounterLearning(Data=sample, Params=params, WeightFunc=cml.PartialWeightAdapative, k_folds=5)
        
        ## Aggregate the accuracy of each methods
        Accuracy = pd.DataFrame(index=acc_super.keys())

        Accuracy["Selected Sample"] = acc_super.values()
        Accuracy["Point Learning"] = acc_point.values()
        Accuracy["Partial Learning"] = acc_partial.values()
        # Accuracy["Partial Learning Robust"] = acc_partial_robust.values()
        Accuracy["Full Sample"] = acc_full.values()


        ## Aggregate the area_under_curve of each methods
        Area_under_curve = pd.DataFrame(index=auc_super.keys())

        Area_under_curve["Selected Sample"] = auc_super.values()
        Area_under_curve["Point Learning"] = auc_point.values()
        Area_under_curve["Partial Learning"] = auc_partial.values()
        # Area_under_curve["Partial Learning Robust"] = auc_partial_robust.values()
        Area_under_curve["Full Sample"] = auc_full.values()
        

        ## Aggregate the results for each round
        ACC_DF = pd.concat([ACC_DF, Accuracy], axis=0)
        AUC_DF = pd.concat([AUC_DF, Area_under_curve], axis=0)


    return tuple(value_tuple), ACC_DF, AUC_DF



if __name__ == '__main__':

    ## Parameters of interest
    Iterations = [list(MISS_TYPE), list(ALPHA), list(BETA)]
    ValueTuple = list(pd.MultiIndex.from_product(Iterations))
    
    print(datetime.datetime.now())    


    ## Multiprocessing via pooling
    with mp.Pool(processes=len(ValueTuple)) as pool:
        Results = pool.map(Experiment, ValueTuple)

    print(datetime.datetime.now())


    ## Unzip the results
    ACC_dict = dict()
    AUC_dict = dict()

    for i in range(len(ValueTuple)):
        value_tuple, acc_df, auc_df = Results[i]
        print(i, type(value_tuple))

        ACC_dict[value_tuple] = acc_df
        AUC_dict[value_tuple] = auc_df

            
    ## Dumping the results to pickles
    with open('HELOC-ACC-Multi-IV.pickle', 'wb') as handle1:
        pickle.dump(ACC_dict, handle1, protocol=pickle.HIGHEST_PROTOCOL)

    with open('HELOC-AUC-Multi-IV.pickle', 'wb') as handle2:
        pickle.dump(AUC_dict, handle2, protocol=pickle.HIGHEST_PROTOCOL)


