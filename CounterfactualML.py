
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.svm import SVC, SVR
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# from econml.grf import CausalIVForest


# =============================================================================
# Hyperparameters Set
# =============================================================================

# PARAMS_LINEAR   = {'C':[1.0], 'kernel': ['linear']}
# PARAMS_GAUSSIAN = {'C':[1.0], 'kernel': ['rbf']}


PARAMS_LOGISTIC = {'C':[1.0], 'max_iter':[500] }
PARAMS_SVM      = {'C':[1.0], 'kernel': ['linear', 'rbf'] }
PARAMS_LOGLOSS  = {'loss':['log_loss'], 'n_estimators':[100, 300], 'max_depth':[3], 'min_samples_leaf':[10, 30]}
PARAMS_EXPLOSS  = {'loss':['exponential'], 'n_estimators':[100, 300], 'max_depth':[3], 'min_samples_leaf':[10, 30]}
PARAMS_RFOREST  = {'criterion':['gini', 'entropy']}
# PARAMS_RFOREST  = {'criterion':['gini'], 'n_estimators':[100, 300], 'max_depth':[5, 10, None], 'min_samples_leaf':[10, 30]}


# =============================================================================
# Supervised Learning
# =============================================================================


def SupervisedLearning(Data, Params=dict(), Selected=True):
    """
        Inputs:
            - Data: pd.DataFrame
            - Parms: dictionary
            - Selected: bool

        Outputs:
            - accuracy: dictionary
            - roc_auc: dictionary
    """
    
    ## Instantiate parameters
    features = list(Params["features"]) + list(Params["instrument"])
    decision = Params["decision"]
    outcome = Params["outcome"]
 
    ## Split into training and testing set with ratio 70-30
    Train, Test = train_test_split(Data, test_size=0.3, random_state=0)
    
    ## Check selective labels condition
    if (Selected == True):
        Train = Train[Train[decision] == 1]
    else:
        pass

    ## Logistic Regression -- Logarithm loss
    model_logistic = GridSearchCV(estimator=LogisticRegression(), param_grid=PARAMS_LOGISTIC, n_jobs=3)
    model_logistic.fit(X=Train[features], y=Train[outcome])

    ypred_logistic = model_logistic.predict_proba(Test[features])[:, 1]
    score_logistic = model_logistic.score(Test[features], Test[outcome])
    auc_logistic   = metrics.roc_auc_score(Test[outcome], ypred_logistic)

    # print("Supervised - LogisticRegression", model_logistic.best_params_)


    ## SVM -- Hinge loss
    model_svm = GridSearchCV(estimator=SVC(probability=True), param_grid=PARAMS_SVM, n_jobs=3)
    model_svm.fit(X=Train[features], y=Train[outcome])

    ypred_svm = model_svm.predict_proba(Test[features])[:, 1]
    score_svm = model_svm.score(Test[features], Test[outcome])
    auc_svm   = metrics.roc_auc_score(Test[outcome], ypred_svm)

    # print("Supervised - SupportVectorClassifier ", model_svm.best_params_)
    
    
    ## Random Forest - Gini / Entropy index
    model_rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=PARAMS_RFOREST, n_jobs=3)
    model_rf.fit(X=Train[features], y=Train[outcome])

    ypred_rf = model_rf.predict_proba(Test[features])[:, 1]
    score_rf = model_rf.score(Test[features], Test[outcome])
    auc_rf = metrics.roc_auc_score(Test[outcome], ypred_rf)

    # print("Supervised - Random Forest", model_rf.best_params_)

    
    ## Gradient Boosting Tree - Logistic loss
    model_logloss = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=PARAMS_LOGLOSS, n_jobs=3)
    model_logloss.fit(X=Train[features], y=Train[outcome])

    ypred_logloss = model_logloss.predict_proba(Test[features])[:, 1]
    score_logloss = model_logloss.score(Test[features], Test[outcome])
    auc_logloss   = metrics.roc_auc_score(Test[outcome], ypred_logloss)

    # print("Supervised - Gradient Boosting", model_logloss.best_params_)


    ## AdaBoosting Tree - Exponential loss
    model_exploss = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=PARAMS_EXPLOSS, n_jobs=3)
    model_exploss.fit(X=Train[features], y=Train[outcome])

    ypred_exploss = model_exploss.predict_proba(Test[features])[:, 1]
    score_exploss = model_exploss.score(Test[features], Test[outcome])
    auc_exploss = metrics.roc_auc_score(Test[outcome], ypred_exploss)

    # print("Supervised - AdaBoosting", model_exploss.best_params_)
    
    
    ## Accuracy scores
    accuracy = dict()

    accuracy["LogisticRegression"] = np.around(score_logistic, decimals=3)
    accuracy["SVM"] = np.around(score_svm, decimals=3)
    accuracy["RandomForest"] = np.around(score_rf, decimals=3)
    accuracy["AdaBoosting"] = np.around(score_exploss, decimals=3)
    accuracy["GradientBoosting"] = np.around(score_logloss, decimals=3)

    
    ## Area under the curve
    roc_auc = dict()

    roc_auc["LogisticRegression"] = np.around(auc_logistic, decimals=3)    
    roc_auc["SVM"] = np.around(auc_svm, decimals=3)    
    roc_auc["RandomForest"] = np.around(auc_rf, decimals=3)
    roc_auc["AdaBoosting"] = np.around(auc_exploss, decimals=3)
    roc_auc["GradientBoosting"] = np.around(auc_logloss, decimals=3)

    return accuracy, roc_auc



# =============================================================================
# Counterfactual Learning
# =============================================================================

def CrossFitting(Train, Params, WeightFunc, k_folds=5):
    """
        Inputs:
            - Train: pd.DataFrame
            - Parms: dictionary
            - WeightFunc: function obejct
            - k_folds: int

        Output: 
            pd.DataFrame
    """
    
    ## Training sample list
    Train_Lst = []

    ## Sample splitting and computing weight function
    for major_idx, minor_idx in KFold(n_splits=k_folds, shuffle=True, random_state=0).split(Train):

        ## New sample with weights
        Train_Lst.append( WeightFunc(Sample_1=Train.iloc[major_idx, :], Sample_2=Train.iloc[minor_idx, :], Params=Params) )

    ## Concat the K-folds weighted sample
    return pd.concat(Train_Lst, axis=0)



def CounterLearning(Data, Params, WeightFunc, k_folds=5):
    """
        Inputs:
            - Data: pd.DataFrame
            - Parms: dictionary
            - k_folds: int

        Outputs:
            - accuracy: dictionary
            - roc_auc: dictionary
    """
    
    ## Sepcify the parameters
    features = Params['features']
    outcome = Params['outcome']

    ## Split the sample into train and test set 
    Train, Test = train_test_split(Data, test_size=0.3, random_state=0)
    
    ## Construct the weighted traning set via cross-fitting
    Train_weighted = CrossFitting(Train, Params, WeightFunc, k_folds)

    
    ## Logistic Regression -- Logarithm loss
    model_logistic = GridSearchCV(estimator=LogisticRegression(), param_grid=PARAMS_LOGISTIC, n_jobs=3)
    model_logistic.fit(X=Train_weighted[features], y=Train_weighted['W'], sample_weight=Train_weighted['Weight'])

    ypred_logistic = model_logistic.predict_proba(Test[features])[:, 1]
    score_logistic = model_logistic.score(Test[features], Test[outcome])
    auc_logistic   = metrics.roc_auc_score(Test[outcome], ypred_logistic)

    # print("Counterfactual - LogisticRegression", model_logistic.best_params_)


    ## SVM -- Hinge loss
    model_svm = GridSearchCV(estimator=SVC(probability=True), param_grid=PARAMS_SVM, n_jobs=3)
    model_svm.fit(X=Train_weighted[features], y=Train_weighted['W'], sample_weight=Train_weighted['Weight'])

    ypred_svm = model_svm.predict_proba(Test[features])[:, 1]
    score_svm = model_svm.score(Test[features], Test[outcome])
    auc_svm   = metrics.roc_auc_score(Test[outcome], ypred_svm)

    # print("Counterfactual - SupportVectorClassifier", model_svm.best_params_)
    
    
    ## Random Forest - Gini / Entropy index
    model_rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=PARAMS_RFOREST, n_jobs=3)
    model_rf.fit(X=Train_weighted[features], y=Train_weighted['W'], sample_weight=Train_weighted['Weight'])

    ypred_rf = model_rf.predict_proba(Test[features])[:, 1]
    score_rf = model_rf.score(Test[features], Test[outcome])
    auc_rf = metrics.roc_auc_score(Test[outcome], ypred_rf)

    # print("Counterfactual - Random Forest", model_rf.best_params_)

    
    ## Gradient Boosting Tree - Logistic loss
    model_logloss = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=PARAMS_LOGLOSS, n_jobs=3)
    model_logloss.fit(X=Train_weighted[features], y=Train_weighted['W'], sample_weight=Train_weighted['Weight'])

    ypred_logloss = model_logloss.predict_proba(Test[features])[:, 1]
    score_logloss = model_logloss.score(Test[features], Test[outcome])
    auc_logloss   = metrics.roc_auc_score(Test[outcome], ypred_logloss)

    # print("Counterfactual - Gradient Boosting", model_logloss.best_params_)


    ## AdaBoosting Tree - Exponential loss
    model_exploss = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=PARAMS_EXPLOSS, n_jobs=3)
    model_exploss.fit(X=Train_weighted[features], y=Train_weighted['W'], sample_weight=Train_weighted['Weight'])

    ypred_exploss = model_exploss.predict_proba(Test[features])[:, 1]
    score_exploss = model_exploss.score(Test[features], Test[outcome])
    auc_exploss = metrics.roc_auc_score(Test[outcome], ypred_exploss)

    # print("Counterfactual - AdaBoosting", model_exploss.best_params_)
    
    
    ## Accuracy scores
    accuracy = dict()

    accuracy["LogisticRegression"] = np.around(score_logistic, decimals=3)
    accuracy["SVM"] = np.around(score_svm, decimals=3)
    accuracy["RandomForest"] = np.around(score_rf, decimals=3)
    accuracy["AdaBoosting"] = np.around(score_exploss, decimals=3)
    accuracy["GradientBoosting"] = np.around(score_logloss, decimals=3)
    

    ## Area under the curve
    roc_auc = dict()

    roc_auc["LogisticRegression"] = np.around(auc_logistic, decimals=3)    
    roc_auc["SVM"] = np.around(auc_svm, decimals=3)    
    roc_auc["RandomForest"] = np.around(auc_rf, decimals=3)
    roc_auc["AdaBoosting"] = np.around(auc_exploss, decimals=3)
    roc_auc["GradientBoosting"] = np.around(auc_logloss, decimals=3)

    return accuracy, roc_auc


# =============================================================================
# Point Identification
# =============================================================================

"""
Point Identification:
    - mu(X) = Cov(DY,Z|X) / Cov(D,Z|X) 
    - mu(X) = (E[DYZ|X] - E[DY|X] * E[Z|X]) / (E[DZ|X] - E[D|X] * E[Z|X])
"""

def PointWeightR(Sample_1, Sample_2, Params):
    """
        Inputs:
            - Sample_1: pd.DataFrame
            - Sample_2: pd.DataFrame
            - Params: dictionary

        Outputs:
            pd.DataFrame
    """
    ## Calling R via rpy2
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    from rpy2.robjects import pandas2ri
    import rpy2.robjects as robj
    numpy2ri.activate()

    ## Specify the parameters
    instrument = Params["instrument"]
    decision = Params["decision"]
    features = Params["features"]
    labels = Params["labels"]
    
    ## Training and estimating sample
    X = Sample_1[features].values
    X = robj.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
    Y = robj.FloatVector(Sample_1[labels['DY']].values)
    D = robj.FloatVector(Sample_1[decision].values)
    Z = robj.FloatVector(Sample_1[instrument].values)

    X_point = Sample_2[features].values
    X_point = robj.r.matrix(X_point, nrow=X_point.shape[0], ncol=X_point.shape[1])

    ## Estimating conditional covariance ratio via GRF
    grf = importr("grf")
    IV_forest = grf.instrumental_forest(X=X, Y=Y, W=D, Z=Z)
    IV_pred = robj.r.predict(IV_forest, newdata=X_point)
    
    ## Converting R dataframe into pandas dataframe
    with (robj.default_converter + pandas2ri.converter).context():
        IV_pred = robj.conversion.get_conversion().rpy2py(IV_pred)
    
    ## From dataframe into numpy array
    W_point = np.array(IV_pred['predictions']) - 1/2

    ## Construct new sample with weight
    PointSample = Sample_2.copy()
    PointSample['W'] = np.int_(W_point > 0)
    PointSample['Weight'] = np.abs(W_point)

    # PointSample['Weight'] = np.minimum(PointSample['Weight'], PointSample['Weight'].quantile(q=0.8))

    return PointSample



def PointWeightGRF(Sample_1, Sample_2, Params):
    """
        Inputs:
            - Sample_1: pd.DataFrame
            - Sample_2: pd.DataFrame
            - Params: dictionary

        Outputs:
            pd.DataFrame
    """

    ## Specify the parameters
    instrument = Params["instrument"]
    decision = Params["decision"]
    features = Params["features"]
    labels = Params["labels"]

    ## Training sample
    DY = Sample_1[labels['DY']].values
    D  = Sample_1[decision].values
    X  = Sample_1[features].values
    Z  = Sample_1[instrument].values

    ## Fitting IV regression model via Generalized Random Forest
    est = CausalIVForest(criterion='mse', n_estimators=500, min_samples_leaf=30)
    est.fit(X=X, T=D, y=DY, Z=Z)

    ## Compute the point weight
    X_point = Sample_2[features].values
    W_point = est.predict(X_point, interval=False) - 1/2

    ## Construct new sample
    PointSample = Sample_2.copy()
    PointSample['W'] = np.int_(W_point > 0)
    PointSample['Weight'] = np.abs(W_point)

    # print(W_point.max(), W_point.min())

    PointSample['Weight'] = np.minimum(PointSample['Weight'], PointSample['Weight'].quantile(q=0.8))

    return PointSample


def PointWeight(Sample_1, Sample_2, Params):
    """
        Inputs:
            - Sample_1: pd.DataFrame
            - Sample_2: pd.DataFrame
            - Params: dictionary

        Outputs:
            pd.DataFrame
    """
    
    ## Specify the parameters
    instrument = Params['instrument']
    features = Params['features']
    decision = Params['decision']
    labels = Params['labels']
    
    ## Estimate the nusiance functions using 
    modelZ = NusianceBoosting(Sample_1, features, instrument, instrument, ztype="weighted")
    modelD = NusianceBoosting(Sample_1, features, decision, instrument, ztype="weighted")
    modelDZ = NusianceBoosting(Sample_1, features, labels["DZ"], instrument, ztype="weighted")
    modelDY = NusianceBoosting(Sample_1, features, labels["DY"], instrument, ztype="weighted")
    modelDYZ = NusianceBoosting(Sample_1, features, labels["DYZ"], instrument, ztype="weighted")

    ## Sample need to predict
    X = Sample_2[features].values
    Z = Sample_2[instrument].values
    Z_max= np.max(Z)
    # print("Maximum value of Z: ", Z_max)

    ## Compute the numerators 
    numerator_1 = pd.DataFrame( modelDYZ.predict_proba(X) )[1].values
    # numerator_0 = pd.DataFrame( modelDY.predict_proba(X) )[1].values * Z_max / 2
    numerator_0 = pd.DataFrame( modelDY.predict_proba(X) )[1].values * pd.DataFrame( modelZ.predict_proba(X) )[1].values

    ## Compute the denominators
    denominator_1 = pd.DataFrame( modelDZ.predict_proba(X) )[1].values
    # denominator_0 = pd.DataFrame( modelD.predict_proba(X) )[1].values * Z_max / 2
    denominator_0 = pd.DataFrame( modelD.predict_proba(X) )[1].values * pd.DataFrame( modelZ.predict_proba(X) )[1].values

    ## Compute the Point Weight
    W_point = (numerator_1 - numerator_0) / (denominator_1 - denominator_0) - 1/2

    ## Construct new sample
    PointSample = Sample_2.copy()
    PointSample['W'] = np.int_(W_point > 0)
    PointSample['Weight'] = np.abs(W_point)

    # print(W_point.max(), W_point.min())

    ## Cuttoff the extreme value
    PointSample['Weight'] = np.minimum(PointSample['Weight'], PointSample['Weight'].quantile(q=0.8))
        
    return PointSample


# =============================================================================
# Partial Identification
# =============================================================================
"""
IV Partial Bounds
    - $ l(X) = max_z { E[DY | X, Z = z] + alpha(X) * (1 - E[D | X, Z = z]) } - 1/2 $
    - $ u(X) = min_z { E[DY | X, Z = z] + beta(X)  * (1 - E[D | X, Z = z]) } - 1/2 $
    - In binary classification, alpha(X) = 0, beta(X) = 1

Partial Weight
    - $ w(X) = |u(X)| * 1(u(X) > 0) - |l(X)| * 1(l(X) < 0) $
"""

def PartialWeightAdapative(Sample_1, Sample_2, Params, alpha=0, beta=1):
    """
        Inputs:
            - Sample_1: pd.DataFrame
            - Sample_2: pd.DataFrame
            - Params: dictionary

        Outputs:
            pd.DataFrame
    """
    
    ## Specify the parameters
    instrument = Params["instrument"]
    features = Params["features"]
    decision = Params["decision"]
    labels = Params["labels"]

    ## Number of sample
    zval = np.unique(Sample_2[instrument]).astype(int)
    size = np.sum(Sample_1[decision] == 1) 

    ## Compute the sequence of lower and upper bounds for Sample_2
    Lbounds = pd.DataFrame()
    Ubounds = pd.DataFrame()

    ## Features of Sample_2
    X = Sample_2[features].values

    ## Run robust estimation of partial weight if sample size is too small
    if (size / len(zval) <= 200):

        ## Estimate the nusiance functions using Data_1
        modelD  = NusianceBoosting(Sample_1, features, decision, instrument, ztype="robust")
        modelDY = NusianceBoosting(Sample_1, features, labels["DY"], instrument, ztype="robust")
    
        ## Enumerate all possible values of z
        for z in zval:
            
            ## Construct the covariates with distinct z values
            Z = z * np.ones(X.shape[0]).reshape(-1, 1)
            X_new = np.append(X, Z, axis=1) 

            ## Compute lower and upper bound for Sample_2
            Additive = 1 - pd.DataFrame( modelD.predict_proba(X_new) )[1].values

            Lbound = pd.DataFrame( modelDY.predict_proba(X_new) )[1].values + alpha * Additive
            Ubound = pd.DataFrame( modelDY.predict_proba(X_new) )[1].values + beta * Additive

            ## Add to the sequences
            Lbounds["z="+str(z)] = Lbound
            Ubounds["z="+str(z)] = Ubound

    else:

        ## Estimate the nusiance functions using Data_1
        modelDs = NusianceBoosting(Sample_1, features, decision, instrument, ztype="pointwise")
        modelDYs = NusianceBoosting(Sample_1, features, labels["DY"], instrument, ztype="pointwise")

        for z in zval:

            ## Compute lower and upper bound for Sample_2
            Additive = 1 - pd.DataFrame( modelDs["z="+str(z)].predict_proba(X) )[1].values

            Lbound = pd.DataFrame( modelDYs["z="+str(z)].predict_proba(X) )[1].values + alpha * Additive
            Ubound = pd.DataFrame( modelDYs["z="+str(z)].predict_proba(X) )[1].values + beta * Additive


            ## Add to the sequences
            Lbounds["z="+str(z)] = Lbound
            Ubounds["z="+str(z)] = Ubound

    ## Construct the lower- and upper-bounds of E[Y^{*} | X] - 1/2
    L = Lbounds.max(axis=1) - 1/2
    U = Ubounds.min(axis=1) - 1/2

    ## Compute the weight function
    W_partial = np.array(np.abs(U) * np.int_(U > 0) - np.abs(L) * np.int_(L < 0))
    
    ## Construct new sample 
    PartialSample = Sample_2.copy()
    PartialSample['W'] = np.int_(W_partial > 0)
    PartialSample['Weight'] = np.abs(W_partial)

    return PartialSample



def PartialWeightRobust(Sample_1, Sample_2, Params, alpha=0, beta=1):
    """
        Inputs:
            - Sample_1: pd.DataFrame
            - Sample_2: pd.DataFrame
            - Params: dictionary

        Outputs:
            pd.DataFrame
    """
    
    ## Specify the parameters
    instrument = Params["instrument"]
    features = Params["features"]
    decision = Params["decision"]
    labels = Params["labels"]

    ## Estimate the nusiance functions using Data_1
    modelD  = NusianceBoosting(Sample_1, features, decision, instrument, ztype="pointwise")
    modelDY = NusianceBoosting(Sample_1, features, labels["DY"], instrument, ztype="pointwise")

    ## Compute the sequence of lower and upper bounds for Sample_2
    Lbounds = pd.DataFrame()
    Ubounds = pd.DataFrame()

    ## Features of Sample_2
    X = Sample_2[features].values
    
    ## Enumerate all possible values of z
    for z in np.unique(Sample_2[instrument]).astype(int):
        
        ## Construct the covariates with distinct z values
        Z = z * np.ones(X.shape[0]).reshape(-1, 1)
        X_new = np.append(X, Z, axis=1) 

        ## Compute lower and upper bound for Sample_2
        Additive = 1 - pd.DataFrame( modelD.predict_proba(X_new) )[1].values

        Lbound = pd.DataFrame( modelDY.predict_proba(X_new) )[1].values + alpha * Additive
        Ubound = pd.DataFrame( modelDY.predict_proba(X_new) )[1].values + beta * Additive

        ## Add to the sequences
        Lbounds["z="+str(z)] = Lbound
        Ubounds["z="+str(z)] = Ubound

    ## Construct the lower- and upper-bounds of (E[Y^{*} | X] - 1/2)
    L = Lbounds.max(axis=1) - 1/2
    U = Ubounds.min(axis=1) - 1/2

    ## Compute the weight function
    W_partial = np.array(np.abs(U) * np.int_(U > 0) - np.abs(L) * np.int_(L < 0))
    
    ## Construct new sample 
    PartialSample = Sample_2.copy()
    PartialSample['W'] = np.int_(W_partial > 0)
    PartialSample['Weight'] = np.abs(W_partial)

    return PartialSample


def PartialWeight(Sample_1, Sample_2, Params, alpha=0, beta=1):
    """
        Inputs:
            - Sample_1: pd.DataFrame
            - Sample_2: pd.DataFrame
            - Params: dictionary

        Outputs:
            pd.DataFrame
    """
    
    ## Specify the parameters
    instrument = Params["instrument"]
    features = Params["features"]
    decision = Params["decision"]
    labels = Params["labels"]

    ## Estimate the nusiance functions using Data_1
    modelDs = NusianceBoosting(Sample_1, features, decision, instrument, ztype="pointwise")
    modelDYs = NusianceBoosting(Sample_1, features, labels["DY"], instrument, ztype="pointwise")

    ## Construct the weights for Data_2
    X = Sample_2[features].values

    ## Compute the sequence of lower and upper bounds
    Lbounds = pd.DataFrame()
    Ubounds = pd.DataFrame()

    ## Enumerate all possible values of z
    zvals = np.unique(Sample_2[instrument]).astype(int)
    for z in zvals:

        Additive = 1 - pd.DataFrame( modelDs["z="+str(z)].predict_proba(X) )[1].values

        Lbound = pd.DataFrame( modelDYs["z="+str(z)].predict_proba(X) )[1].values + alpha * Additive
        Ubound = pd.DataFrame( modelDYs["z="+str(z)].predict_proba(X) )[1].values + beta * Additive

        Lbounds["z="+str(z)] = Lbound
        Ubounds["z="+str(z)] = Ubound

    L = Lbounds.max(axis=1) - 1/2
    U = Ubounds.min(axis=1) - 1/2

    ## Compute the weight function
    W_partial = np.array(np.abs(U) * np.int_(U > 0) - np.abs(L) * np.int_(L < 0))
    
    ## Construct new sample 
    PartialSample = Sample_2.copy()
    PartialSample['W'] = np.int_(W_partial > 0)
    PartialSample['Weight'] = np.abs(W_partial)

    return PartialSample


# =============================================================================
# Nuisance Tools
# =============================================================================

# labels.key() = {DZ, DY, DYZ}
# params.keys() = {labels, outcome, decision, instrument, features}

def NusianceBoosting(Sample, features, label, instrument, ztype="weighted"):
    """
        Inputs:
            - Sample: pd.DataFrame
            - features: list of str
            - label: str 
            - instrument: str
            - ztype: str

        Output: 
            sklearn.model or list of sklearn.models
    """ 

    if ztype == "weighted":

        ## Features and labels
        X = np.array( Sample[features] )
        Y = np.array( Sample[label]    ).reshape(-1)

        # Training procedure
        model = GradientBoostingClassifier(n_estimators=500, max_depth=3)
        model.fit(X, Y)

        return model

    elif ztype == "robust":

        ## Feautres + instrument as covariates
        features_z = list(features).copy()
        features_z.append(instrument)

        ## Covaraites and Outcome
        X = np.array( Sample[features_z] )
        Y = np.array( Sample[label] ).reshape(-1)

        ## Training procedure
        model = GradientBoostingClassifier(n_estimators=500, max_depth=3)
        model.fit(X, Y)

        return model

    elif ztype == "pointwise":

        ## Enumerate possible z values
        z_vals = np.unique(Sample[instrument]).astype(int)

        ## Initiate a containter for the estimator of each possible z
        models = dict() 

        ## Fit estimators for each possible z
        for z in z_vals:

            ## Features and labels
            X = np.array( Sample[Sample[instrument] == z][features] )
            Y = np.array( Sample[Sample[instrument] == z][label]    ).reshape(-1)

            # Grid search fitting
            model = GradientBoostingClassifier(n_estimators=500, max_depth=3)
            model.fit(X, Y)

            ## Adding the estimator to the container
            models["z=" + str(z)] = model

        return models
    
    else:
        print("ERROR: ztype")
        return 




