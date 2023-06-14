import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def Standardization(df):
    return ( df - df.mean(axis=0) ) / df.std(axis=0)


# =============================================================================
# 1. Preprocessing
# =============================================================================

# input original data
HelocData = pd.read_csv("Heloc/heloc_dataset_v1.csv")

# factorize category variable
HelocData['RiskPerformance'] = pd.factorize(HelocData["RiskPerformance"])[0]

# standardization
StandardData = HelocData.copy()
StandardData.iloc[:, 1:] = Standardization(StandardData.iloc[:, 1:])

# features name
features = HelocData.columns[1:]


# =============================================================================
# 2. Construct Unmeasured Residuals
# =============================================================================

# specify features and outcome
X = HelocData[features]
y = HelocData['RiskPerformance']

# run random forest regression
reg = RandomForestClassifier(n_estimators=50)
reg.fit(X, y)
reg.score(X, y)

# compute residuals of prediction
residuals = y - pd.DataFrame(reg.predict_proba(X))[1]


# =============================================================================
# 3. Generate Multiple Decision-makers
# =============================================================================

# Create 10 distinct decision-makers
np.random.seed(10)
num = 10
experts = np.random.multinomial(n=1, pvals=np.ones(num)/num, size=HelocData.shape[0])
experts = np.argmax(experts, axis=1)


# =============================================================================
# Aggregate Synthetic Data
# =============================================================================

# Create semi-synthetic dataset
OutputData = StandardData.copy()
OutputData['Residuals'] = residuals
OutputData['Z'] = experts
# OutputData.hist()

# Output new dataframe
with open('Heloc_IV_2.pickle', 'wb') as handle:
    pickle.dump(OutputData, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('Heloc_IV_2.pickle', 'rb') as handle:
    TmpData = pickle.load(handle)