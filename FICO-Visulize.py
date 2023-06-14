import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


## Global hyperparameters
MISS_TYPE = {'NUCEM', 'UC'}
ALPHA = {0.1, 0.3, 0.5, 0.7, 0.9}
BETA = {1.0, 0.5, 0.25}


def ACCBoxplot(Data, model, alphas, betas, labeled=True):

    # fig, axes = plt.subplots(nrows=len(betas), ncols=len(alphas), sharey=False, figsize=(6*len(alphas), 6*len(betas)), dpi=400)
    # plt.subplots_adjust(hspace=.35)

    handles = None
    labels = None

    for i in range(len(betas)):

        fig, axes = plt.subplots(nrows=1, ncols=len(alphas), sharey=False, figsize=(6*len(alphas), 5), dpi=400)
        plt.subplots_adjust(hspace=.35)        

        # Speicfy the stength of missing parameter
        beta = list(betas)[i]

        for j in range(len(alphas)):
            
            # Speicfy the strength of unmesaured confounding
            alpha = list(alphas)[j]

            # Retreive data of (alpha, beta) pair
            data = Data.loc[(alpha, beta), :]
            data_lot = pd.DataFrame(data=data.stack().reset_index().values, columns=['Estimators', 'Learners', 'Test Data Accuracy'])
            
            # Boxlots with no legends
            bp = sns.boxplot(ax=axes[j], data=data_lot, x='Learners', y='Test Data Accuracy', hue='Estimators')
            bp.legend_.remove()

            # Handle legend
            handles, labels = axes[j].get_legend_handles_labels()
            # print(labels)
            axes[j].tick_params(axis='x', rotation=30)
            axes[j].set_title('(alpha, beta) = ({}, {})'.format(alpha, beta))
            axes[j].set_ylim(0.60,0.75)

        if labeled:
            labels = ['AdaBoost', 'GradientBoosting', 'LogisticRegression', 'RandomForest', 'SVM']
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncols=5)
            fig.suptitle('Model {}'.format(model, beta), x=0.5, y=1.05, fontsize=16)
        
        else:
            fig.suptitle('Model {}'.format(model, beta), x=0.5, y=1.0, fontsize=16)

        # Output graphs 
        fig.savefig('Accuracy-{}-beta-{}.pdf'.format(model, beta), bbox_inches='tight')

    return


def AUCBoxplot(Data, model, alphas, betas):

    handles = None
    labels = None

    # Plot ACC results
    for i in range(len(betas)):

        fig, axes = plt.subplots(nrows=1, ncols=len(alphas), sharey=False, figsize=(6*len(alphas), 5), dpi=400)
        plt.subplots_adjust(hspace=.35)

        # Speicfy the stength of missing parameter
        beta = list(betas)[i]

        for j in range(len(alphas)):
            
            # Speicfy the strength of unmesaured confounding
            alpha = list(alphas)[j]

            # Retreive data of (alpha, beta) pair
            data = Data.loc[(alpha, beta), :]
            data_lot = pd.DataFrame(data=data.stack().reset_index().values, columns=['Estimators', 'Learners', 'Test Data AUC'])
            
            # Boxlots  
            bp = sns.boxplot(ax=axes[j], data=data_lot, x='Learners', y='Test Data AUC', hue='Estimators')
            bp.legend_.remove()

            # Handle legend
            handles, labels = axes[j].get_legend_handles_labels()
            axes[j].tick_params(axis='x', rotation=30)
            axes[j].set_title('(alpha, beta) = ({}, {})'.format(alpha, beta))
            axes[j].set_ylim(0.65,0.85)

        labels = ['AdaBoost', 'GradientBoosting', 'LogisticRegression', 'RandomForest', 'SVM']
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncols=5)
        fig.suptitle('Area-under-the-curves of Model {} with beta = {}'.format(model, beta), x=0.5, y=1.05, fontsize=16)

        # Output graphs 
        fig.savefig('AUC-{}-beta-{}.pdf'.format(model, beta), bbox_inches='tight')

    return



if __name__ == '__main__':

    with open('HELOC-ACC-Multi-IV.pickle', 'rb') as handle:
        ACC_Pickle = pickle.load(handle)

    with open('HELOC-AUC-Multi-IV.pickle', 'rb') as handle:
        AUC_Pickle = pickle.load(handle)

    print('------DATA INPUT COMPLETE------')

    ## Convert dictionaries into multi-index dataframes
    ACC_DF = pd.concat([ACC_Pickle[key] for key in ACC_Pickle.keys()], axis=0, keys=ACC_Pickle.keys())
    AUC_DF = pd.concat([AUC_Pickle[key] for key in AUC_Pickle.keys()], axis=0, keys=AUC_Pickle.keys())

    ## Extract and plot ACC and AUC scores of NUCEM model
    ACC_NUCEM = ACC_DF.loc['NUCEM', :].sort_index()
    AUC_NUCEM = AUC_DF.loc['NUCEM', :].sort_index()

    ACC_UC    = ACC_DF.loc['UC', :].sort_index()
    AUC_UC    = AUC_DF.loc['UC', :].sort_index()

    # ACC_NUCEM.to_csv('ACC_Model_1.csv')
    # AUC_NUCEM.to_csv('AUC_Model_1.csv')

    # ACC_UC.to_csv('ACC_Model_2.csv')
    # AUC_UC.to_csv('AUC_Model_2.csv')
        
    ## Accuracy Graphs
    ACCBoxplot(Data=ACC_NUCEM, model='1', alphas=[ 0.5, 0.7, 0.9], betas=[1.0, 0.5, 0.25], labeled=True)
    # AUCBoxplot(Data=AUC_NUCEM, model='1', alphas=[ 0.5, 0.7, 0.9], betas=[1.0, 0.5, 0.25])

    ACCBoxplot(Data=ACC_UC, model='2', alphas=[ 0.5, 0.7, 0.9], betas=[1.0, 0.5, 0.25], labeled=False)
    # AUCBoxplot(Data=AUC_UC, model='2', alphas=[ 0.5, 0.7, 0.9], betas=[1.0, 0.5, 0.25])

    print('------PLOTS COMPLETE------')

    ## Aread-under-the-curve groupby mean
    # AUC_NUCEM.groupby(level=[0,1,2]).mean().to_csv('AUC-NUCEM.csv')
    # AUC_UC.groupby(level=[0,1,2]).mean().to_csv('AUC-UC.csv')
            
    # print('------OUTPUT DATAFRAMES COMPLETE------')










