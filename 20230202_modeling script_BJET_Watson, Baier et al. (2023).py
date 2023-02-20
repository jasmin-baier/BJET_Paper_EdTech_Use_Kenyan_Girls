###### --------------------------------------------- ######
# AUTHOR: Joe Mark Watson
# REVIEWER: Jasmin Baier
# DATE: February 2023
###### --------------------------------------------- ######


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from numpy import arange
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from relaxed_lasso import RelaxedLasso  # https://stats.stackexchange.com/questions/122955/why-is-relaxed-lasso-different-from-standard-lasso
from sklearn.feature_selection import RFE  # see here for all feature selection stuff available through sklearn https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
# RFE is equiv to backward elim: https://stats.stackexchange.com/questions/450518/rfe-vs-backward-elimination-is-there-a-difference#:~:text=Here%2C%20the%20writer%20suggests%20that%20RFE%20is%20a%20type%20of,essential%20difference%20is%20not%20addressed.
# from scipy.stats import shapiro  # not currently using
# from scipy.stats import normaltest  # not currently using


# LOAD DATA PRE-CLEANING AND MANIP
child_df = pd.read_csv(
    "~/Downloads/raw data_anonymized_BJET_Watson, Baier et al. (2023).csv")
modelling_items = list(pd.read_csv("~/Downloads/modelling_cols.csv")['0'])


# INITIAL CLEANING AND MANIP
child_df_reg = child_df[child_df['gender_child'] == 'Female']  # subset to retain female only (929 to 604)
child_df_reg = child_df_reg[modelling_items]
child_df_reg.isnull().sum() * 100 / len(child_df_reg)  # how much missingness pre dropping
child_df_reg = child_df_reg.dropna()  # drop any where missing (604 to 544)


def binary_fun(in_var):
    out_var = [1 if x == "Yes" else 0 for x in in_var]
    return (out_var)


C8_cols = [i for i in list(child_df_reg.columns) if i.startswith("C8_")]
C8_cols_numeric = child_df_reg[C8_cols].transform(lambda x: binary_fun(x))  # apply function to all C8_cols
child_df_reg['C8sum'] = C8_cols_numeric.sum(axis=1)
child_df_reg = child_df_reg.drop(C8_cols, axis=1)

# manip access variable - changing to binary if > 0
child_df_reg['binary_access_hrs_tech'] = np.where(child_df_reg['sum_access_hrs_tech'] > 0, 1, 0)
child_df_reg = child_df_reg.drop(['sum_access_hrs_tech'], axis=1)

binary_cols1 = [i for i in list(child_df_reg.columns) if i.startswith("C7_")]
binary_cols2 = [i for i in list(child_df_reg.columns) if i.startswith("B15_")]
binary_cols = binary_cols1 + binary_cols2
trans_df = child_df_reg[binary_cols].transform(lambda x: binary_fun(x))  # apply function to all binary_cols
child_df_reg2 = pd.concat([child_df_reg.drop(binary_cols, axis=1).reset_index(drop=True),
                           trans_df.reset_index(drop=True)], axis=1)
# child_df_reg2['county'] = np.where(child_df_reg2['county'] == 'Nairobi', 1, 0)  # not needed as country dropped at start
# make income an ordered cat before later transformation
child_df_reg2['income'] = pd.Categorical(child_df_reg2['income'])
child_df_reg2['income'] = child_df_reg2['income'].cat.codes


# LOAD DATA POST-CLEANING AND MANIP
child_df = pd.read_csv(
    "~/Downloads/clean subset data_BJET_Watson, Baier et al. (2023).csv")


# CREATION OF y_train AND X_train, WITH y BEING THE NATURAL LOG OF sum_use_hrs_tech
y_train = np.log(1 + child_df_reg2['sum_use_hrs_tech'])  # natural log of 1 + sum_use_hrs_tech
X_train = child_df_reg2.drop(columns=['sum_use_hrs_tech'])


# MODELLING

def create_pipes():
    """make lasso, ridge and OLS pipes"""

    relasso_pipe = Pipeline([  # https://relaxedlasso.readthedocs.io/en/latest/content.html
        # ('transform', PowerTransformer(method='yeo-johnson', standardize=True)),
        # ('scale', MinMaxScaler(feature_range=(0, 1))),
        ('relasso', RelaxedLasso())
    ])

    relasso_param_grid = [
        {
            'relasso__alpha': np.arange(0.005, 0.251, 0.005),  # np.arange(0.015, 0.11, 0.01)
            'relasso__theta': np.arange(0.005, 0.251, 0.005),  # Note that θ = 1 corresponds to standard Lasso,
            # and θ = 0 theoretically corresponds to the OLS solution for var.s retained with α.
        },
    ]

    lasso_pipe = Pipeline([
        # ('transform', PowerTransformer(method='yeo-johnson', standardize=True)),
        # ('scale', MinMaxScaler(feature_range=(0, 1))),
        ('lasso', Lasso(random_state=0))
    ])

    lasso_param_grid = [
        {
            'lasso__alpha': arange(0.0001, 0.25001, 0.0001),  # arange(0.0001, 0.01001, 0.0001)
        },
    ]

    ridge_pipe = Pipeline([
        # ('transform', PowerTransformer(method='yeo-johnson', standardize=True)),
        # ('scale', MinMaxScaler(feature_range=(0, 1))),
        ('ridge', Ridge(random_state=0))
    ])

    ridge_param_grid = [
        {
            'ridge__alpha': arange(0.01, 25.01, 0.01),  # arange(0.1, 10.1, 0.1)
        },
    ]

    OLS_pipe = Pipeline([
        # ('transform', PowerTransformer(method='yeo-johnson', standardize=True)),
        # ('scale', MinMaxScaler(feature_range=(0, 1))),
        ('OLS', LinearRegression())
    ])

    back_pipe = Pipeline([
        ('selector', RFE(estimator=LinearRegression(), step=1, importance_getter='coef_'))
    ])

    back_param_grid = [
        {
            'selector__n_features_to_select': np.arange(1, 33, 1),
        },
    ]

    return relasso_pipe, relasso_param_grid, lasso_pipe, lasso_param_grid, ridge_pipe, ridge_param_grid, \
           OLS_pipe, back_pipe, back_param_grid


def do_modelling(X_train, y_train, n_folds):
    """get all key output from 3 competing models"""
    relasso_pipe, relasso_param_grid, lasso_pipe, lasso_param_grid, ridge_pipe, ridge_param_grid, \
    OLS_pipe, back_pipe, back_param_grid = create_pipes()

    # model relasso and get output
    clf_rl = GridSearchCV(relasso_pipe, param_grid=relasso_param_grid, cv=10, n_jobs=-1, verbose=1,
                          refit=True, scoring="r2")  # useful for me re refit implications for coef.s:
    # https://orvindemsy.medium.com/understanding-grid-search-randomized-cvs-refit-true-120d783a5e94
    clf_rl.fit(X_train, y_train)
    d = []
    for coef, col in enumerate(X_train.columns):
        new_row = [col, clf_rl.best_estimator_.named_steps.relasso.coef_[coef]]
        d.append(new_row)
    coefs_rl_df = pd.DataFrame(d, columns=['IV', 'coef'])
    relasso_dict = {'r2': clf_rl.best_score_, 'alpha': clf_rl.best_estimator_.named_steps.relasso.alpha,
                    'theta': clf_rl.best_estimator_.named_steps.relasso.theta,
                    'coefs': coefs_rl_df, 'all_model_info': clf_rl}

    # model lasso and get output
    clf_l = GridSearchCV(lasso_pipe, param_grid=lasso_param_grid, cv=n_folds, n_jobs=-1, verbose=1,
                         refit=True, scoring="r2")
    clf_l.fit(X_train, y_train)
    d = []
    for coef, col in enumerate(X_train.columns):
        new_row = [col, clf_l.best_estimator_.named_steps.lasso.coef_[coef]]
        d.append(new_row)
    coefs_l_df = pd.DataFrame(d, columns=['IV', 'coef'])
    lasso_dict = {'r2': clf_l.best_score_, 'alpha': clf_l.best_estimator_.named_steps.lasso.alpha,
                  'coefs': coefs_l_df, 'all_model_info': clf_l}

    # model ridge and get output
    clf_r = GridSearchCV(ridge_pipe, param_grid=ridge_param_grid, cv=n_folds, n_jobs=-1, verbose=1,
                         refit=True, scoring="r2")
    clf_r.fit(X_train, y_train)
    d = []
    for coef, col in enumerate(X_train.columns):
        new_row = [col, clf_r.best_estimator_.named_steps.ridge.coef_[coef]]
        d.append(new_row)
    coefs_r_df = pd.DataFrame(d, columns=['IV', 'coef'])
    ridge_dict = {'r2': clf_r.best_score_,
                  'alpha': clf_r.best_estimator_.named_steps.ridge.alpha,
                  'coefs': coefs_r_df, 'all_model_info': clf_r}

    # model ols and get output
    clf_o = cross_validate(OLS_pipe, X_train, y_train, cv=n_folds, return_estimator=True)
    d = []
    for fold in clf_o['estimator']:
        d.append(
            fold.named_steps.OLS.coef_)  # https://stackoverflow.com/questions/54307137/how-to-get-coefficients-with-cross-validation-model
    col_df = pd.DataFrame(d).T
    col_df['coef'] = col_df.mean(numeric_only=True, axis=1)  # giving mean coef across val folds
    col_df['IV'] = X_train.columns
    coefs_o_df = col_df[['IV', 'coef']]  # giving output in same format as lasso and ridge
    ols_dict = {'r2': clf_o['test_score'].mean(),  # ['test_score'] gives same info as cross_val_score
                'alpha': 'NA',
                'coefs': coefs_o_df, 'all_model_info': clf_o}

    # model backwards elimination and get output
    clf_b = GridSearchCV(back_pipe, param_grid=back_param_grid, cv=n_folds, n_jobs=-1, verbose=1,
                         refit=True, scoring="r2")
    clf_b.fit(X_train, y_train)
    d = []
    for coef, col in enumerate(X_train.columns[clf_b.best_estimator_.named_steps.selector.support_]):
        new_row = [col, clf_b.best_estimator_.named_steps.selector.estimator_.coef_[coef]]
        d.append(new_row)
    coefs_b_df = pd.DataFrame(d, columns=['IV', 'coef'])
    back_dict = {'r2': clf_b.best_score_,
                  'retained_feats': clf_b.best_params_['selector__n_features_to_select'],
                  'coefs': coefs_b_df, 'all_model_info': clf_b}

    return relasso_dict, lasso_dict, ridge_dict, ols_dict, back_dict


# plot outputs of do_modelling

def plot_n_coefs(technique_string, coefs_df, n_IVs):
    """plot the n largest absolute IVs"""
    # temp below
    coefs_df_plot = ols_dict['coefs'].copy()
    # temp above
    #coefs_df_plot = coefs_df.copy()  # temp hashed
    coefs_df_plot['abs_coef'] = coefs_df_plot['coef'].abs()
    coefs_df_plot = coefs_df_plot.sort_values(by=['abs_coef'], ascending=False)
    # temp below
    coefs_df_plot = coefs_df_plot.iloc[0:10, :][['IV', 'coef']]
    # temp above
    coefs_df_plot = coefs_df_plot.iloc[0:n_IVs, :][['IV', 'coef']]

    # fig = plt.figure(figsize=(7, 5))
    # creating the bar plot
    plt.bar(coefs_df_plot['IV'], coefs_df_plot['coef'], color='maroon',
            width=0.4)
    plt.ylabel("Coefficient")
    plt.xticks(rotation=45)
    title_text = str(n_IVs) + " largest absolute coeffients produced through " + technique_string + " approach"
    plt.title(title_text)

    plt.show()


relasso_dict, lasso_dict, ridge_dict, ols_dict, back_dict = do_modelling(X_train, y_train, 10)

# check desired parts of output of models
relasso_dict['theta']
relasso_dict['r2']
rdc = relasso_dict['coefs']
len(rdc[rdc['coef'] != 0])  # no. of predictors retained

# make desired model plots
plot_n_coefs("Relasso", relasso_dict['coefs'], 16)


# ________________________________________

# STANDARD OLS ON IVs RETAINED BY RELASSO - A ROBUSTNESS CHECK THAT ALSO GIVES P-VALS (AND SHOWS R2 HIGHER WHEN NO CV)
rdc1 = rdc.copy()
rdc1['abs_coef'] = rdc1['coef'].abs()
cols_list = list(rdc1[rdc1['abs_coef'] > 0]['IV'])
X_train_mini = X_train[cols_list]
X2 = sm.add_constant(X_train_mini)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())
# to convert output into standard df:
table = (est2.summary().tables[1])
table = pd.DataFrame(table)
table.columns = table.iloc[0, :]
table = table.iloc[1:, :]


# STANDARD OLS ON ALL IVs
X2 = sm.add_constant(X_train)
esta = sm.OLS(y_train, X2)
esta2 = esta.fit()
print(esta2.summary())
