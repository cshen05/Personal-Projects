import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_stata("voucher-1.dta")

# 4(a) counts
print("Never awarded (selectyrs=0):", (df["selectyrs"]==0).sum())
print("Voucher 4 years (selectyrs=4):", (df["selectyrs"]==4).sum())
print("Attended 4 years (choiceyrs=4):", (df["choiceyrs"]==4).sum())

def ols_hc1(data, y, xcols):
    X = sm.add_constant(data[xcols], has_constant="add")
    return sm.OLS(data[y], X, missing="drop").fit(cov_type="HC1")

def tsls_robust(data, y, endog, exog_list, instr):
    # drop missing relevant vars
    use = [y, endog, instr] + exog_list
    d = data[use].dropna()

    Y = d[y].to_numpy()
    X = sm.add_constant(d[[endog] + exog_list], has_constant="add").to_numpy()
    Z = sm.add_constant(d[[instr] + exog_list], has_constant="add").to_numpy()

    n, k = X.shape
    PZ = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    Xhat = PZ @ X
    beta = np.linalg.inv(Xhat.T @ X) @ (Xhat.T @ Y)
    resid = Y - X @ beta

    PZX = PZ @ X
    meat = PZX.T @ ((resid[:, None]**2) * PZX)
    meat *= n / (n - k)  # HC1-like correction
    bread_inv = np.linalg.inv(Xhat.T @ X)
    V = bread_inv @ meat @ bread_inv
    se = np.sqrt(np.diag(V))

    return beta, se, n

# 4(b) choiceyrs on selectyrs
m_b = ols_hc1(df, "choiceyrs", ["selectyrs"])
print(m_b.summary())

# 4(c) mnce on choiceyrs (simple), then add controls
m_c1 = ols_hc1(df, "mnce", ["choiceyrs"])
m_c2 = ols_hc1(df, "mnce", ["choiceyrs","black","hispanic","female"])
print(m_c1.summary())
print(m_c2.summary())

# 4(e) IV for (d) equation: mnce on choiceyrs + demographics, IV = selectyrs
beta_e, se_e, n_e = tsls_robust(df, "mnce", "choiceyrs",
                               ["black","hispanic","female"], "selectyrs")
print("IV beta (const, choiceyrs, black, hispanic, female):", beta_e)
print("IV se:", se_e, "n=", n_e)

# 4(f) add mnce90, compare OLS vs IV
m_f_ols = ols_hc1(df, "mnce", ["choiceyrs","black","hispanic","female","mnce90"])
beta_f, se_f, n_f = tsls_robust(df, "mnce", "choiceyrs",
                               ["black","hispanic","female","mnce90"], "selectyrs")
print(m_f_ols.summary())
print("IV with mnce90 beta:", beta_f)
print("IV with mnce90 se:", se_f, "n=", n_f)

# 4(h) multiple endog dummies IV with selectyrs dummies
def tsls_multi_robust(data, y, endog_list, exog_list, instr_list):
    use = [y] + endog_list + exog_list + instr_list
    d = data[use].dropna()

    Y = d[y].to_numpy()
    X = sm.add_constant(d[endog_list + exog_list], has_constant="add").to_numpy()
    Z = sm.add_constant(d[instr_list + exog_list], has_constant="add").to_numpy()

    n, k = X.shape
    PZ = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    Xhat = PZ @ X
    beta = np.linalg.inv(Xhat.T @ X) @ (Xhat.T @ Y)
    resid = Y - X @ beta

    PZX = PZ @ X
    meat = PZX.T @ ((resid[:, None]**2) * PZX)
    meat *= n / (n - k)
    bread_inv = np.linalg.inv(Xhat.T @ X)
    V = bread_inv @ meat @ bread_inv
    se = np.sqrt(np.diag(V))
    return beta, se, n

beta_h, se_h, n_h = tsls_multi_robust(
    df, "mnce",
    ["choiceyrs1","choiceyrs2","choiceyrs3","choiceyrs4"],
    ["black","hispanic","female"],
    ["selectyrs1","selectyrs2","selectyrs3","selectyrs4"]
)
print("IV (h) beta:", beta_h)
print("IV (h) se:", se_h, "n=", n_h)