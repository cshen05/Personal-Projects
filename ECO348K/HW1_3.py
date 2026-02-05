import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load
df = pd.read_stata("FERTIL2-1.DTA")

# Create missing indicator and fill
df["heducmissing"] = df["heduc"].isna().astype(int)
df["heduc"] = df["heduc"].fillna(0)  # replace missing with 0

def ols_hc1(data, y, xcols):
    X = sm.add_constant(data[xcols], has_constant="add")
    model = sm.OLS(data[y], X, missing="drop").fit(cov_type="HC1")
    return model

# 3(a)
m_a = ols_hc1(df, "children", ["age","agesq","educ","evermarr","heduc","heducmissing"])
print(m_a.summary())

# (a)(i) marginal effect of age at mean age (because age and agesq)
mean_age = df["age"].mean()
me_age = m_a.params["age"] + 2*m_a.params["agesq"]*mean_age
print("Mean age:", mean_age)
print("Marginal effect of age at mean age:", me_age)

# 3(b) separate regressions by evermarr
m_b_mar = ols_hc1(df[df["evermarr"]==1], "children", ["age","agesq","educ","heduc","heducmissing"])
m_b_unm = ols_hc1(df[df["evermarr"]==0], "children", ["age","agesq","educ","heduc","heducmissing"])
print(m_b_mar.summary())
print(m_b_unm.summary())

# 3(c)
df["electricity"] = df["electric"]
m_c = ols_hc1(df, "children", ["age","agesq","educ","electricity"])
print(m_c.summary())

# 3(d) interactions with age, agesq, educ
df["age_elec"]   = df["age"]   * df["electricity"]
df["agesq_elec"] = df["agesq"] * df["electricity"]
df["educ_elec"]  = df["educ"]  * df["electricity"]

m_d = ols_hc1(df, "children", ["age","agesq","educ","electricity","age_elec","agesq_elec","educ_elec"])
print(m_d.summary())

# Partial effect of electricity for each observation:
pe = (m_d.params["electricity"]
      + m_d.params["age_elec"]*df["age"]
      + m_d.params["agesq_elec"]*df["agesq"]
      + m_d.params["educ_elec"]*df["educ"])

print("APE (mean partial effect):", pe.mean())

plt.figure()
plt.hist(pe.dropna(), bins=40)
plt.title("Estimated in-sample partial effects of electricity")
plt.xlabel("Partial effect of electricity on children")
plt.ylabel("Count")
plt.show()

# 3(e) subsample regressions
m_e0 = ols_hc1(df[df["electricity"]==0], "children", ["age","agesq","educ"])
m_e1 = ols_hc1(df[df["electricity"]==1], "children", ["age","agesq","educ"])
print(m_e0.summary())
print(m_e1.summary())