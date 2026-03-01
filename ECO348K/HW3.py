import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")

# =========================
# Paths (as uploaded)
# =========================
LOAN_PATH = "loanapp.dta"
CHILD_PATH = "children_sample.dta"

loan = pd.read_stata(LOAN_PATH)
child = pd.read_stata(CHILD_PATH)

# ============================================================
# Problem 3: loanapp.dta
# ============================================================
print("\n=== Problem 3: loanapp.dta ===")

controls = [
    "hrat","obrat","loanprc","unem","male","married","dep","sch",
    "cosign","chist","pubrec","mortlat1","mortlat2","vr"
]

# (a) LPM approve on white (robust SE)
lpm1 = smf.ols("approve ~ white", data=loan).fit(cov_type="HC1")
print("\n(3a) LPM approve ~ white (robust HC1)")
print(lpm1.summary().tables[1])

p_nonwhite_lpm = lpm1.params["Intercept"]
p_white_lpm = lpm1.params["Intercept"] + lpm1.params["white"]
print(f"Predicted P(approve|nonwhite) = {p_nonwhite_lpm:.6f}")
print(f"Predicted P(approve|white)    = {p_white_lpm:.6f}")

# (b) Probit approve on white
probit1 = smf.probit("approve ~ white", data=loan).fit(disp=False)
b0, b1 = probit1.params["Intercept"], probit1.params["white"]
p_nonwhite_probit = st.norm.cdf(b0)
p_white_probit = st.norm.cdf(b0 + b1)

print("\n(3b) Probit approve ~ white")
print(probit1.summary().tables[1])
print(f"Predicted P(approve|nonwhite) = {p_nonwhite_probit:.6f}")
print(f"Predicted P(approve|white)    = {p_white_probit:.6f}")

# (c) LPM with controls (robust)
formula = "approve ~ white + " + " + ".join(controls)
lpm2 = smf.ols(formula, data=loan).fit(cov_type="HC1")
print("\n(3c) LPM approve ~ white + controls (robust HC1)")
print(lpm2.summary().tables[1])

print("\nKey coefficient (white):")
print(f"white coef = {lpm2.params['white']:.6f}, robust SE = {lpm2.bse['white']:.6f}, p = {lpm2.pvalues['white']:.3g}")

# (d) Probit with controls (robust)
probit2 = smf.probit(formula, data=loan).fit(disp=False, cov_type="HC1")
print("\n(3d) Probit approve ~ white + controls (robust HC1)")
print(probit2.summary().tables[1])

print("\nKey coefficient (white):")
print(f"white coef = {probit2.params['white']:.6f}, robust SE = {probit2.bse['white']:.6f}, p = {probit2.pvalues['white']:.3g}")

# Average marginal effects (overall), discrete change for dummy vars
margeff = probit2.get_margeff(at="overall", method="dydx", dummy=True)
print("\nAverage marginal effects (overall):")
print(margeff.summary())

# ============================================================
# Problem 4: children_sample.dta (keep if white & male)
# ============================================================
print("\n=== Problem 4: children_sample.dta ===")

sub = child[(child["white"] == 1) & (child["male"] == 1)].copy()
print(f"Subsample size (white & male): n = {len(sub)}")

# (a) tabstat-like summary
bmi = sub["bmi"]
stats = {
    "mean": bmi.mean(),
    "p10": bmi.quantile(0.10),
    "p25": bmi.quantile(0.25),
    "p50": bmi.quantile(0.50),
    "p75": bmi.quantile(0.75),
    "p90": bmi.quantile(0.90),
}
print("\n(4a) BMI summary (mean and quantiles):")
for k in ["mean","p10","p25","p50","p75","p90"]:
    print(f"{k}: {stats[k]:.6f}")

# (c) OLS with robust SE
formula4 = "bmi ~ educ + age + mombmi + dadbmi"
ols4 = smf.ols(formula4, data=sub).fit(cov_type="HC1")
print("\n(4c) OLS bmi ~ educ + age + mombmi + dadbmi (robust HC1)")
print(ols4.summary().tables[1])

# Quantile regression helper + bootstrap SE (sqreg-like)
X = sm.add_constant(sub[["educ","age","mombmi","dadbmi"]])
y = sub["bmi"]

def bootstrap_quantreg(df, q, reps=100, seed=0):
    rng = np.random.default_rng(seed)
    n = len(df)
    params = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        samp = df.iloc[idx]
        Xb = sm.add_constant(samp[["educ","age","mombmi","dadbmi"]])
        yb = samp["bmi"]
        fit = QuantReg(yb, Xb).fit(q=q, max_iter=2000)
        params.append(fit.params.values)
    params = np.array(params)
    se = params.std(axis=0, ddof=1)
    return se

# (d) Median regression (q=0.5) with bootstrap reps
qr50 = QuantReg(y, X).fit(q=0.5, max_iter=2000)
se50 = bootstrap_quantreg(sub, q=0.5, reps=100, seed=42)
print("\n(4d) Median regression (q=0.5) with 100 bootstrap reps")
print("Params:")
print(qr50.params)
print("Bootstrap SEs:")
print(pd.Series(se50, index=qr50.params.index))

# (e) Quantiles 0.10, 0.25, 0.50, 0.75, 0.90 with bootstrap reps
quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
results = []

print("\n(4e) Quantile regressions with 100 bootstrap reps each")
for q in quantiles:
    fit = QuantReg(y, X).fit(q=q, max_iter=2000)
    se = bootstrap_quantreg(sub, q=q, reps=100, seed=int(q*1000)+7)
    out = pd.DataFrame({
        "term": fit.params.index,
        "coef": fit.params.values,
        "boot_se": se
    })
    out["quantile"] = q
    results.append(out)

all_qr = pd.concat(results, ignore_index=True)
print(all_qr)

# Optional: compact slope-only table
slopes = all_qr[all_qr["term"].isin(["educ","age","mombmi","dadbmi"])].copy()
coef_table = slopes.pivot(index="term", columns="quantile", values="coef")
se_table = slopes.pivot(index="term", columns="quantile", values="boot_se")

print("\nSlope coefficients by quantile:")
print(coef_table)

print("\nBootstrap SEs by quantile:")
print(se_table)