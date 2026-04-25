"""
Smart Disease Prediction System using AIML
===========================================
Components:
  1. Logistic Regression  — Predict disease (Yes/No)
  2. Apriori Algorithm    — Find symptom/feature patterns  [built-in, no extra install]
  3. Expert System        — Rule-based recommendations
"""

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 & 2: Dataset & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
n = 303

age      = np.random.randint(29, 78, n)
sex      = np.random.randint(0, 2, n)
cp       = np.random.randint(0, 4, n)
trestbps = np.random.randint(94, 200, n)
chol     = np.random.randint(126, 564, n)
fbs      = (chol > 200).astype(int)
restecg  = np.random.randint(0, 3, n)
thalach  = np.random.randint(71, 202, n)
exang    = np.random.randint(0, 2, n)
oldpeak  = np.round(np.random.uniform(0, 6.2, n), 1)
slope    = np.random.randint(0, 3, n)
ca       = np.random.randint(0, 5, n)
thal     = np.random.randint(0, 4, n)

risk   = ((age>55).astype(int)+(chol>240).astype(int)+(trestbps>140).astype(int)
          +(cp>1).astype(int)+(exang==1).astype(int))
target = (risk >= 2).astype(int)
flip   = np.random.choice(n, size=int(n*0.08), replace=False)
target[flip] = 1 - target[flip]

df = pd.DataFrame({'age':age,'sex':sex,'cp':cp,'trestbps':trestbps,'chol':chol,
                   'fbs':fbs,'restecg':restecg,'thalach':thalach,'exang':exang,
                   'oldpeak':oldpeak,'slope':slope,'ca':ca,'thal':thal,'target':target})

print("="*58)
print("    SMART DISEASE PREDICTION SYSTEM — AIML PROJECT")
print("="*58)
print(f"\n[Dataset]  Rows:{df.shape[0]}  Cols:{df.shape[1]}")
print(df.head(5).to_string())
print(f"\nClass distribution:\n{df['target'].value_counts().rename({0:'No Disease',1:'Disease'}).to_string()}")

X = df.drop("target", axis=1)
y = df["target"]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train)
Xte = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(Xtr, y_train)
y_pred = model.predict(Xte)
acc    = accuracy_score(y_test, y_pred)

print("\n"+"─"*58)
print("  MODULE 1 — LOGISTIC REGRESSION")
print("─"*58)
print(f"  Accuracy : {acc*100:.2f}%")
print(f"\n  Classification Report:\n")
print(classification_report(y_test,y_pred,target_names=["No Disease","Disease"]))
cm = confusion_matrix(y_test, y_pred)
print(f"  Confusion Matrix:  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Apriori (built-in)
# ─────────────────────────────────────────────────────────────────────────────

print("\n"+"─"*58)
print("  MODULE 2 — APRIORI ASSOCIATION RULES")
print("─"*58)

df_ap = pd.DataFrame({
    'age_gt55':   (df['age']>55),
    'high_chol':  (df['chol']>240),
    'high_bp':    (df['trestbps']>140),
    'chest_pain': (df['cp']>1),
    'exang':      (df['exang']==1),
    'disease':    (df['target']==1),
}, dtype=bool)

def get_support(d, items):
    return d[list(items)].all(axis=1).sum() / len(d)

def apriori_mine(d, min_sup=0.15):
    cols = list(d.columns)
    freq = []
    L = [(frozenset([c]), get_support(d,[c])) for c in cols if get_support(d,[c])>=min_sup]
    freq.extend(L); prev = L; k = 2
    while prev:
        cands = {a|b for a,b in combinations([p[0] for p in prev],2) if len(a|b)==k}
        new   = [(c, get_support(d,list(c))) for c in cands if get_support(d,list(c))>=min_sup]
        freq.extend(new); prev = new; k += 1
    return freq

def gen_rules(freq_sets, min_conf=0.55):
    fd = {fs:s for fs,s in freq_sets}
    rules = []
    for fs,sup in freq_sets:
        if len(fs)<2: continue
        for sz in range(1,len(fs)):
            for ant in combinations(sorted(fs),sz):
                ant=frozenset(ant); cons=fs-ant
                as_ = fd.get(ant)
                if as_ and as_>0:
                    conf = sup/as_
                    if conf >= min_conf:
                        cs = fd.get(cons, get_support(df_ap,list(cons)))
                        lift = conf/cs if cs>0 else 0
                        rules.append((ant,cons,sup,conf,lift))
    return sorted(rules, key=lambda x:-x[3])

freq_sets = apriori_mine(df_ap, min_sup=0.15)
rules     = gen_rules(freq_sets, min_conf=0.55)

print(f"  Frequent itemsets : {len(freq_sets)}")
print(f"  Association rules : {len(rules)}")
print("\n  Top Rules:\n")
for ant,cons,sup,conf,lift in rules[:6]:
    print(f"  [{', '.join(sorted(ant))}]  ->  [{', '.join(sorted(cons))}]")
    print(f"    Support={sup:.2f}  Confidence={conf:.2f}  Lift={lift:.2f}\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Expert System
# ─────────────────────────────────────────────────────────────────────────────

print("─"*58)
print("  MODULE 3 — EXPERT SYSTEM")
print("─"*58)

def expert_recommendation(age, chol, bp, cp, exang_val):
    recs=[]; score=0
    if chol>240:  recs.append("HIGH CHOLESTEROL: low-fat diet, increase activity, consult physician."); score+=2
    if bp>140:    recs.append("HIGH BLOOD PRESSURE: reduce sodium, manage stress, monitor BP."); score+=2
    if age>55:    recs.append("AGE RISK: schedule annual cardiac screening."); score+=1
    if cp>1:      recs.append("CHEST PAIN ALERT: seek immediate medical evaluation."); score+=3
    if exang_val==1: recs.append("EXERCISE ANGINA: avoid high-intensity exercise without clearance."); score+=2
    if score==0:  recs.append("ALL CLEAR: maintain healthy diet, exercise, and routine check-ups.")
    label = "LOW RISK" if score<=1 else ("MODERATE RISK" if score<=3 else "HIGH RISK")
    return label, recs

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def predict_patient(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal):
    inp = pd.DataFrame([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]],
                        columns=X.columns)
    pred = model.predict(scaler.transform(inp))[0]
    prob = model.predict_proba(scaler.transform(inp))[0][1]
    label,recs = expert_recommendation(age,chol,trestbps,cp,exang)
    print(f"\n  Prediction  : {'DISEASE DETECTED' if pred==1 else 'NO DISEASE'}")
    print(f"  Probability : {prob*100:.1f}%")
    print(f"  Risk Level  : {label}")
    print("  Recommendations:")
    for r in recs: print(f"    • {r}")

print()
print("  [Sample A — 45yr, low-risk]")
predict_patient(45,0,0,120,200,0,0,160,0,0.5,1,0,1)
print()
print("  [Sample B — 65yr, high-risk]")
predict_patient(65,1,3,165,295,1,2,98,1,3.8,2,3,3)

print("\n"+"="*58)
print(f"  Final Model Accuracy : {acc*100:.2f}%")
print("="*58)
