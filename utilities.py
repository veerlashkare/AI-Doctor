import pandas as pd
import os, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score

RANDOM_STATE = 42

def load_tabular(path='data.csv', target_col='diagnosis'):
    """Load and preprocess the breast cancer dataset."""
    import pandas as pd
    import numpy as np

    df = pd.read_csv(path)

    # Drop useless or empty columns
    df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col or df[col].isna().all()], errors='ignore')

    # Normalize diagnosis column
    df[target_col] = df[target_col].astype(str).str.upper().str.strip()

    # Map diagnosis: M → 1, B → 0
    mapping = {'M': 1, 'B': 0}
    df['target'] = df[target_col].map(mapping)

    # Drop ID column if present
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Replace any inf/-inf and fill remaining NaN with median
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))

    print("✅ Data cleaned successfully!")
    print("Shape after cleaning:", df.shape)
    print("Remaining NaN count:", df.isna().sum().sum())

    return df

def train_and_save(df, outdir='models', save_name='ensemble_tabular.joblib'):
    """Train ensemble model and save it to disk."""
    os.makedirs(outdir, exist_ok=True)
    X = df.drop(columns=['diagnosis', 'target'], errors='ignore')
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    lr = Pipeline([('scaler', scaler),
                   ('clf', LogisticRegression(solver='liblinear'))])
    rf = Pipeline([('scaler', scaler),
                   ('clf', RandomForestClassifier(n_estimators=200,
                                                  random_state=RANDOM_STATE))])

    ensemble = VotingClassifier([('lr', lr), ('rf', rf)], voting='soft')
    ensemble.fit(X_train, y_train)

    preds = ensemble.predict(X_test)
    probs = ensemble.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, preds))
    print("AUC:", roc_auc_score(y_test, probs))

    joblib.dump({'model': ensemble}, os.path.join(outdir, save_name))
    print("✅ Model saved to", os.path.join(outdir, save_name))