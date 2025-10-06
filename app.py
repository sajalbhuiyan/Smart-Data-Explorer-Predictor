# ============================================
# Smart Data Explorer & Predictor (Advanced)
# Enhanced: dashboard, EDA, feature selection, inferential stats
# ============================================

# --------------------
# 1) Imports
# --------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_classif
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# --------------------
# 2) App Title
# --------------------
st.title("📊 Smart Data Explorer & Predictor (Advanced)")

# --------------------
# 3) Upload CSV
# --------------------
uploaded_file = st.sidebar.file_uploader("Upload Your CSV", type=["csv"]) 

def safe_encode(df):
    """Return a copy where object dtypes are label-encoded (for algorithms that need numeric inputs)."""
    df2 = df.copy()
    for col in df2.select_dtypes(include=['object', 'category']).columns:
        try:
            le = LabelEncoder()
            df2[col] = le.fit_transform(df2[col].astype(str))
        except Exception:
            df2[col] = df2[col].astype(str)
    return df2


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Basic dataset overview tabbed layout
    tabs = st.tabs(["Overview", "EDA & Dashboard", "Feature Selection", "Modeling & Explainability", "Inferential Stats"])

    # ---------- Overview
    with tabs[0]:
        st.subheader("Dataset Overview")
        st.dataframe(df.head())
        st.write("Shape:", df.shape)
        st.write("Columns:", list(df.columns))
        st.write("Missing Values:\n", df.isnull().sum())
        if st.sidebar.checkbox("Fill missing values"):
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "", inplace=True)
        if st.sidebar.checkbox("Encode categorical variables"):
            df = safe_encode(df)

    # ---------- EDA & Dashboard
    with tabs[1]:
        st.subheader("Exploratory Data Analysis & Dashboard")
        st.write("Summary Statistics")
        st.dataframe(df.describe(include='all').T)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            if numeric_cols:
                num = st.selectbox("Histogram: select numeric column", numeric_cols, key='hist')
                fig = px.histogram(df, x=num, nbins=30, marginal='box', title=f'Histogram & boxplot - {num}')
                st.plotly_chart(fig)
        with col2:
            if numeric_cols and len(numeric_cols) >= 2:
                x_col = st.selectbox("Scatter: x", numeric_cols, key='xcol')
                y_col = st.selectbox("Scatter: y", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key='ycol')
                color_opt = st.selectbox("Color by (optional)", [None] + numeric_cols + categorical_cols, key='color')
                fig = px.scatter(df, x=x_col, y=y_col, color=color_opt) if color_opt else px.scatter(df, x=x_col, y=y_col)
                st.plotly_chart(fig)

        # Correlation heatmap
        if len(numeric_cols) >= 2:
            st.write("Correlation matrix")
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto='.2f', title='Correlation Heatmap')
            st.plotly_chart(fig)

        # PCA preview if enough numeric cols
        if st.checkbox("Show PCA (2 components)"):
            df_numeric = df.select_dtypes(include=np.number)
            if df_numeric.shape[1] >= 2:
                pca = PCA(n_components=2)
                comps = pca.fit_transform(df_numeric.fillna(0))
                df_pca = pd.DataFrame(comps, columns=['PC1', 'PC2'])
                fig = px.scatter(df_pca, x='PC1', y='PC2', title='PCA (2 components)')
                st.plotly_chart(fig)
            else:
                st.info("Need at least 2 numeric features for PCA")

        # KMeans preview
        if st.checkbox("KMeans Clustering (preview)"):
            df_numeric = df.select_dtypes(include=np.number).dropna()
            if df_numeric.shape[1] >= 2 and df_numeric.shape[0] >= 3:
                k = st.slider("Number of clusters", 2, min(10, max(2, df_numeric.shape[0]//2)), 3)
                km = KMeans(n_clusters=k, random_state=42)
                df['Cluster'] = km.fit_predict(df_numeric)
                fig = px.scatter(df, x=df_numeric.columns[0], y=df_numeric.columns[1], color='Cluster', title='KMeans Clusters')
                st.plotly_chart(fig)
            else:
                st.info("Need at least 2 numeric features and 3 rows for KMeans")

    # ---------- Feature Selection
    with tabs[2]:
        st.subheader("Feature Selection")
        st.write("Choose a target column and a selection method")
        target_fs = st.selectbox("Select target for feature selection", df.columns, key='fs_target')
        method = st.selectbox("Method", ["Univariate (f-test)", "Mutual Information"], key='fs_method')
        k = st.slider("Number of top features", 1, max(1, min(20, max(1, df.shape[1]-1))), 5)

        X_fs = df.drop(columns=[target_fs])
        y_fs = df[target_fs]
        X_fs_encoded = safe_encode(X_fs).fillna(0)

        if st.button("Run feature selection"):
            try:
                if pd.api.types.is_numeric_dtype(y_fs):
                    if method.startswith("Univariate"):
                        selector = SelectKBest(score_func=f_regression, k=min(k, X_fs_encoded.shape[1]))
                    else:
                        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X_fs_encoded.shape[1]))
                else:
                    if method.startswith("Univariate"):
                        selector = SelectKBest(score_func=f_classif, k=min(k, X_fs_encoded.shape[1]))
                    else:
                        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X_fs_encoded.shape[1]))

                selector.fit(X_fs_encoded, safe_encode(pd.DataFrame(y_fs)).iloc[:,0])
                scores = selector.scores_
                feature_scores = pd.DataFrame({'feature': X_fs_encoded.columns, 'score': scores}).sort_values('score', ascending=False).head(k)
                st.dataframe(feature_scores)
                fig = px.bar(feature_scores, x='feature', y='score', title='Top features')
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Feature selection failed: {e}")

    # ---------- Modeling & Explainability
    with tabs[3]:
        st.subheader("Modeling & Explainability")
        target = st.selectbox("Select Target Column", df.columns, key='model_target')
        X = df.drop(columns=[target])
        y = df[target]

        # Detect task
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
            task_type = "Regression"
        else:
            task_type = "Classification"
        st.write(f"Detected task type: {task_type}")

        # Preprocessing: encode categorical and scale numeric
        X_encoded = safe_encode(X)
        numeric_features = X_encoded.select_dtypes(include=np.number).columns.tolist()
        scaler = StandardScaler()
        if numeric_features:
            X_encoded[numeric_features] = scaler.fit_transform(X_encoded[numeric_features].fillna(0))

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        if st.button("Train & Compare Models"):
            models = {}
            if task_type == "Regression":
                models = {"Linear Regression": LinearRegression(), "Random Forest": RandomForestRegressor(random_state=42)}
            else:
                models = {"Logistic Regression": LogisticRegression(max_iter=1000), "Random Forest": RandomForestClassifier(random_state=42)}

            results = []
            fitted_models = {}
            preds = None
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    fitted_models[name] = model
                    y_pred = model.predict(X_test)
                    if task_type == "Regression":
                        results.append([name, r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred, squared=False)])
                    else:
                        results.append([name, accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted')])
                        preds = y_pred
                except Exception as e:
                    results.append([name, str(e), ""]) 

            results_df = pd.DataFrame(results, columns=["Model", "Metric1", "Metric2"])
            st.subheader("Model Comparison")
            st.dataframe(results_df)

            # Use RandomForest if available for feature importance / SHAP
            if st.checkbox("Show Feature Importance / SHAP"):
                if "Random Forest" in fitted_models:
                    rf = fitted_models["Random Forest"]
                    try:
                        explainer = shap.TreeExplainer(rf)
                        shap_values = explainer.shap_values(X_train)
                        st.write("SHAP summary plot:")
                        plt.figure(figsize=(8,6))
                        try:
                            shap.summary_plot(shap_values, X_train, show=False)
                        except Exception:
                            # fallback for newer shap versions
                            shap.summary_plot(shap_values, features=X_train, show=False)
                        st.pyplot(plt)
                    except Exception as e:
                        st.error(f"SHAP failed: {e}")
                else:
                    st.info("Random Forest was not trained or failed. Feature importance unavailable.")

            # Post-train outputs: classification reports, ROC, download
            if task_type == "Classification" and preds is not None:
                st.subheader("Classification results")
                st.write("Confusion Matrix")
                cm = confusion_matrix(y_test, preds)
                fig = px.imshow(cm, text_auto=True, title='Confusion Matrix')
                st.plotly_chart(fig)
                st.write("Classification metrics")
                st.text(pd.Series({
                    'accuracy': accuracy_score(y_test, preds),
                    'f1_weighted': f1_score(y_test, preds, average='weighted')
                }))

                # ROC curve if probability available
                if hasattr(list(fitted_models.values())[0], 'predict_proba'):
                    rf = list(fitted_models.values())[-1]
                    try:
                        y_proba = rf.predict_proba(X_test)
                        if y_proba.shape[1] == 2:
                            fpr, tpr, _ = roc_curve(y_test, y_proba[:,1])
                            roc_auc = auc(fpr, tpr)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC={roc_auc:.3f}'))
                            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash')))
                            fig.update_layout(title='ROC Curve', xaxis_title='FPR', yaxis_title='TPR')
                            st.plotly_chart(fig)
                    except Exception:
                        pass

            # Download predictions (for last model trained)
            download_btn = st.button("Download predictions (last trained model)")
            if download_btn and fitted_models:
                last_model = list(fitted_models.values())[-1]
                y_pred = last_model.predict(X_test)
                df_pred = X_test.copy()
                df_pred['Actual'] = y_test
                df_pred['Predicted'] = y_pred
                csv = df_pred.to_csv(index=False)
                st.download_button("Download Predictions CSV", csv, "predictions.csv")

    # ---------- Inferential Statistics
    with tabs[4]:
        st.subheader("Inferential Statistics & Tests")
        st.write("Select two columns to run quick statistical tests")
        cols = st.multiselect("Select up to 2 columns", df.columns.tolist(), max_selections=2)
        if len(cols) == 2:
            a = df[cols[0]].dropna()
            b = df[cols[1]].dropna()
            # numeric-numeric: Pearson and Spearman
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                st.write("Pearson correlation")
                try:
                    corr, p = stats.pearsonr(a, b)
                    st.write(f"Pearson r={corr:.3f}, p={p:.3g}")
                except Exception as e:
                    st.error(f"Pearson failed: {e}")
                st.write("Spearman correlation")
                try:
                    corr, p = stats.spearmanr(a, b)
                    st.write(f"Spearman rho={corr:.3f}, p={p:.3g}")
                except Exception as e:
                    st.error(f"Spearman failed: {e}")
            # categorical-categorical: chi-square
            elif (not pd.api.types.is_numeric_dtype(a)) and (not pd.api.types.is_numeric_dtype(b)):
                st.write("Chi-square test of independence")
                try:
                    ct = pd.crosstab(a, b)
                    chi2, p, dof, ex = stats.chi2_contingency(ct)
                    st.write(f"chi2={chi2:.3f}, p={p:.3g}, dof={dof}")
                except Exception as e:
                    st.error(f"Chi-square failed: {e}")
            # numeric vs categorical: t-test / ANOVA
            else:
                if pd.api.types.is_numeric_dtype(a):
                    num = a
                    cat = b.astype(str)
                else:
                    num = b
                    cat = a.astype(str)
                groups = [num[cat == lvl] for lvl in cat.unique()]
                if len(groups) == 2:
                    tstat, p = stats.ttest_ind(groups[0].dropna(), groups[1].dropna())
                    st.write(f"t-test (2 groups): t={tstat:.3f}, p={p:.3g}")
                elif len(groups) > 2:
                    fstat, p = stats.f_oneway(*[g.dropna() for g in groups])
                    st.write(f"ANOVA: F={fstat:.3f}, p={p:.3g}")
                else:
                    st.info("Not enough group levels for this test")

else:
    st.info("Upload a CSV file to get started!")
