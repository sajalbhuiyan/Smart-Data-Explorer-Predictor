# ============================================
# Smart Data Explorer & Predictor (Pro)
# Fully upgraded: dashboard, EDA, ML, explainability, stats
# ============================================

# --------------------
# 1) Imports
# --------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_classif
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import warnings
import os
import io
import base64
import joblib

warnings.filterwarnings("ignore")

# --------------------
# 2) App Title
# --------------------
st.title("📊 Smart Data Explorer & Predictor (Pro)")

# --------------------
# 3) CSV Upload
# --------------------
uploaded_file = st.sidebar.file_uploader("Upload Your CSV", type=["csv"]) 
SAMPLE_PATH = "sample_data.csv"

# Standard UX features
MAX_UPLOAD_MB = 5


def safe_encode(df):
    df2 = df.copy()
    for col in df2.select_dtypes(include=['object', 'category']).columns:
        try:
            le = LabelEncoder()
            df2[col] = le.fit_transform(df2[col].astype(str))
        except:
            df2[col] = df2[col].astype(str)
    return df2

use_sample = False
if uploaded_file is None:
    if st.sidebar.checkbox("Use sample dataset (demo)"):
        use_sample = True
        uploaded_file = SAMPLE_PATH

if uploaded_file is not None:
    # If uploaded_file is an UploadedFile (from Streamlit), check size
    try:
        size_bytes = uploaded_file.size
    except Exception:
        # when uploaded_file is a path (sample), no size attribute
        size_bytes = None

    if size_bytes and size_bytes > MAX_UPLOAD_MB * 1024 * 1024:
        if not st.sidebar.checkbox(f"Confirm load file > {MAX_UPLOAD_MB} MB"):
            st.warning(f"File is larger than {MAX_UPLOAD_MB} MB. Check the box to confirm loading.")
            st.stop()

    # Read dataframe (works for UploadedFile or path)
    if isinstance(uploaded_file, str):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    # small dataset preview control
    preview_n = st.sidebar.number_input("Preview rows", min_value=3, max_value=500, value=10)

    # -------------------- Tabs --------------------
    tabs = st.tabs(["Overview", "EDA & Dashboard", "Feature Selection", 
                    "Modeling & Explainability", "Inferential Stats", "Outlier Detection", "Reports & Export"])

    # ---------- Overview
    with tabs[0]:
        st.subheader("Dataset Overview")
        st.dataframe(df.head(preview_n))
        st.write("Shape:", df.shape)
        st.write("Columns:", list(df.columns))
        st.write("Missing Values:\n", df.isnull().sum())
        if st.sidebar.checkbox("Drop columns with >90% missing values"):
            thresh = int(0.9 * df.shape[0])
            drop_cols = df.columns[df.isnull().sum() > thresh].tolist()
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)
                st.info(f"Dropped columns: {drop_cols}")
            else:
                st.info("No columns exceed 90% missing values")

        if st.sidebar.checkbox("Fill missing values"):
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "", inplace=True)
        if st.sidebar.checkbox("Encode categorical variables"):
            df = safe_encode(df)

    # Ensure models directory exists for persistence
    if not os.path.exists('models'):
        os.makedirs('models')

    # ---------- EDA & Dashboard
    with tabs[1]:
        st.subheader("Exploratory Data Analysis & Dashboard")
        st.dataframe(df.describe(include='all').T)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            if numeric_cols:
                num = st.selectbox("Select column (for simple plots)", numeric_cols, key='hist')
                plot_type = st.selectbox("Plot type", ["Histogram", "Box", "Violin", "KDE (density)", "Line"], key='plot_type')
                if plot_type == "Histogram":
                    fig = px.histogram(df, x=num, nbins=30, marginal='box', title=f'Histogram & Boxplot - {num}')
                    st.plotly_chart(fig)
                elif plot_type == "Box":
                    fig = px.box(df, y=num, title=f'Boxplot - {num}')
                    st.plotly_chart(fig)
                elif plot_type == "Violin":
                    fig = px.violin(df, y=num, box=True, points='all', title=f'Violin - {num}')
                    st.plotly_chart(fig)
                elif plot_type == "KDE (density)":
                    try:
                        sns.kdeplot(data=df[num].dropna())
                        st.pyplot(plt)
                        plt.clf()
                    except Exception:
                        st.info("KDE plot failed — ensure numeric column has enough unique values")
                elif plot_type == "Line":
                    fig = px.line(df.reset_index(), x='index', y=num, title=f'Line plot - {num}')
                    st.plotly_chart(fig)
        with col2:
            if numeric_cols and len(numeric_cols) >= 2:
                x_col = st.selectbox("Scatter: x", numeric_cols, key='xcol')
                y_col = st.selectbox("Scatter: y", numeric_cols, index=1, key='ycol')
                color_opt = st.selectbox("Color by (optional)", [None] + numeric_cols + categorical_cols, key='color')
                scatter_type = st.selectbox("Scatter type", ["2D Scatter", "3D Scatter", "Pairplot"], key='scatter_type')
                if scatter_type == "2D Scatter":
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_opt) if color_opt else px.scatter(df, x=x_col, y=y_col)
                    st.plotly_chart(fig)
                elif scatter_type == "3D Scatter":
                    if len(numeric_cols) >= 3:
                        z_col = st.selectbox("Z axis (3D)", [c for c in numeric_cols if c not in [x_col, y_col]], key='zcol')
                        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_opt)
                        st.plotly_chart(fig)
                    else:
                        st.info("Need at least 3 numeric columns for 3D scatter")
                elif scatter_type == "Pairplot":
                    try:
                        sns.pairplot(df[numeric_cols].dropna().sample(min(200, len(df))))
                        st.pyplot(plt)
                        plt.clf()
                    except Exception:
                        st.info("Pairplot failed or dataset too large — try sampling fewer rows")

        if len(numeric_cols) >= 2:
            st.write("Correlation matrix")
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto='.2f', title='Correlation Heatmap')
            st.plotly_chart(fig)

        if st.checkbox("Show PCA (2 components)"):
            if len(numeric_cols) >= 2:
                pca = PCA(n_components=2)
                comps = pca.fit_transform(df[numeric_cols].fillna(0))
                df_pca = pd.DataFrame(comps, columns=['PC1', 'PC2'])
                fig = px.scatter(df_pca, x='PC1', y='PC2', title='PCA (2 components)')
                st.plotly_chart(fig)
            else:
                st.info("Need at least 2 numeric features for PCA")

        if st.checkbox("KMeans Clustering (preview)"):
            if len(numeric_cols) >= 2 and df.shape[0] >= 3:
                k = st.slider("Number of clusters", 2, min(10, max(2, df.shape[0]//2)), 3)
                km = KMeans(n_clusters=k, random_state=42)
                df['Cluster'] = km.fit_predict(df[numeric_cols])
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color='Cluster', title='KMeans Clusters')
                st.plotly_chart(fig)
            else:
                st.info("Need at least 2 numeric features and 3 rows for KMeans")

    # ---------- Feature Selection
    with tabs[2]:
        st.subheader("Feature Selection")
        target_fs = st.selectbox("Select target", df.columns, key='fs_target')
        method = st.selectbox("Method", ["Univariate (f-test)", "Mutual Information"], key='fs_method')
        k = st.slider("Top features", 1, min(20, df.shape[1]-1), 5)

        X_fs = df.drop(columns=[target_fs])
        y_fs = df[target_fs]
        X_fs_encoded = safe_encode(X_fs).fillna(0)

        if st.button("Run feature selection"):
            try:
                if pd.api.types.is_numeric_dtype(y_fs):
                    selector = SelectKBest(score_func=f_regression if method.startswith("Univariate") else mutual_info_classif, k=min(k, X_fs_encoded.shape[1]))
                else:
                    selector = SelectKBest(score_func=f_classif if method.startswith("Univariate") else mutual_info_classif, k=min(k, X_fs_encoded.shape[1]))
                selector.fit(X_fs_encoded, safe_encode(pd.DataFrame(y_fs)).iloc[:,0])
                feature_scores = pd.DataFrame({'feature': X_fs_encoded.columns, 'score': selector.scores_}).sort_values('score', ascending=False).head(k)
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

        task_type = "Regression" if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10 else "Classification"
        st.write(f"Detected task type: {task_type}")

        X_encoded = safe_encode(X)
        numeric_features = X_encoded.select_dtypes(include=np.number).columns.tolist()
        scaler = StandardScaler()
        if numeric_features:
            X_encoded[numeric_features] = scaler.fit_transform(X_encoded[numeric_features].fillna(0))

        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        model_options = []
        if task_type == "Regression":
            model_options = ["Linear Regression", "Random Forest", "Decision Tree", "XGBoost", "KNN", "SVR"]
        else:
            model_options = ["Logistic Regression", "Random Forest", "Decision Tree", "XGBoost", "KNN", "SVM", "Naive Bayes"]

        chosen = st.multiselect("Select models to train", model_options, default=[model_options[0], model_options[1]])

        # Hyperparameter tuning option
        tune = st.checkbox("Enable hyperparameter tuning (small grids, can be slow)")
        param_grids = {}
        if tune:
            st.write("Tuning enabled — small default grids will be used for supported models")
            # small, conservative grids
            param_grids = {
                'Random Forest': {'n_estimators': [50, 100], 'max_depth': [None, 5]},
                'Decision Tree': {'max_depth': [None, 5, 10]},
                'KNN': {'n_neighbors': [3,5]},
                'SVM': {'C': [0.1, 1.0]},
                'XGBoost': {'n_estimators': [50, 100], 'max_depth': [3, 6]}
            }
            st.write("Available grids: ", list(param_grids.keys()))

        if st.button("Train & Compare Models"):
            models = {}
            if task_type == "Regression":
                for m in chosen:
                    if m == "Linear Regression":
                        models[m] = LinearRegression()
                    elif m == "Random Forest":
                        models[m] = RandomForestRegressor(random_state=42)
                    elif m == "Decision Tree":
                        models[m] = DecisionTreeRegressor(random_state=42)
                    elif m == "XGBoost":
                        models[m] = xgb.XGBRegressor(random_state=42)
                    elif m == "KNN":
                        models[m] = KNeighborsRegressor()
                    elif m == "SVR":
                        models[m] = SVR()
            else:
                for m in chosen:
                    if m == "Logistic Regression":
                        models[m] = LogisticRegression(max_iter=1000)
                    elif m == "Random Forest":
                        models[m] = RandomForestClassifier(random_state=42)
                    elif m == "Decision Tree":
                        models[m] = DecisionTreeClassifier(random_state=42)
                    elif m == "XGBoost":
                        models[m] = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
                    elif m == "KNN":
                        models[m] = KNeighborsClassifier()
                    elif m == "SVM":
                        models[m] = SVC(probability=True)
                    elif m == "Naive Bayes":
                        models[m] = GaussianNB()

            results = []
            fitted_models = {}
            preds = None
            progress = st.progress(0)
            total = len(models)
            cm_img_b64 = None
            roc_img_b64 = None
            for i, (name, model) in enumerate(models.items()):
                try:
                    if tune and name in param_grids:
                        grid = GridSearchCV(model, param_grids[name], cv=3, n_jobs=1)
                        grid.fit(X_train, y_train)
                        best = grid.best_estimator_
                        fitted_models[name] = best
                        model_used = best
                    else:
                        model.fit(X_train, y_train)
                        fitted_models[name] = model
                        model_used = model

                    y_pred = model_used.predict(X_test)
                    if task_type == "Regression":
                        results.append([name, r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred, squared=False)])
                    else:
                        results.append([name, accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted')])
                        preds = y_pred
                        # capture confusion matrix and ROC for the last classifier
                        try:
                            cm = confusion_matrix(y_test, preds)
                            fig_cm = px.imshow(cm, text_auto=True, title='Confusion Matrix')
                            cm_img = fig_cm.to_image(format='png')
                            cm_img_b64 = base64.b64encode(cm_img).decode()
                        except Exception:
                            cm_img_b64 = None
                        try:
                            if hasattr(model_used, 'predict_proba'):
                                y_proba = model_used.predict_proba(X_test)
                                if y_proba.shape[1] == 2:
                                    fpr, tpr, _ = roc_curve(y_test, y_proba[:,1])
                                    fig_roc = go.Figure()
                                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
                                    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash')))
                                    fig_roc.update_layout(title='ROC Curve', xaxis_title='FPR', yaxis_title='TPR')
                                    roc_img = fig_roc.to_image(format='png')
                                    roc_img_b64 = base64.b64encode(roc_img).decode()
                        except Exception:
                            roc_img_b64 = None
                except Exception as e:
                    results.append([name, str(e), ""])
                progress.update(int(((i+1)/total) * 100))

            results_df = pd.DataFrame(results, columns=["Model", "Metric1", "Metric2"])
            st.subheader("Model Comparison")
            st.dataframe(results_df)

            # --- Model persistence: save/load
            st.markdown("**Model persistence**")
            if fitted_models:
                default_name = "model_latest.pkl"
                save_name = st.text_input("Save last trained model as (filename .pkl)", value=default_name)
                if st.button("Save last trained model"):
                    try:
                        last_model = list(fitted_models.values())[-1]
                        save_path = os.path.join('models', save_name)
                        joblib.dump(last_model, save_path)
                        st.success(f"Saved model to {save_path}")
                    except Exception as e:
                        st.error(f"Failed to save model: {e}")

            # Load a saved model and run predictions
            saved_models = [f for f in os.listdir('models') if f.endswith('.pkl')]
            if saved_models:
                chosen_model = st.selectbox("Choose saved model to load", ["-- none --"] + saved_models)
                if chosen_model and chosen_model != "-- none --":
                    if st.button("Load selected model"):
                        try:
                            model_obj = joblib.load(os.path.join('models', chosen_model))
                            st.success(f"Loaded model {chosen_model}")
                            if st.button("Predict with loaded model"):
                                try:
                                    y_pred = model_obj.predict(X_test)
                                    df_pred = X_test.copy()
                                    df_pred['Predicted'] = y_pred
                                    st.dataframe(df_pred.head())
                                except Exception as e:
                                    st.error(f"Prediction failed: {e}")
                        except Exception as e:
                            st.error(f"Failed to load model: {e}")

            # --- Export simple HTML report (with embedded images when available)
            if st.button("Export HTML report"):
                try:
                    html_parts = [f"<h1>Smart Data Explorer Report</h1>"]
                    html_parts.append(f"<h2>Dataset: {uploaded_file}</h2>")
                    html_parts.append("<h3>Summary statistics</h3>")
                    html_parts.append(df.describe(include='all').to_html())

                    # correlation heatmap image
                    try:
                        if len(numeric_cols) >= 2:
                            corr = df[numeric_cols].corr()
                            fig_corr = px.imshow(corr, text_auto='.2f', title='Correlation Heatmap')
                            img_bytes = fig_corr.to_image(format='png')
                            b64 = base64.b64encode(img_bytes).decode()
                            html_parts.append(f"<h3>Correlation heatmap</h3><img src='data:image/png;base64,{b64}' />")
                    except Exception:
                        pass

                    # histogram image (first numeric)
                    try:
                        if numeric_cols:
                            fig_hist = px.histogram(df, x=numeric_cols[0], nbins=30)
                            img_bytes = fig_hist.to_image(format='png')
                            b64 = base64.b64encode(img_bytes).decode()
                            html_parts.append(f"<h3>Histogram: {numeric_cols[0]}</h3><img src='data:image/png;base64,{b64}' />")
                    except Exception:
                        pass

                    html_parts.append("<h3>Model comparison</h3>")
                    html_parts.append(results_df.to_html(index=False))

                    # embedded confusion matrix and ROC (from last training)
                    try:
                        if 'cm_img_b64' in locals() and cm_img_b64:
                            html_parts.append(f"<h3>Confusion Matrix</h3><img src='data:image/png;base64,{cm_img_b64}' />")
                    except Exception:
                        pass
                    try:
                        if 'roc_img_b64' in locals() and roc_img_b64:
                            html_parts.append(f"<h3>ROC Curve</h3><img src='data:image/png;base64,{roc_img_b64}' />")
                    except Exception:
                        pass

                    html = "\n".join(html_parts)
                    b = html.encode('utf-8')
                    href = "data:text/html;base64," + base64.b64encode(b).decode()
                    st.markdown(f"[Download report HTML]({href})")
                except Exception as e:
                    st.error(f"Report export failed: {e}")

            if st.checkbox("Show SHAP / Feature Importance"):
                if "Random Forest" in fitted_models:
                    rf = fitted_models["Random Forest"]
                    try:
                        explainer = shap.TreeExplainer(rf)
                        shap_values = explainer.shap_values(X_train)
                        plt.figure(figsize=(8,6))
                        shap.summary_plot(shap_values, X_train, show=False)
                        st.pyplot(plt)
                    except:
                        st.info("SHAP visualization failed.")

            if task_type == "Classification" and preds is not None:
                st.subheader("Classification Results")
                cm = confusion_matrix(y_test, preds)
                fig = px.imshow(cm, text_auto=True, title='Confusion Matrix')
                st.plotly_chart(fig)

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
                    except:
                        pass

            if st.button("Download predictions (last trained model)"):
                last_model = list(fitted_models.values())[-1]
                y_pred = last_model.predict(X_test)
                df_pred = X_test.copy()
                df_pred['Actual'] = y_test
                df_pred['Predicted'] = y_pred
                st.download_button("Download Predictions CSV", df_pred.to_csv(index=False), "predictions.csv")

    # ---------- Inferential Statistics
    with tabs[4]:
        st.subheader("Inferential Statistics & Tests")
        cols = st.multiselect("Select up to 2 columns", df.columns.tolist(), max_selections=2)
        if len(cols) == 2:
            a, b = df[cols[0]].dropna(), df[cols[1]].dropna()
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                corr, p = stats.pearsonr(a, b)
                st.write(f"Pearson r={corr:.3f}, p={p:.3g}")
                corr, p = stats.spearmanr(a, b)
                st.write(f"Spearman rho={corr:.3f}, p={p:.3g}")
            elif (not pd.api.types.is_numeric_dtype(a)) and (not pd.api.types.is_numeric_dtype(b)):
                ct = pd.crosstab(a, b)
                chi2, p, dof, ex = stats.chi2_contingency(ct)
                st.write(f"Chi2={chi2:.3f}, p={p:.3g}, dof={dof}")
            else:
                num, cat = (a, b.astype(str)) if pd.api.types.is_numeric_dtype(a) else (b, a.astype(str))
                groups = [num[cat==lvl] for lvl in cat.unique()]
                if len(groups)==2:
                    tstat, p = stats.ttest_ind(groups[0], groups[1])
                    st.write(f"T-test: t={tstat:.3f}, p={p:.3g}")
                elif len(groups)>2:
                    fstat, p = stats.f_oneway(*groups)
                    st.write(f"ANOVA: F={fstat:.3f}, p={p:.3g}")
                else:
                    st.info("Not enough group levels")

    # ---------- Outlier Detection
    with tabs[5]:
        st.subheader("Outlier Detection")
        if numeric_cols:
            col = st.selectbox("Select numeric column", numeric_cols)
            z_thresh = st.slider("Z-score threshold", 2.0, 5.0, 3.0)
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = df[col].dropna()[z_scores > z_thresh]
            st.write(f"Detected {len(outliers)} outliers")
            st.dataframe(outliers)

    # ---------- Reports & Export
    with tabs[6]:
        st.subheader("Dataset & Report Export")
        if st.button("Download cleaned dataset"):
            st.download_button("Download CSV", df.to_csv(index=False), "cleaned_dataset.csv")
        st.info("All predictions can also be downloaded from Modeling tab.")

else:
    st.info("Upload a CSV file to get started!")
