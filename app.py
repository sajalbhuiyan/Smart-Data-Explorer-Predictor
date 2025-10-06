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
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_classif
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, roc_curve, auc, confusion_matrix, precision_score, recall_score, classification_report
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
    # Ensure exports directory exists for server-side exports
    if not os.path.exists('exports'):
        os.makedirs('exports')

    # ---------- EDA & Dashboard (enhanced)
    with tabs[1]:
        st.subheader("Exploratory Data Analysis & Dashboard (All plots)")
        st.dataframe(df.describe(include='all').T)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        st.markdown("Select plot type and configure columns/appearance. Large datasets are auto-sampled for heavy plots.")

        # universal sampling safeguard
        sample_mode = st.checkbox('Auto-sample large datasets for plotting (recommended)', value=True)
        sample_n = st.number_input('Sample size (when sampling enabled)', min_value=100, max_value=100000, value=1000)
        def _sample_df(df_in):
            if not sample_mode:
                return df_in
            n = min(len(df_in), int(sample_n))
            if len(df_in) > n:
                return df_in.sample(n, random_state=42)
            return df_in

        # comprehensive plot selector
        plot_group = st.selectbox('Plot category', ['Univariate (numeric)', 'Univariate (categorical)', 'Bivariate numeric', 'Bivariate categorical', 'Multivariate', 'Time series', 'Distribution / Density', 'Advanced (Pairplot, PCA, Clustering)'])

        if plot_group == 'Univariate (numeric)':
            if not numeric_cols:
                st.info('No numeric columns available')
            else:
                col = st.selectbox('Numeric column', numeric_cols, key='uni_num')
                ptype = st.selectbox('Plot type', ['Histogram', 'Box', 'Violin', 'KDE', 'Line', 'ECDF', 'Strip'])
                df_plot = _sample_df(df[[col]].dropna())
                if ptype == 'Histogram':
                    bins = st.slider('Bins', 5, 200, 30)
                    fig = px.histogram(df_plot, x=col, nbins=bins, marginal='box', title=f'Histogram - {col}')
                    st.plotly_chart(fig)
                elif ptype == 'Box':
                    fig = px.box(df_plot, y=col, title=f'Boxplot - {col}')
                    st.plotly_chart(fig)
                elif ptype == 'Violin':
                    fig = px.violin(df_plot, y=col, box=True, points='all', title=f'Violin - {col}')
                    st.plotly_chart(fig)
                elif ptype == 'KDE':
                    try:
                        fig = px.density_contour(df_plot, x=col) if len(df_plot) > 1 else None
                        if fig is not None:
                            st.plotly_chart(fig)
                        else:
                            st.info('Not enough data for KDE')
                    except Exception:
                        try:
                            sns.kdeplot(df_plot[col])
                            st.pyplot(plt)
                            plt.clf()
                        except Exception:
                            st.info('KDE failed')
                elif ptype == 'Line':
                    fig = px.line(df_plot.reset_index(), x='index', y=col, title=f'Line - {col}')
                    st.plotly_chart(fig)
                elif ptype == 'ECDF':
                    try:
                        vals = np.sort(df_plot[col].dropna())
                        y = np.arange(1, len(vals)+1) / len(vals)
                        fig = go.Figure(data=go.Scatter(x=vals, y=y, mode='lines'))
                        fig.update_layout(title=f'ECDF - {col}', xaxis_title=col, yaxis_title='ECDF')
                        st.plotly_chart(fig)
                    except Exception:
                        st.info('ECDF failed')
                elif ptype == 'Strip':
                    fig = px.strip(df_plot, y=col, title=f'Strip - {col}')
                    st.plotly_chart(fig)

        elif plot_group == 'Univariate (categorical)':
            if not categorical_cols:
                st.info('No categorical columns available')
            else:
                col = st.selectbox('Categorical column', categorical_cols, key='uni_cat')
                df_plot = _sample_df(df[[col]].dropna())
                agg = st.selectbox('Aggregate/count', ['Count', 'Proportion', 'Top values only'])
                counts = df_plot[col].value_counts()
                if agg == 'Count':
                    fig = px.bar(counts.reset_index().rename(columns={'index': col, col: 'count'}), x=col, y='count', title=f'Counts - {col}')
                    st.plotly_chart(fig)
                elif agg == 'Proportion':
                    pct = counts / counts.sum()
                    fig = px.pie(values=pct.values, names=pct.index, title=f'Proportions - {col}')
                    st.plotly_chart(fig)
                else:
                    topn = st.slider('Top N', 1, min(50, len(counts)), 10)
                    fig = px.bar(counts.head(topn).reset_index().rename(columns={'index': col, col: 'count'}), x=col, y='count', title=f'Top {topn} - {col}')
                    st.plotly_chart(fig)

        elif plot_group == 'Bivariate numeric':
            if len(numeric_cols) < 2:
                st.info('Need at least 2 numeric columns')
            else:
                x_col = st.selectbox('X', numeric_cols, key='bi_x')
                y_col = st.selectbox('Y', [c for c in numeric_cols if c != x_col], key='bi_y')
                hue = st.selectbox('Color by (optional)', [None] + categorical_cols + numeric_cols, key='bi_hue')
                ptype = st.selectbox('Plot type', ['Scatter', 'Hexbin (large data)', '2D Density', 'Line', 'Regression'])
                df_plot = _sample_df(df[[x_col, y_col] + ([hue] if hue else [])].dropna())
                if ptype == 'Scatter':
                    fig = px.scatter(df_plot, x=x_col, y=y_col, color=hue) if hue else px.scatter(df_plot, x=x_col, y=y_col)
                    st.plotly_chart(fig)
                elif ptype == 'Hexbin':
                    try:
                        fig = px.density_heatmap(df_plot, x=x_col, y=y_col)
                        st.plotly_chart(fig)
                    except Exception:
                        st.info('Hexbin failed')
                elif ptype == '2D Density':
                    try:
                        fig = px.density_contour(df_plot, x=x_col, y=y_col)
                        st.plotly_chart(fig)
                    except Exception:
                        st.info('2D density failed')
                elif ptype == 'Line':
                    fig = px.line(df_plot, x=x_col, y=y_col, title=f'Line {x_col} vs {y_col}')
                    st.plotly_chart(fig)
                elif ptype == 'Regression':
                    try:
                        import statsmodels.api as sm
                        Xr = sm.add_constant(df_plot[x_col])
                        model_ = sm.OLS(df_plot[y_col], Xr).fit()
                        st.text(str(model_.summary()))
                    except Exception:
                        try:
                            sns.regplot(x=x_col, y=y_col, data=df_plot)
                            st.pyplot(plt)
                            plt.clf()
                        except Exception:
                            st.info('Regression plot failed')

        elif plot_group == 'Bivariate categorical':
            if len(categorical_cols) < 1 or len(numeric_cols) < 1:
                st.info('Need at least one categorical and one numeric column')
            else:
                cat = st.selectbox('Categorical column', categorical_cols, key='bi_cat')
                num = st.selectbox('Numeric column (aggregate)', numeric_cols, key='bi_cat_num')
                agg = st.selectbox('Aggregate function', ['mean', 'median', 'sum', 'count'])
                df_agg = df.groupby(cat)[num].agg(agg).reset_index()
                fig = px.bar(df_agg, x=cat, y=num, title=f'{agg} of {num} by {cat}')
                st.plotly_chart(fig)

        elif plot_group == 'Multivariate':
            if len(numeric_cols) < 2:
                st.info('Need at least 2 numeric columns')
            else:
                selected = st.multiselect('Select numeric columns (2-6)', numeric_cols, default=numeric_cols[:3], key='multi_cols')
                if len(selected) >= 2:
                    try:
                        df_plot = _sample_df(df[selected].dropna())
                        fig = px.scatter_matrix(df_plot[selected], dimensions=selected, title='Scatter matrix')
                        st.plotly_chart(fig)
                    except Exception:
                        st.info('Scatter matrix failed')
                else:
                    st.info('Select at least 2 columns')

        elif plot_group == 'Time series':
            # try to infer datetime columns
            datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            if not datetime_cols:
                # try to parse any object column as datetime
                candidates = [c for c in df.columns if df[c].dtype == object]
                parsed = []
                for c in candidates:
                    try:
                        pd.to_datetime(df[c])
                        parsed.append(c)
                    except Exception:
                        pass
                datetime_cols = parsed
            if not datetime_cols:
                st.info('No datetime-like columns detected')
            else:
                dt = st.selectbox('Datetime column', datetime_cols)
                value = st.selectbox('Value column', numeric_cols)
                df_ts = df[[dt, value]].dropna()
                try:
                    df_ts[dt] = pd.to_datetime(df_ts[dt])
                    df_ts = df_ts.sort_values(dt)
                    df_plot = _sample_df(df_ts)
                    fig = px.line(df_plot, x=dt, y=value, title=f'Time series: {value} over {dt}')
                    st.plotly_chart(fig)
                except Exception:
                    st.info('Time series plotting failed')

        elif plot_group == 'Distribution / Density':
            if not numeric_cols:
                st.info('No numeric columns')
            else:
                col = st.selectbox('Numeric column', numeric_cols, key='dist_col')
                overlay = st.selectbox('Overlay by (optional)', [None] + categorical_cols)
                df_plot = _sample_df(df[[col, overlay] if overlay else [col]].dropna())
                try:
                    if overlay:
                        fig = px.histogram(df_plot, x=col, color=overlay, nbins=40, marginal='rug', barmode='overlay')
                    else:
                        fig = px.histogram(df_plot, x=col, nbins=40, marginal='rug')
                    st.plotly_chart(fig)
                except Exception:
                    st.info('Distribution plot failed')

        elif plot_group == 'Advanced (Pairplot, PCA, Clustering)':
            adv = st.selectbox('Advanced option', ['Pairplot', 'PCA (2 comps)', 'Correlation heatmap', 'KMeans clustering'])
            if adv == 'Pairplot':
                if len(numeric_cols) < 2:
                    st.info('Need at least 2 numeric columns')
                else:
                    sel = st.multiselect('Select numeric columns for pairplot', numeric_cols, default=numeric_cols[:4])
                    try:
                        sns.pairplot(_sample_df(df[sel].dropna()).sample(min(200, len(df))))
                        st.pyplot(plt)
                        plt.clf()
                    except Exception:
                        st.info('Pairplot failed')
            elif adv == 'PCA (2 comps)':
                if len(numeric_cols) < 2:
                    st.info('Need at least 2 numeric columns for PCA')
                else:
                    comps = PCA(n_components=2).fit_transform(df[numeric_cols].fillna(0))
                    df_pca = pd.DataFrame(comps, columns=['PC1', 'PC2'])
                    fig = px.scatter(df_pca, x='PC1', y='PC2', title='PCA (2 components)')
                    st.plotly_chart(fig)
            elif adv == 'Correlation heatmap':
                if len(numeric_cols) < 2:
                    st.info('Need numeric columns for correlation')
                else:
                    corr = df[numeric_cols].corr()
                    fig = px.imshow(corr, text_auto='.2f', title='Correlation heatmap')
                    st.plotly_chart(fig)
            elif adv == 'KMeans clustering':
                if len(numeric_cols) < 2:
                    st.info('Need numeric columns for clustering')
                else:
                    k = st.slider('Clusters (k)', 2, min(20, max(2, df.shape[0]//2)), 3)
                    km = KMeans(n_clusters=k, random_state=42)
                    dfc = df[numeric_cols].fillna(0)
                    df['Cluster'] = km.fit_predict(dfc)
                    fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color='Cluster', title='KMeans clusters')
                    st.plotly_chart(fig)

        # small export helpers
        exports_dir = 'exports'
        if not os.path.exists(exports_dir):
            os.makedirs(exports_dir)

        if st.button('Export current dataframe sample as CSV'):
            sampled = _sample_df(df)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'data_sample_{timestamp}.csv'
            server_path = os.path.join(exports_dir, filename)
            try:
                sampled.to_csv(server_path, index=False)
                st.success(f'Saved server-side: {server_path}')
                # provide download for the server-saved file
                with open(server_path, 'rb') as f:
                    data = f.read()
                    st.download_button('Download saved CSV', data, file_name=filename)
            except Exception as e:
                st.error(f'Failed to save export: {e}')

        # show available exports and models
        try:
            st.markdown('**Server-side exports**')
            exports = sorted([p for p in os.listdir(exports_dir) if p.endswith('.csv')], reverse=True)
            if exports:
                for p in exports[:10]:
                    st.write(p)
            else:
                st.write('No server-side exports yet')
        except Exception:
            pass

        try:
            st.markdown('**Saved models (server-side)**')
            if os.path.exists('models'):
                mods = sorted([p for p in os.listdir('models') if p.endswith('.pkl')], reverse=True)
                if mods:
                    for m in mods[:10]:
                        st.write(m)
                else:
                    st.write('No saved models yet')
            else:
                st.write('Models directory not found')
        except Exception:
            pass

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

        # Cross-validation option
        cv_enable = st.checkbox("Enable cross-validation (k-fold)")
        cv_folds = st.number_input("CV folds", min_value=2, max_value=10, value=3)

        # training log
        training_log = []

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
            preds_dict = {}
            probs_dict = {}
            # If no models were selected, warn and skip training to avoid division by zero
            if len(models) == 0:
                st.warning("No models selected to train. Please select at least one model.")
            if models:
                progress = st.progress(0)
                total = len(models)
                cm_img_b64 = None
                roc_img_b64 = None
                model_stats = []
                for i, (name, model) in enumerate(models.items()):
                    try:
                        start_time = time.time()
                        model_used = model
                        if tune and name in param_grids:
                            grid = GridSearchCV(model, param_grids[name], cv=3, n_jobs=1)
                            grid.fit(X_train, y_train)
                            model_used = grid.best_estimator_
                        else:
                            model_used.fit(X_train, y_train)
                        fitted_models[name] = model_used

                        train_time = time.time() - start_time

                        # cross-validation score if enabled
                        cv_mean = cv_std = None
                        if cv_enable:
                            try:
                                scores = cross_val_score(model_used, X_encoded, y, cv=int(cv_folds))
                                cv_mean = np.mean(scores)
                                cv_std = np.std(scores)
                            except Exception:
                                cv_mean = cv_std = None

                        y_pred = model_used.predict(X_test)
                        preds_dict[name] = y_pred

                        # capture probabilities if available
                        try:
                            if hasattr(model_used, 'predict_proba'):
                                probs = model_used.predict_proba(X_test)
                                probs_dict[name] = probs
                        except Exception:
                            pass

                        if task_type == "Regression":
                            results.append([name, r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred, squared=False)])
                            model_stats.append({'model': name, 'train_time': train_time, 'cv_mean': cv_mean, 'cv_std': cv_std, 'test_metric': r2_score(y_test, y_pred)})
                        else:
                            acc = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            results.append([name, acc, f1])
                            preds = y_pred
                            model_stats.append({'model': name, 'train_time': train_time, 'cv_mean': cv_mean, 'cv_std': cv_std, 'test_metric': acc, 'f1': f1})

                            # capture confusion matrix and ROC for this classifier
                            try:
                                cm = confusion_matrix(y_test, preds)
                                fig_cm = px.imshow(cm, text_auto=True, title=f'Confusion Matrix - {name}')
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
                                        fig_roc.update_layout(title=f'ROC Curve - {name}', xaxis_title='FPR', yaxis_title='TPR')
                                        roc_img = fig_roc.to_image(format='png')
                                        roc_img_b64 = base64.b64encode(roc_img).decode()
                            except Exception:
                                roc_img_b64 = None

                    except Exception as e:
                        results.append([name, str(e), ""])

                    # safe progress percentage between 0 and 100
                    pct = int(((i+1)/total) * 100) if total > 0 else 100
                    pct = max(0, min(100, pct))
                    if 'progress' in locals():
                        try:
                            # use .progress to set value; wrap in try to avoid StreamlitAPIException if UI reruns
                            progress.progress(pct)
                        except Exception:
                            pass
            if 'progress' in locals():
                try:
                    progress.empty()
                except Exception:
                    pass

            results_df = pd.DataFrame(results, columns=["Model", "Metric1", "Metric2"]) if len(results)>0 else pd.DataFrame()
            st.subheader("Model Comparison")
            st.dataframe(results_df)

            # Show richer comparison table
            if model_stats:
                comp = pd.DataFrame(model_stats)
                st.subheader('Comparison (train time, CV mean±std, test metric)')
                st.dataframe(comp)

                # choose best model by metric (only show metrics present in the comparison table)
                available_metrics = [c for c in ['test_metric', 'f1', 'cv_mean', 'train_time'] if c in comp.columns]
                if not available_metrics:
                    available_metrics = comp.columns.tolist()
                choose_metric = st.selectbox('Choose metric to pick best model', available_metrics)
                try:
                    best_row = comp.sort_values(by=choose_metric, ascending=False).iloc[0]
                    st.info(f"Best model by {choose_metric}: {best_row['model']}")
                    if st.button('Save best model'):
                        try:
                            bm = fitted_models[best_row['model']]
                            save_path = os.path.join('models', f"best_{best_row['model']}.pkl")
                            joblib.dump(bm, save_path)
                            st.success(f"Saved best model to {save_path}")
                        except Exception as e:
                            st.error(f"Failed to save best model: {e}")
                except Exception:
                    pass

            # Detailed classification metrics per model
            if task_type == "Classification":
                detailed = []
                for name in results_df['Model'].tolist():
                    y_pred_m = preds_dict.get(name)
                    if y_pred_m is None:
                        detailed.append([name, None, None, None, None, None])
                        continue
                    try:
                        acc = accuracy_score(y_test, y_pred_m)
                        prec = precision_score(y_test, y_pred_m, average='weighted', zero_division=0)
                        rec = recall_score(y_test, y_pred_m, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred_m, average='weighted', zero_division=0)
                    except Exception:
                        acc = prec = rec = f1 = None
                    # ROC AUC when probability for binary
                    roc_auc = None
                    probs = probs_dict.get(name)
                    if probs is not None and probs.shape[1] == 2:
                        try:
                            fpr, tpr, _ = roc_curve(y_test, probs[:,1])
                            roc_auc = auc(fpr, tpr)
                        except Exception:
                            roc_auc = None
                    detailed.append([name, acc, prec, rec, f1, roc_auc])
                metrics_df = pd.DataFrame(detailed, columns=['Model','Accuracy','Precision','Recall','F1','ROC_AUC'])
                st.subheader('Per-model classification metrics')
                st.dataframe(metrics_df)

                # Option to auto-show full report for all trained models
                show_all_reports = st.checkbox('Show full report for all trained models')

                # Select a model for detailed report
                sel = st.selectbox('Select model for detailed report', ['-- none --'] + metrics_df['Model'].tolist())

                def show_model_report(name):
                    y_pred_sel = preds_dict.get(name)
                    st.write('Classification report for', name)
                    try:
                        cr = classification_report(y_test, y_pred_sel, zero_division=0, output_dict=False)
                        st.text(cr)
                    except Exception:
                        st.info('Could not produce classification report')
                    # show confusion matrix
                    try:
                        cm = confusion_matrix(y_test, y_pred_sel)
                        fig = px.imshow(cm, text_auto=True, title=f'Confusion Matrix - {name}')
                        st.plotly_chart(fig)
                    except Exception:
                        pass
                    # ROC if available
                    if name in probs_dict and probs_dict[name].shape[1] == 2:
                        try:
                            y_proba_sel = probs_dict[name]
                            fpr, tpr, _ = roc_curve(y_test, y_proba_sel[:,1])
                            roc_auc_sel = auc(fpr, tpr)
                            figroc = go.Figure()
                            figroc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC={roc_auc_sel:.3f}'))
                            figroc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash')))
                            figroc.update_layout(title=f'ROC Curve - {name}', xaxis_title='FPR', yaxis_title='TPR')
                            st.plotly_chart(figroc)
                        except Exception:
                            pass

                if sel and sel != '-- none --':
                    show_model_report(sel)

                if show_all_reports:
                    st.markdown('### Full reports for all trained models')
                    for m in metrics_df['Model'].tolist():
                        st.markdown(f'---\n**Model:** {m}')
                        try:
                            show_model_report(m)
                        except Exception:
                            st.info(f'Could not produce report for {m}')

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
                    html_bytes = html.encode('utf-8')
                    st.download_button("Download report (HTML)", html_bytes, file_name="report.html", mime="text/html")
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
                csv_bytes = df_pred.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions CSV", csv_bytes, file_name="predictions.csv", mime="text/csv")

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
        try:
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download cleaned dataset (CSV)", csv_bytes, file_name="cleaned_dataset.csv", mime='text/csv')
        except Exception:
            st.info('Could not prepare download for cleaned dataset')
        st.info("All predictions can also be downloaded from Modeling tab.")

else:
    st.info("Upload a CSV file to get started!")
