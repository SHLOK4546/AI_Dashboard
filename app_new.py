import streamlit as st
import pandas as pd
import json
import plotly.express as px
import numpy as np
import uuid
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from transformers import pipeline
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(page_title="Data Analytics Dashboard", layout="wide")

# Load external CSS
css_path = "styles.css"
try:
    with open(css_path, "r") as css_file:
        css_content = css_file.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.error(f"CSS file not found at {css_path}. Please ensure styles.css is in the correct directory.")
    css_content = ""

# Initialize session state
if 'is_authenticated' not in st.session_state:
    st.session_state.is_authenticated = False
if 'charts' not in st.session_state:
    st.session_state.charts = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_path' not in st.session_state:
    st.session_state.file_path = None
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'page' not in st.session_state:
    st.session_state.page = "login"
if 'show_insights' not in st.session_state:
    st.session_state.show_insights = False
if 'chart_insights' not in st.session_state:
    st.session_state.chart_insights = {}
if 'active_drill_down' not in st.session_state:
    st.session_state.active_drill_down = None

# Function to suggest chart type
def suggest_chart_type(x_col, y_col, z_col, df):
    x_dtype = df[x_col].dtype
    y_dtype = df[y_col].dtype if y_col != "None" else None
    z_dtype = df[z_col].dtype if z_col != "None" else None
    x_unique = df[x_col].nunique()
    x_is_categorical = x_dtype == "object" or x_unique < 20
    y_is_numerical = y_dtype in ["int64", "float64"] if y_col != "None" else False
    z_is_numerical = z_dtype in ["int64", "float64"] if z_col != "None" else False
    is_timestamp = "datetime" in str(x_dtype).lower() or (z_col != "None" and "datetime" in str(z_dtype).lower())

    if is_timestamp and y_is_numerical:
        return "Line", "Line chart is ideal for time-series sales data with numerical Y."
    elif x_is_categorical and y_is_numerical and z_col == "None":
        return "Bar", "Bar chart is suitable for comparing sales across categories."
    elif x_is_categorical and y_col == "None" and z_col == "None":
        return "Pie", "Pie chart is ideal for showing distribution of sales categories."
    elif x_dtype in ["int64", "float64"] and y_is_numerical and z_is_numerical:
        return "Scatter3D", "3D Scatter plot is suitable for analyzing three numerical metrics."
    elif x_dtype in ["int64", "float64"] and y_is_numerical and z_col == "None":
        return "Scatter", "Scatter plot is great for exploring sales relationships."
    else:
        return "Bar", "Defaulting to Bar chart for mixed or unclear data types."

# Function to generate AI insights
def generate_ai_insights(df, selected_columns):
    summaries = []
    numerical_cols = df[selected_columns].select_dtypes(include=["int64", "float64"]).columns
    timestamp_cols = df[selected_columns].select_dtypes(include=["datetime64"]).columns
    categorical_cols = df[selected_columns].select_dtypes(include=["object"]).columns

    if numerical_cols.size > 0:
        X = df[numerical_cols].dropna()
        if not X.empty:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            if X.shape[1] >= 2 and X.shape[0] >= 3:
                kmeans = KMeans(n_clusters=min(3, X.shape[0]), random_state=42)
                clusters = kmeans.fit_predict(X_scaled)
                cluster_sizes = pd.Series(clusters).value_counts().sort_index()
                for i, size in cluster_sizes.items():
                    summaries.append(
                        f"Cluster {i+1} contains {size} records ({size/X.shape[0]*100:.2f}% of data) in {', '.join(numerical_cols)}. "
                        f"Analyze this cluster to identify key customer segments or high-value sales groups."
                    )
            if X.shape[0] >= 3:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomalies = iso_forest.fit_predict(X_scaled)
                num_anomalies = (anomalies == -1).sum()
                if num_anomalies > 0:
                    summaries.append(
                        f"Detected {num_anomalies} anomalies in {', '.join(numerical_cols)} ({num_anomalies/X.shape[0]*100:.2f}% of data). "
                        "Investigate these outliers as potential high-value sales or errors impacting revenue."
                    )
    
    if timestamp_cols.size > 0 and numerical_cols.size > 0:
        time_col = timestamp_cols[0]
        df_sorted = df.sort_values(by=time_col).dropna(subset=[time_col])
        for num_col in numerical_cols:
            data = df_sorted[[time_col, num_col]].dropna()
            if data.shape[0] >= 2:
                time_num = (data[time_col] - data[time_col].min()).dt.total_seconds() / (24 * 3600)
                X_time = time_num.values.reshape(-1, 1)
                y = data[num_col].values
                reg = LinearRegression().fit(X_time, y)
                slope = reg.coef_[0]
                trend = "increasing" if slope > 0 else "decreasing"
                summaries.append(
                    f"{num_col} shows a {trend} trend (~{abs(slope):.2f}/day). "
                    f"Capitalize on {trend} sales by adjusting marketing or inventory strategies."
                )
    
    for cat_col in categorical_cols:
        if df[cat_col].notnull().sum() > 0:
            top_category = df[cat_col].value_counts().index[0]
            top_count = df[cat_col].value_counts().iloc[0]
            total_count = df[cat_col].notnull().sum()
            percent = (top_count / total_count) * 100
            summaries.append(
                f"In '{cat_col}', '{top_category}' dominates with {percent:.2f}% of records. "
                f"Focus marketing efforts on this category or explore untapped segments to boost sales."
            )
    
    if summaries:
        try:
            summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", framework="pt", device=-1)
            input_text = ". ".join(summaries)
            max_input_length = 512
            input_text = input_text[:max_input_length]
            summary = summarizer(input_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            return [summary]
        except Exception as e:
            return summaries[:1]
    else:
        return ["No clear patterns or trends found. Consider collecting more detailed sales data."]

# Function to generate summarized statistical insights
def generate_stat_insights(df, selected_columns=None):
    if selected_columns is None:
        selected_columns = df.columns
    
    insights = []
    
    num_rows, num_cols = df.shape
    insights.append(
        f"Your dataset contains {num_rows} records across {num_cols} columns, providing a "
        f"{'rich' if num_rows > 1000 else 'modest'} dataset for business analysis."
    )
    
    missing = df[selected_columns].isnull().sum()
    missing_total = missing.sum()
    missing_cols = missing[missing > 0]
    if missing_total > 0:
        missing_percent = (missing_total / (num_rows * len(selected_columns))) * 100
        top_missing_col = missing_cols.idxmax() if not missing_cols.empty else None
        top_missing_count = missing_cols.max() if not missing_cols.empty else 0
        insights.append(
            f"There are {missing_total} missing values across the dataset, affecting "
            f"{len(missing_cols)} columns ({missing_percent:.2f}% of the data). "
            f"The column '{top_missing_col}' has the most missing values ({top_missing_count})."
        )
        insights.append(
            f"<b>Business Tip:</b> Missing data in '{top_missing_col}' could indicate incomplete customer or sales records. "
            "Implement stricter data collection processes or use predictive models to estimate missing values."
        )
    else:
        insights.append(
            "No missing values found in the selected columns, ensuring reliable data for analysis."
        )
    
    numerical_cols = df[selected_columns].select_dtypes(include=["int64", "float64"]).columns
    if numerical_cols.size > 0:
        desc = df[numerical_cols].describe()
        most_variable_col = desc.loc['std'].idxmax() if not desc.empty else None
        if most_variable_col:
            mean_val = desc[most_variable_col]['mean']
            std_val = desc[most_variable_col]['std']
            min_val = desc[most_variable_col]['min']
            max_val = desc[most_variable_col]['max']
            insights.append(
                f"Numerical data in '{most_variable_col}' shows significant variation with an average of {mean_val:.2f} "
                f"and a standard deviation of {std_val:.2f}, ranging from {min_val:.2f} to {max_val:.2f}."
            )
            insights.append(
                f"<b>Business Tip:</b> High variability in '{most_variable_col}' suggests inconsistent sales or performance metrics. "
                "Investigate outliers to identify top-performing products or anomalies."
            )
    
    categorical_cols = df[selected_columns].select_dtypes(include=["object"]).columns
    if categorical_cols.size > 0:
        most_diverse_col = None
        max_unique = 0
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count > max_unique:
                max_unique = unique_count
                most_diverse_col = col
        if most_diverse_col:
            top_category = df[most_diverse_col].value_counts().index[0]
            top_category_count = df[most_diverse_col].value_counts().iloc[0]
            total_count = df[most_diverse_col].notnull().sum()
            top_category_percent = (top_category_count / total_count) * 100
            insights.append(
                f"The categorical column '{most_diverse_col}' has {max_unique} unique categories, with "
                f"'{top_category}' being the most common, appearing in {top_category_percent:.2f}% of records."
            )
            insights.append(
                f"<b>Business Tip:</b> The dominance of '{top_category}' in '{most_diverse_col}' suggests a strong market segment. "
                "Focus marketing efforts on this segment or diversify into underrepresented categories."
            )
    
    timestamp_cols = df[selected_columns].select_dtypes(include=["datetime64"]).columns
    if timestamp_cols.size > 0:
        time_col = timestamp_cols[0]
        time_range_start = df[time_col].min()
        time_range_end = df[time_col].max()
        time_span_days = (time_range_end - time_range_start).days
        insights.append(
            f"Time data in '{time_col}' spans from {time_range_start.strftime('%Y-%m-%d')} to "
            f"{time_range_end.strftime('%Y-%m-%d')}, covering {time_span_days} days."
        )
        insights.append(
            f"<b>Business Tip:</b> Analyze trends over this {time_span_days}-day period to identify seasonal patterns or peak sales periods."
        )
    
    if numerical_cols.size > 0:
        skew_col = df[numerical_cols].skew().abs().idxmax()
        skew_value = df[numerical_cols].skew()[skew_col]
        if abs(skew_value) > 1:
            skew_type = "positively" if skew_value > 0 else "negatively"
            insights.append(
                f"The column '{skew_col}' is {skew_type} skewed (skewness: {skew_value:.2f}), indicating uneven distribution."
            )
            insights.append(
                f"<b>Business Tip:</b> The skewed distribution in '{skew_col}' suggests a few high-value transactions or outliers. "
                "Target these high-value customers or address low-performing segments."
            )
    
    if numerical_cols.size > 1:
        insights.append(
            f"<b>General Tip:</b> Combine columns like '{numerical_cols[0]}' and '{numerical_cols[1]}' to create metrics "
            "to identify underperforming products or regions."
        )
    
    return insights

# Function to delete a chart
def delete_chart(chart_id):
    st.session_state.charts = [chart for chart in st.session_state.charts if chart['id'] != chart_id]
    if chart_id in st.session_state.chart_insights:
        del st.session_state.chart_insights[chart_id]
    if st.session_state.active_drill_down and st.session_state.active_drill_down['chart_key'] == f"chart_{chart_id}":
        st.session_state.active_drill_down = None

# Function to toggle chart insights
def toggle_chart_insights(chart_id):
    st.session_state.chart_insights[chart_id] = not st.session_state.chart_insights.get(chart_id, False)

# Function to close drill-down modal
def close_drill_down():
    st.session_state.active_drill_down = None

# Function to generate automatic charts
def generate_auto_charts(df):
    charts = []
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    timestamp_cols = df.select_dtypes(include=["datetime64"]).columns
    
    target_charts = 10
    chart_count = 0

    for cat_col in categorical_cols[:2]:
        if df[cat_col].nunique() < 20 and chart_count < 2:
            try:
                fig = px.pie(df, names=cat_col, title=f"Distribution of {cat_col}")
                fig.update_layout(
                    template="plotly_dark",
                    autosize=True,
                    margin=dict(l=5, r=5, t=10, b=5),
                    title_font_size=20,
                    font=dict(size=10),
                    legend=dict(
                        font=dict(size=15),
                        orientation="h",
                        yanchor="bottom",
                        y=-0.1,
                        xanchor="center",
                        x=0.5
                    ),
                    clickmode='event+select'  # Enable selection events
                )
                chart_id = str(uuid.uuid4())
                selected_columns = [cat_col]
                charts.append({
                    'id': chart_id,
                    'fig': fig,
                    'title': fig.layout.title.text,
                    'type': "Pie",
                    'x_col': cat_col,
                    'y_col': "None",
                    'z_col': "None",
                    'ai_insights': generate_ai_insights(df, selected_columns),
                    'stat_insights': generate_stat_insights(df, selected_columns)
                })
                chart_count += 1
            except Exception as e:
                st.warning(f"Failed to generate Pie chart for {cat_col}: {e}")

    if timestamp_cols.size > 0 and numerical_cols.size > 0 and chart_count < 3:
        time_col = timestamp_cols[0]
        num_col = numerical_cols[0]
        try:
            fig = px.line(df, x=time_col, y=num_col, title=f"Sales Trend of {num_col} over {time_col}")
            fig.update_layout(
                template="plotly_dark",
                xaxis_title=time_col,
                yaxis_title=num_col,
                autosize=True,
                margin=dict(l=5, r=5, t=10, b=5),
                title_font_size=20,
                font=dict(size=15),
                xaxis=dict(tickfont=dict(size=6), title_font=dict(size=6), automargin=True),
                yaxis=dict(tickfont=dict(size=6), title_font=dict(size=6), automargin=True),
                legend=dict(
                    font=dict(size=15),
                    orientation="h",
                    yanchor="bottom",
                    y=-0.1,
                    xanchor="center",
                    x=0.5
                )
            )
            chart_id = str(uuid.uuid4())
            selected_columns = [time_col, num_col]
            charts.append({
                'id': chart_id,
                'fig': fig,
                'title': fig.layout.title.text,
                'type': "Line",
                'x_col': time_col,
                'y_col': num_col,
                'z_col': "None",
                'ai_insights': generate_ai_insights(df, selected_columns),
                'stat_insights': generate_stat_insights(df, selected_columns)
            })
            chart_count += 1
        except Exception as e:
            st.warning(f"Failed to generate Line chart for {time_col} vs {num_col}: {e}")

    if categorical_cols.size > 0 and numerical_cols.size > 0 and chart_count < 5:
        for cat_col in categorical_cols[:2]:
            if chart_count < 5:
                num_col = numerical_cols[0]
                try:
                    fig = px.bar(df, x=cat_col, y=num_col, title=f"Sales of {num_col} by {cat_col}")
                    fig.update_layout(
                        template="plotly_dark",
                        xaxis_title=cat_col,
                        yaxis_title=num_col,
                        autosize=True,
                        margin=dict(l=5, r=5, t=10, b=5),
                        title_font_size=7,
                        font=dict(size=6),
                        xaxis=dict(tickfont=dict(size=4), title_font=dict(size=5), automargin=True),
                        yaxis=dict(tickfont=dict(size=4), title_font=dict(size=5), automargin=True),
                        legend=dict(
                            font=dict(size=4),
                            orientation="h",
                            yanchor="bottom",
                            y=-0.1,
                            xanchor="center",
                            x=0.5
                        )
                    )
                    chart_id = str(uuid.uuid4())
                    selected_columns = [cat_col, num_col]
                    charts.append({
                        'id': chart_id,
                        'fig': fig,
                        'title': fig.layout.title.text,
                        'type': "Bar",
                        'x_col': cat_col,
                        'y_col': num_col,
                        'z_col': "None",
                        'ai_insights': generate_ai_insights(df, selected_columns),
                        'stat_insights': generate_stat_insights(df, selected_columns)
                    })
                    chart_count += 1
                except Exception as e:
                    st.warning(f"Failed to generate Bar chart for {cat_col} vs {num_col}: {e}")

    if numerical_cols.size >= 2 and chart_count < target_charts:
        for i in range(len(numerical_cols)):
            for j in range(i + 1, len(numerical_cols)):
                if chart_count < target_charts:
                    x_col, y_col = numerical_cols[i], numerical_cols[j]
                    try:
                        fig = px.scatter(df, x=x_col, y=y_col, title=f"Sales Analysis: {y_col} vs {x_col}")
                        fig.update_layout(
                            template="plotly_dark",
                            xaxis_title=x_col,
                            yaxis_title=y_col,
                            autosize=True,
                            margin=dict(l=5, r=5, t=10, b=5),
                            title_font_size=7,
                            font=dict(size=6),
                            xaxis=dict(tickfont=dict(size=4), title_font=dict(size=5), automargin=True),
                            yaxis=dict(tickfont=dict(size=4), title_font=dict(size=5), automargin=True),
                            legend=dict(
                                font=dict(size=4),
                                orientation="h",
                                yanchor="bottom",
                                y=-0.1,
                                xanchor="center",
                                x=0.5
                            )
                        )
                        chart_id = str(uuid.uuid4())
                        selected_columns = [x_col, y_col]
                        charts.append({
                            'id': chart_id,
                            'fig': fig,
                            'title': fig.layout.title.text,
                            'type': "Scatter",
                            'x_col': x_col,
                            'y_col': y_col,
                            'z_col': "None",
                            'ai_insights': generate_ai_insights(df, selected_columns),
                            'stat_insights': generate_stat_insights(df, selected_columns)
                        })
                        chart_count += 1
                    except Exception as e:
                        st.warning(f"Failed to generate Scatter chart for {x_col} vs {y_col}: {e}")

    if numerical_cols.size >= 3 and chart_count < target_charts:
        for i in range(len(numerical_cols) - 2):
            x_col, y_col, z_col = numerical_cols[i:i+3]
            if chart_count < target_charts:
                try:
                    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, title=f"3D Sales Analysis: {x_col} vs {y_col} vs {z_col}")
                    fig.update_layout(
                        template="plotly_dark",
                        scene=dict(
                            xaxis_title=x_col,
                            yaxis_title=y_col,
                            zaxis_title=z_col,
                            xaxis=dict(tickfont=dict(size=4), title_font=dict(size=5)),
                            yaxis=dict(tickfont=dict(size=4), title_font=dict(size=5)),
                            zaxis=dict(tickfont=dict(size=4), title_font=dict(size=5))
                        ),
                        autosize=True,
                        margin=dict(l=5, r=5, t=10, b=5),
                        title_font_size=7,
                        font=dict(size=6),
                        legend=dict(
                            font=dict(size=4),
                            orientation="h",
                            yanchor="bottom",
                            y=-0.1,
                            xanchor="center",
                            x=0.5
                        )
                    )
                    chart_id = str(uuid.uuid4())
                    selected_columns = [x_col, y_col, z_col]
                    charts.append({
                        'id': chart_id,
                        'fig': fig,
                        'title': fig.layout.title.text,
                        'type': "Scatter3D",
                        'x_col': x_col,
                        'y_col': y_col,
                        'z_col': z_col,
                        'ai_insights': generate_ai_insights(df, selected_columns),
                        'stat_insights': generate_stat_insights(df, selected_columns)
                    })
                    chart_count += 1
                except Exception as e:
                    st.warning(f"Failed to generate Scatter3D chart for {x_col} vs {y_col} vs {z_col}: {e}")

    return charts

# Page navigation logic
def main():
    if st.session_state.page == "login":
        login_page()
    elif st.session_state.is_authenticated:
        if st.session_state.page == "upload":
            upload_page()
        elif st.session_state.page == "charts":
            charts_page()
    else:
        st.session_state.page = "login"
        login_page()

def login_page():
    st.title("🔒 Login")
    st.markdown(
    "<h4 style='text-align: center;'>Please enter your credentials to access the dashboard.</h4>",
    unsafe_allow_html=True
)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container():
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                if username == "admin" and password == "password":
                    st.session_state.is_authenticated = True
                    st.session_state.page = "upload"
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password. Try again.")
            

def upload_page():
    st.title("📂 Data Upload & Edit")
    file_type = None
    uploaded_file = st.file_uploader(
        "Upload a file (.csv, .xlsx, .json)",
        type=["csv", "xlsx", "json"]
    )
    
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]
        try:
            if file_type == 'csv':
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.file_path = uploaded_file.name
            elif file_type == 'xlsx':
                st.session_state.df = pd.read_excel(uploaded_file)
                st.session_state.file_path = uploaded_file.name
            elif file_type == 'json':
                json_data = json.load(uploaded_file)
                st.session_state.df = pd.json_normalize(json_data)
                st.session_state.file_path = uploaded_file.name
            st.success(f"File '{st.session_state.file_path}' uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        col_preview, col_edit = st.columns([8, 1])
        with col_preview:
            st.subheader("🔍 Data Preview")
        with col_edit:
            if file_type == 'csv' and st.button("Edit CSV", key="edit_csv"):
                st.session_state.edit_mode = not st.session_state.edit_mode
        
        if file_type == 'csv' and st.session_state.edit_mode:
            edited_df = st.data_editor(df, num_rows="dynamic", key="data_editor")
            if not edited_df.equals(st.session_state.df):
                st.session_state.df = edited_df
                st.success("Data updated in memory.")
        else:
            st.dataframe(df.head())
        
        st.markdown(f"*Rows:* {df.shape[0]} | *Columns:* {df.shape[1]}")
        
        if st.button("Submit Data"):
            if st.session_state.df is not None:
                st.session_state.page = "charts"
                st.session_state.charts = generate_auto_charts(st.session_state.df)
                st.session_state.chart_insights = {chart['id']: False for chart in st.session_state.charts}
                st.session_state.active_drill_down = None
                st.rerun()
            else:
                st.error("Please upload a valid file before submitting.")
    else:
        st.info("Please upload a file to see the preview.")

def charts_page():
    if st.session_state.df is None:
        st.error("No data available. Please return to the upload page.")
        if st.button("Back to Upload"):
            st.session_state.page = "upload"
            st.session_state.charts = []
            st.session_state.chart_insights = {}
            st.session_state.active_drill_down = None
            st.rerun()
        return
    
    df = st.session_state.df

    # JavaScript to toggle modal visibility
    modal_js = """
    <script>
        function showModal() {
            document.getElementById('drill-down-modal').classList.add('active');
        }
        function hideModal() {
            document.getElementById('drill-down-modal').classList.remove('active');
        }
    </script>
    """
    st.markdown(modal_js, unsafe_allow_html=True)

    # Render drill-down modal if active
    if st.session_state.active_drill_down:
        chart_key = st.session_state.active_drill_down['chart_key']
        points = st.session_state.active_drill_down['points']
        chart_id = chart_key.replace("chart_", "")
        chart = next((c for c in st.session_state.charts if c['id'] == chart_id), None)
        
        if chart and points:
            with st.container():
                st.markdown('<div id="drill-down-modal" class="modal">', unsafe_allow_html=True)
                st.markdown('<div class="modal-content">', unsafe_allow_html=True)
                st.markdown("**Drill-Down Analysis**")
                try:
                    # Filter data based on chart type
                    if chart['type'] == "Bar":
                        x_val = points[0]['x']
                        filtered_df = df[df[chart['x_col']] == x_val]
                    elif chart['type'] == "Pie":
                        label = points[0].get('label', None)
                        if label is not None:
                            filtered_df = df[df[chart['x_col']] == label]
                        else:
                            st.warning("No valid label selected in Pie chart.")
                            filtered_df = df
                    elif chart['type'] == "Scatter":
                        point = points[0]
                        filtered_df = df[
                            (df[chart['x_col']] == point['x']) & 
                            (df[chart['y_col']] == point['y'])
                        ]
                    elif chart['type'] == "Line":
                        point = points[0]
                        filtered_df = df[
                            (df[chart['x_col']] == point['x']) & 
                            (df[chart['y_col']] == point['y'])
                        ]
                    elif chart['type'] == "Scatter3D":
                        point = points[0]
                        filtered_df = df[
                            (df[chart['x_col']] == point['x']) & 
                            (df[chart['y_col']] == point['y']) & 
                            (df[chart['z_col']] == point['z'])
                        ]
                    else:
                        filtered_df = df

                    # Show all filtered data
                    st.write(f"Filtered Data ({len(filtered_df)} rows):")
                    st.dataframe(filtered_df, use_container_width=True)

                    # Generate automatic drill-down visualization
                    time_cols = filtered_df.select_dtypes(include=["datetime64"]).columns
                    num_cols = filtered_df.select_dtypes(include=["int64", "float64"]).columns

                    if len(time_cols) > 0 and len(num_cols) > 0:
                        time_col = time_cols[0]
                        num_col = num_cols[0]
                        drill_fig = px.line(filtered_df, x=time_col, y=num_col,
                                            title=f"Temporal Trend for Selected Segment")
                        drill_fig.update_layout(
                            template="plotly_dark",
                            autosize=True,
                            margin=dict(l=5, r=5, t=20, b=5),
                            title_font_size=12,
                            font=dict(size=10)
                        )
                        st.plotly_chart(drill_fig, use_container_width=True)
                    elif len(num_cols) > 1:
                        drill_fig = px.scatter(filtered_df, x=num_cols[0], y=num_cols[1],
                                               title="Correlation Analysis")
                        drill_fig.update_layout(
                            template="plotly_dark",
                            autosize=True,
                            margin=dict(l=5, r=5, t=20, b=5),
                            title_font_size=12,
                            font=dict(size=10)
                        )
                        st.plotly_chart(drill_fig, use_container_width=True)

                    # Close button
                    st.button("Close", key=f"close_drill_down_{chart_id}", on_click=close_drill_down)
                
                except Exception as e:
                    st.warning(f"Error generating drill-down: {e}")
                    st.button("Close", key=f"close_drill_down_error_{chart_id}", on_click=close_drill_down)
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📈 Auto-Generated Sales Charts")
    if st.session_state.charts:
        num_charts = len(st.session_state.charts)
        chart_height = 200

        # First row: 3 charts
        if num_charts >= 1:
            cols = st.columns(3)
            for j in range(min(3, num_charts)):
                chart = st.session_state.charts[j]
                with cols[j]:
                    with st.container(height=chart_height, border=False):
                        chart['fig'].update_layout(
                            autosize=True,
                            margin=dict(l=5, r=5, t=20, b=5),
                            title_font_size=15,
                            font=dict(size=15),
                            xaxis=dict(
                                tickfont=dict(size=10),
                                title_font=dict(size=10),
                                automargin=True
                            ),
                            yaxis=dict(
                                tickfont=dict(size=10),
                                title_font=dict(size=10),
                                automargin=True
                            ),
                            scene=dict(
                                xaxis=dict(tickfont=dict(size=8), title_font=dict(size=8)),
                                yaxis=dict(tickfont=dict(size=8), title_font=dict(size=8)),
                                zaxis=dict(tickfont=dict(size=8), title_font=dict(size=8)),
                            ) if chart['type'] == "Scatter3D" else {},
                            legend=dict(
                                font=dict(size=10),
                                orientation="h",
                                yanchor="bottom",
                                y=-0.1,
                                xanchor="center",
                                x=0.5
                            ),
                            height=chart_height
                        )
                        chart_key = f"chart_{chart['id']}"
                        st.plotly_chart(
                            chart['fig'],
                            use_container_width=True,
                            config={'responsive': True, 'staticPlot': False},
                            key=chart_key,
                            on_select="rerun",
                            selection_mode="points"
                        )

                        # Debug selection for Pie chart
                        if chart_key in st.session_state:
                            selected_data = st.session_state[chart_key]
                            points = selected_data.get('selection', {}).get('points', [])
                            if points and chart['type'] == "Pie":
                                st.write(f"Debug: Pie chart selected points: {points}")  # Debug output
                            if points:
                                st.session_state.active_drill_down = {
                                    'chart_key': chart_key,
                                    'points': points
                                }
                                st.markdown("<script>showModal();</script>", unsafe_allow_html=True)

                        # st.markdown('<div class="button-container">', unsafe_allow_html=True)
                        # st.button("🗑️", key=f"delete_{chart['id']}", 
                        #           on_click=delete_chart, args=(chart['id'],), 
                        #           help="Delete this chart")
                        # st.button("📊", key=f"insights_{chart['id']}", 
                        #           on_click=toggle_chart_insights, args=(chart['id'],), 
                        #           help="Toggle insights for this chart")
                        # st.markdown('</div>', unsafe_allow_html=True)
                        if st.session_state.show_insights or st.session_state.chart_insights.get(chart['id'], False):
                            with st.container():
                                insights_class = "chart-insights chart-insights-active" if st.session_state.chart_insights.get(chart['id'], False) else "chart-insights"
                                st.markdown(f'<div class="{insights_class}">', unsafe_allow_html=True)
                                st.markdown(f'<h4>Insights for Chart: {chart["title"]}</h4>', unsafe_allow_html=True)
                                st.markdown('<h5>Key AI Insight</h5>', unsafe_allow_html=True)
                                ai_insight = chart['ai_insights'][0]
                                st.markdown(f'<p>{ai_insight}</p>', unsafe_allow_html=True)
                                st.markdown('<h5>Key Statistics</h5>', unsafe_allow_html=True)
                                for stat_insight in chart['stat_insights']:
                                    formatted = stat_insight.replace("<b>", "<b>").replace("</b>", "</b>")
                                    st.markdown(f'<p>{formatted}</p>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)

        # Second row: 2 charts
        if num_charts >= 4:
            cols = st.columns(2)
            for j in range(2):
                idx = 3 + j
                if idx < num_charts:
                    chart = st.session_state.charts[idx]
                    with cols[j]:
                        with st.container(height=chart_height, border=False):
                            chart['fig'].update_layout(
                                autosize=True,
                                margin=dict(l=5, r=5, t=20, b=5),
                                title_font_size=15,
                                font=dict(size=15),
                                xaxis=dict(
                                    tickfont=dict(size=10),
                                    title_font=dict(size=10),
                                    automargin=True
                                ),
                                yaxis=dict(
                                    tickfont=dict(size=10),
                                    title_font=dict(size=10),
                                    automargin=True
                                ),
                                scene=dict(
                                    xaxis=dict(tickfont=dict(size=8), title_font=dict(size=8)),
                                    yaxis=dict(tickfont=dict(size=8), title_font=dict(size=8)),
                                    zaxis=dict(tickfont=dict(size=8), title_font=dict(size=8)),
                                ) if chart['type'] == "Scatter3D" else {},
                                legend=dict(
                                    font=dict(size=4),
                                    orientation="h",
                                    yanchor="bottom",
                                    y=-0.1,
                                    xanchor="center",
                                    x=0.5
                                ),
                                height=chart_height
                            )
                            chart_key = f"chart_{chart['id']}"
                            st.plotly_chart(
                                chart['fig'],
                                use_container_width=True,
                                config={'responsive': True, 'staticPlot': False},
                                key=chart_key,
                                on_select="rerun",
                                selection_mode="points"
                            )

                            # Debug selection for Pie chart
                            if chart_key in st.session_state:
                                selected_data = st.session_state[chart_key]
                                points = selected_data.get('selection', {}).get('points', [])
                                if points and chart['type'] == "Pie":
                                    st.write(f"Debug: Pie chart selected points: {points}")  # Debug output
                                if points:
                                    st.session_state.active_drill_down = {
                                        'chart_key': chart_key,
                                        'points': points
                                    }
                                    st.markdown("<script>showModal();</script>", unsafe_allow_html=True)

                            # st.markdown('<div class="button-container">', unsafe_allow_html=True)
                            # st.button("🗑️", key=f"delete_{chart['id']}", 
                            #           on_click=delete_chart, args=(chart['id'],), 
                            #           help="Delete this chart")
                            # st.button("📊", key=f"insights_{chart['id']}", 
                            #           on_click=toggle_chart_insights, args=(chart['id'],), 
                            #           help="Toggle insights for this chart")
                            # st.markdown('</div>', unsafe_allow_html=True)
                            if st.session_state.show_insights or st.session_state.chart_insights.get(chart['id'], False):
                                with st.container():
                                    insights_class = "chart-insights chart-insights-active" if st.session_state.chart_insights.get(chart['id'], False) else "chart-insights"
                                    st.markdown(f'<div class="{insights_class}">', unsafe_allow_html=True)
                                    st.markdown(f'<h4>Insights for Chart: {chart["title"]}</h4>', unsafe_allow_html=True)
                                    st.markdown('<h5>Key AI Insight</h5>', unsafe_allow_html=True)
                                    ai_insight = chart['ai_insights'][0]
                                    st.markdown(f'<p>{ai_insight}</p>', unsafe_allow_html=True)
                                    st.markdown('<h5>Key Statistics</h5>', unsafe_allow_html=True)
                                    for stat_insight in chart['stat_insights']:
                                        formatted = stat_insight.replace("<b>", "<b>").replace("</b>", "</b>")
                                        st.markdown(f'<p>{formatted}</p>', unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)

        # Remaining rows: 3 charts per row
        if num_charts > 5:
            for i in range(5, num_charts, 3):
                cols = st.columns(3)
                for j in range(3):
                    idx = i + j
                    if idx < num_charts:
                        chart = st.session_state.charts[idx]
                        with cols[j]:
                            with st.container(height=chart_height, border=False):
                                chart['fig'].update_layout(
                                    autosize=True,
                                    margin=dict(l=5, r=5, t=20, b=5),
                                    title_font_size=15,
                                    font=dict(size=15),
                                    xaxis=dict(
                                        tickfont=dict(size=10),
                                        title_font=dict(size=10),
                                        automargin=True
                                    ),
                                    yaxis=dict(
                                        tickfont=dict(size=10),
                                        title_font=dict(size=10),
                                        automargin=True
                                    ),
                                    scene=dict(
                                        xaxis=dict(tickfont=dict(size=8), title_font=dict(size=8)),
                                        yaxis=dict(tickfont=dict(size=8), title_font=dict(size=8)),
                                        zaxis=dict(tickfont=dict(size=8), title_font=dict(size=8)),
                                    ) if chart['type'] == "Scatter3D" else {},
                                    legend=dict(
                                        font=dict(size=4),
                                        orientation="h",
                                        yanchor="bottom",
                                        y=-0.1,
                                        xanchor="center",
                                        x=0.5
                                    ),
                                    height=chart_height
                                )
                                chart_key = f"chart_{chart['id']}"
                                st.plotly_chart(
                                    chart['fig'],
                                    use_container_width=True,
                                    config={'responsive': True, 'staticPlot': False},
                                    key=chart_key,
                                    on_select="rerun",
                                    selection_mode="points"
                                )

                                # Debug selection for Pie chart
                                if chart_key in st.session_state:
                                    selected_data = st.session_state[chart_key]
                                    points = selected_data.get('selection', {}).get('points', [])
                                    if points and chart['type'] == "Pie":
                                        st.write(f"Debug: Pie chart selected points: {points}")  # Debug output
                                    if points:
                                        st.session_state.active_drill_down = {
                                            'chart_key': chart_key,
                                            'points': points
                                        }
                                        st.markdown("<script>showModal();</script>", unsafe_allow_html=True)

                                # st.markdown('<div class="button-container">', unsafe_allow_html=True)
                                # st.button("🗑️", key=f"delete_{chart['id']}", 
                                #           on_click=delete_chart, args=(chart['id'],), 
                                #           help="Delete this chart")
                                # st.button("📊", key=f"insights_{chart['id']}", 
                                #           on_click=toggle_chart_insights, args=(chart['id'],), 
                                #           help="Toggle insights for this chart")
                                # st.markdown('</div>', unsafe_allow_html=True)
                                if st.session_state.show_insights or st.session_state.chart_insights.get(chart['id'], False):
                                    with st.container():
                                        insights_class = "chart-insights chart-insights-active" if st.session_state.chart_insights.get(chart['id'], False) else "chart-insights"
                                        st.markdown(f'<div class="{insights_class}">', unsafe_allow_html=True)
                                        st.markdown(f'<h4>Insights for Chart: {chart["title"]}</h4>', unsafe_allow_html=True)
                                        st.markdown('<h5>Key AI Insight</h5>', unsafe_allow_html=True)
                                        ai_insight = chart['ai_insights'][0]
                                        st.markdown(f'<p>{ai_insight}</p>', unsafe_allow_html=True)
                                        st.markdown('<h5>Key Statistics</h5>', unsafe_allow_html=True)
                                        for stat_insight in chart['stat_insights']:
                                            formatted = stat_insight.replace("<b>", "<b>").replace("</b>", "</b>")
                                            st.markdown(f'<p>{formatted}</p>', unsafe_allow_html=True)
                                        st.markdown('</div>', unsafe_allow_html=True)

    # Custom Chart Plotting Section
    st.subheader("📈 Plot Custom Sales Charts")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_col = st.selectbox("Select X-axis column", df.columns)
    with col2:
        y_col = st.selectbox("Select Y-axis column (optional)", ["None"] + list(df.columns))
    with col3:
        z_col = st.selectbox("Select Z-axis or additional column (optional)", ["None"] + list(df.columns))
    
    selected_columns = [x_col]
    if y_col != "None":
        selected_columns.append(y_col)
    if z_col != "None":
        selected_columns.append(z_col)
    
    suggested_chart, suggestion_reason = suggest_chart_type(x_col, y_col, z_col, df)
    st.info(f"Suggested Chart: {suggested_chart} - {suggestion_reason}")
    
    chart_types = ["Bar", "Line", "Scatter", "Pie", "Scatter3D"]
    selected_chart = st.selectbox("Select Chart Type", chart_types, index=chart_types.index(suggested_chart))
    
    if st.button("Generate Chart"):
        try:
            fig = None
            if selected_chart == "Bar":
                if y_col != "None":
                    fig = px.bar(df, x=x_col, y=y_col, color=z_col if z_col != "None" else None, 
                                 title=f"Sales of {y_col} by {x_col}" + (f" colored by {z_col}" if z_col != "None" else ""))
                else:
                    fig = px.bar(df[x_col].value_counts().reset_index(), x="index", y=x_col, 
                                 title=f"Count of {x_col}")
            elif selected_chart == "Line":
                if y_col != "None":
                    fig = px.line(df, x=x_col, y=y_col, color=z_col if z_col != "None" else None, 
                                  title=f"Sales Trend of {y_col} over {x_col}" + (f" colored by {z_col}" if z_col != "None" else ""))
                else:
                    st.error("Line chart requires a Y-axis column.")
            elif selected_chart == "Scatter":
                if y_col != "None":
                    fig = px.scatter(df, x=x_col, y=y_col, color=z_col if z_col != "None" else None, 
                                     title=f"Sales Analysis: {y_col} vs {x_col}" + (f" colored by {z_col}" if z_col != "None" else ""))
                else:
                    st.error("Scatter chart requires a Y-axis column.")
            elif selected_chart == "Pie":
                if y_col == "None" and z_col == "None":
                    fig = px.pie(df, names=x_col, title=f"Distribution of {x_col}")
                    fig.update_layout(
                        clickmode='event+select'  # Enable selection for custom Pie chart
                    )
                else:
                    st.error("Pie chart does not require Y or Z-axis columns.")
            elif selected_chart == "Scatter3D":
                if y_col != "None" and z_col != "None":
                    fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, 
                                        title=f"3D Sales Analysis: {x_col} vs {y_col} vs {z_col}")
                else:
                    st.error("3D Scatter chart requires X, Y, and Z-axis columns.")
            
            if fig:
                fig.update_layout(
                    template="plotly_dark",
                    autosize=True,
                    margin=dict(l=5, r=5, t=20, b=5),
                    xaxis_title=x_col,
                    yaxis_title=y_col if y_col != "None" else "Count",
                    scene=dict(
                        xaxis_title=x_col,
                        yaxis_title=y_col,
                        zaxis_title=z_col if z_col != "None" else "Z",
                        xaxis=dict(tickfont=dict(size=8), title_font=dict(size=8)),
                        yaxis=dict(tickfont=dict(size=8), title_font=dict(size=8)),
                        zaxis=dict(tickfont=dict(size=8), title_font=dict(size=8))
                    ) if selected_chart == "Scatter3D" else {},
                    title_x=0.05,
                    title_font_size=15,
                    font=dict(size=15),
                    xaxis=dict(tickfont=dict(size=10), title_font=dict(size=10), automargin=True),
                    yaxis=dict(tickfont=dict(size=10), title_font=dict(size=10), automargin=True),
                    legend=dict(
                        font=dict(size=8),
                        orientation="h",
                        yanchor="bottom",
                        y=-0.1,
                        xanchor="center",
                        x=0.5
                    ),
                    height=chart_height
                )
                chart_id = str(uuid.uuid4())
                st.session_state.charts.append({
                    'id': chart_id,
                    'fig': fig,
                    'title': fig.layout.title.text,
                    'type': selected_chart,
                    'x_col': x_col,
                    'y_col': y_col,
                    'z_col': z_col,
                    'ai_insights': generate_ai_insights(df, selected_columns),
                    'stat_insights': generate_stat_insights(df, selected_columns)
                })
                st.session_state.chart_insights[chart_id] = False
                st.rerun()
        except Exception as e:
            st.error(f"Error generating chart: {e}")
    
    # Consolidated Insights Section and Navigation Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Upload"):
            st.session_state.page = "upload"
            st.session_state.charts = []
            st.session_state.chart_insights = {}
            st.session_state.active_drill_down = None
            st.rerun()
    with col2:
        st.button(
            "Toggle All Insights" if st.session_state.show_insights else "Show All Insights",
            key="toggle_all_insights",
            help="Show or hide all insights",
            on_click=lambda: setattr(st.session_state, 'show_insights', not st.session_state.show_insights)
        )

    if st.session_state.show_insights and st.session_state.charts:
        st.subheader("📝 Summarized Business Insights")
        with st.container():
            st.markdown('<div class="insights-container">', unsafe_allow_html=True)
            st.markdown(f'<h4>Dataset Overview</h4>', unsafe_allow_html=True)
            dataset_insights = generate_stat_insights(df)
            for insight in dataset_insights:
                formatted = insight.replace("<b>", "<b>").replace("</b>", "</b>")
                st.markdown(f'<p>{formatted}</p>', unsafe_allow_html=True)
            insights_text = "\n".join(dataset_insights)
            st.download_button(
                label="Download Insights",
                data=insights_text,
                file_name="business_insights.txt",
                mime="text/plain"
            )
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    #works final