import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌸 Iris Flower AI Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 16px; text-align: center;
        color: white; margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102,126,234,0.3);
    }
    .main-header h1 { font-size: 2.5rem; font-weight: 700; margin: 0; }
    .main-header p { font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem; }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8ecff 100%);
        border: 1px solid #dde3ff; border-radius: 12px;
        padding: 1.2rem; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-card h3 { color: #667eea; font-size: 2rem; font-weight: 700; margin: 0; }
    .metric-card p { color: #666; font-size: 0.85rem; margin: 0.3rem 0 0 0; }
    
    .species-card {
        border-radius: 12px; padding: 1rem 1.5rem;
        margin-bottom: 0.5rem; color: white; font-weight: 500;
    }
    .setosa { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .versicolor { background: linear-gradient(135deg, #2196F3, #21CBF3); }
    .virginica { background: linear-gradient(135deg, #f093fb, #f5576c); }
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border-radius: 16px; padding: 2rem;
        text-align: center; font-size: 1.4rem; font-weight: 600;
        box-shadow: 0 8px 25px rgba(102,126,234,0.4);
    }
    
    .chat-user {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border-radius: 18px 18px 4px 18px;
        padding: 0.8rem 1.2rem; margin: 0.5rem 0;
        max-width: 75%; float: right; clear: both;
    }
    .chat-bot {
        background: #f0f2ff; color: #333;
        border-radius: 18px 18px 18px 4px;
        padding: 0.8rem 1.2rem; margin: 0.5rem 0;
        max-width: 80%; float: left; clear: both;
        border: 1px solid #dde3ff;
    }
    .chat-container { min-height: 400px; max-height: 500px; overflow-y: auto; padding: 1rem; }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 500; font-size: 0.95rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important; border-radius: 8px !important;
    }
    
    .section-title {
        font-size: 1.3rem; font-weight: 600; color: #333;
        border-left: 4px solid #667eea; padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea15, #764ba215);
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Data & Train Models ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/Iris.csv")
        df.columns = [c.replace('Cm','').strip() for c in df.columns]
        if 'Id' in df.columns:
            df = df.drop('Id', axis=1)
        df['Species'] = df['Species'].str.replace('Iris-','')
    except:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=['SepalLength','SepalWidth','PetalLength','PetalWidth'])
        df['Species'] = [iris.target_names[t] for t in iris.target]
    return df

@st.cache_resource
def train_models(df):
    X = df.drop('Species', axis=1)
    y = df['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=500),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    trained, results = {}, {}
    for name, m in models.items():
        m.fit(X_train_sc, y_train)
        y_pred = m.predict(X_test_sc)
        trained[name] = m
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred, output_dict=True),
            'cm': confusion_matrix(y_test, y_pred)
        }
    return trained, results, scaler, X_test, y_test, X_train, y_train

df = load_data()
trained_models, results, scaler, X_test, y_test, X_train, y_train = train_models(df)
FEATURES = [c for c in df.columns if c != 'Species']
SPECIES = df['Species'].unique().tolist()
COLORS = {'setosa':'#11998e', 'versicolor':'#2196F3', 'virginica':'#f5576c'}

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:3rem;">🌸</div>
        <h2 style="color:#667eea; margin:0;">Iris AI</h2>
        <p style="color:#888; font-size:0.85rem;">Dashboard v2.0</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    page = st.radio("📌 Navigation", 
        ["🏠 Overview", "📊 Data Analysis", "🤖 ML Models", 
         "🔮 Live Prediction", "👁️ Computer Vision", "💬 AI Chatbot"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.info(f"**Total Samples:** {len(df)}\n\n**Features:** {len(FEATURES)}\n\n**Classes:** {len(SPECIES)}")
    
    st.markdown("### 🎯 Best Model")
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_acc = results[best_model_name]['accuracy']
    st.success(f"**{best_model_name}**\n\nAccuracy: `{best_acc:.1%}`")

# ─── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌸 Iris Flower Classification Dashboard</h1>
    <p>AI-Powered Analysis • Machine Learning • Computer Vision • Interactive Chatbot</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("150", "Total Samples"),
        ("4", "Features"),
        ("3", "Flower Species"),
        (f"{best_acc:.1%}", "Best Accuracy"),
    ]
    for col, (val, label) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{val}</h3>
                <p>{label}</p>
            </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown('<div class="section-title">Species Distribution</div>', unsafe_allow_html=True)
        counts = df['Species'].value_counts()
        fig = px.pie(values=counts.values, names=counts.index,
                     color=counts.index,
                     color_discrete_map=COLORS,
                     hole=0.45)
        fig.update_layout(height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown('<div class="section-title">Feature Overview (Mean by Species)</div>', unsafe_allow_html=True)
        mean_df = df.groupby('Species')[FEATURES].mean().reset_index()
        fig2 = px.bar(mean_df.melt(id_vars='Species'), x='variable', y='value',
                      color='Species', barmode='group',
                      color_discrete_map=COLORS)
        fig2.update_layout(height=320, margin=dict(t=20, b=20),
                           xaxis_title="Feature", yaxis_title="Mean (cm)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Species Info</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    infos = [
        ("setosa", "Iris Setosa", "Found in Arctic areas. Smallest petals. Easily separable from others. Best for beginners."),
        ("versicolor", "Iris Versicolor", "Common blue flag iris. Medium size. Found in North America. Slightly overlaps with Virginica."),
        ("virginica", "Iris Virginica", "Southern blue flag. Largest petals. Found in Eastern USA. Most overlap with Versicolor."),
    ]
    for col, (cls, title, desc) in zip([c1, c2, c3], infos):
        with col:
            st.markdown(f'<div class="species-card {cls}"><b>🌸 {title}</b><br><small>{desc}</small></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: DATA ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Analysis":
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Correlation", "📦 Box Plots", "🗂️ Raw Data"])
    
    with tab1:
        st.markdown('<div class="section-title">Feature Distributions</div>', unsafe_allow_html=True)
        fig = make_subplots(rows=2, cols=2, subplot_titles=FEATURES)
        positions = [(1,1),(1,2),(2,1),(2,2)]
        for feat, pos in zip(FEATURES, positions):
            for sp in SPECIES:
                vals = df[df['Species']==sp][feat]
                fig.add_trace(go.Histogram(x=vals, name=sp, opacity=0.6,
                    marker_color=COLORS.get(sp,'gray'), showlegend=(pos==(1,1))),
                    row=pos[0], col=pos[1])
        fig.update_layout(height=500, barmode='overlay', title_text="Feature Distributions by Species")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
        corr = df[FEATURES].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1, aspect='auto')
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="section-title">Box Plots by Species</div>', unsafe_allow_html=True)
        feat = st.selectbox("Select Feature", FEATURES)
        fig = px.box(df, x='Species', y=feat, color='Species',
                     color_discrete_map=COLORS, points='all',
                     notched=True)
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown('<div class="section-title">Raw Dataset</div>', unsafe_allow_html=True)
        filter_sp = st.multiselect("Filter Species", SPECIES, default=SPECIES)
        filtered = df[df['Species'].isin(filter_sp)]
        st.dataframe(filtered, use_container_width=True, height=400)
        st.info(f"Showing {len(filtered)} of {len(df)} rows")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: ML MODELS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Models":
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
    
    # Accuracy comparison
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    fig = px.bar(x=model_names, y=accuracies,
                 color=accuracies, color_continuous_scale='Purples',
                 text=[f"{a:.1%}" for a in accuracies])
    fig.update_traces(textposition='outside')
    fig.update_layout(height=350, yaxis_range=[0.8, 1.02],
                      xaxis_title="Model", yaxis_title="Accuracy",
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Detailed Model Analysis</div>', unsafe_allow_html=True)
    selected = st.selectbox("Select Model", model_names)
    res = results[selected]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Accuracy: `{res['accuracy']:.1%}`**")
        rep = res['report']
        rows = []
        for sp in SPECIES:
            if sp in rep:
                r = rep[sp]
                rows.append({'Species': sp, 'Precision': f"{r['precision']:.2f}",
                             'Recall': f"{r['recall']:.2f}", 'F1': f"{r['f1-score']:.2f}",
                             'Support': int(r['support'])})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    
    with col2:
        cm = res['cm']
        fig = px.imshow(cm, text_auto=True,
                        x=SPECIES, y=SPECIES,
                        color_continuous_scale='Purples',
                        labels=dict(x="Predicted", y="Actual"))
        fig.update_layout(height=320, title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance (Random Forest)
    if 'Random Forest' in trained_models:
        st.markdown('<div class="section-title">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
        importances = trained_models['Random Forest'].feature_importances_
        fig = px.bar(x=FEATURES, y=importances,
                     color=importances, color_continuous_scale='Purples',
                     text=[f"{v:.2f}" for v in importances])
        fig.update_traces(textposition='outside')
        fig.update_layout(height=320, coloraxis_showscale=False,
                          xaxis_title="Feature", yaxis_title="Importance")
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Prediction":
    st.markdown('<div class="section-title">Predict Iris Species</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        model_choice = st.selectbox("🤖 Choose Model", list(trained_models.keys()))
        st.markdown("**Enter Flower Measurements (cm):**")
        
        col_a, col_b = st.columns(2)
        with col_a:
            sl = st.slider("🌿 Sepal Length", 4.0, 8.0, 5.8, 0.1)
            pl = st.slider("🌺 Petal Length", 1.0, 7.0, 4.0, 0.1)
        with col_b:
            sw = st.slider("🍃 Sepal Width", 2.0, 4.5, 3.0, 0.1)
            pw = st.slider("🔍 Petal Width", 0.1, 2.5, 1.2, 0.1)
        
        if st.button("🔮 Predict Species", use_container_width=True, type="primary"):
            sample = np.array([[sl, sw, pl, pw]])
            sample_sc = scaler.transform(sample)
            prediction = trained_models[model_choice].predict(sample_sc)[0]
            proba = trained_models[model_choice].predict_proba(sample_sc)[0]
            classes = trained_models[model_choice].classes_
            
            color = COLORS.get(prediction, '#667eea')
            st.markdown(f"""
            <div class="prediction-result" style="background: linear-gradient(135deg, {color}, #764ba2);">
                🌸 Predicted Species<br>
                <span style="font-size:2rem;">Iris {prediction.capitalize()}</span>
            </div>""", unsafe_allow_html=True)
            
            st.markdown("<br>**Confidence Scores:**", unsafe_allow_html=True)
            for cls, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
                c = COLORS.get(cls, '#667eea')
                st.markdown(f"""
                <div style="margin:4px 0;">
                    <span style="width:120px;display:inline-block;font-weight:500;">{cls}</span>
                    <div style="display:inline-block;width:{p*200:.0f}px;height:12px;
                         background:{c};border-radius:6px;vertical-align:middle;"></div>
                    <span style="margin-left:8px;color:#666;">{p:.1%}</span>
                </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Your Input vs Dataset Range:**")
        input_vals = [sl, sw, pl, pw]
        fig = go.Figure()
        for i, (feat, val) in enumerate(zip(FEATURES, input_vals)):
            q25, q75 = df[feat].quantile([0.25, 0.75])
            mn, mx = df[feat].min(), df[feat].max()
            fig.add_trace(go.Scatter(
                x=[mn, q25, q75, mx], y=[i]*4,
                mode='lines', line=dict(color='#dde3ff', width=10),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[val], y=[i], mode='markers',
                marker=dict(size=14, color='#667eea', symbol='diamond'),
                name=feat, showlegend=True
            ))
        fig.update_layout(
            height=350, 
            yaxis=dict(tickvals=list(range(len(FEATURES))), ticktext=FEATURES),
            xaxis_title="Value (cm)", title="Input vs Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: COMPUTER VISION
# ════════════════════════════════════════════════════════════════════════════════
elif page == "👁️ Computer Vision":
    st.markdown('<div class="section-title">Computer Vision — Iris Image Analyzer</div>', unsafe_allow_html=True)
    
    st.info("📌 Upload an Iris flower image to analyze its visual features and predict species using CV + ML pipeline.")
    
    tab1, tab2, tab3 = st.tabs(["📷 Image Analysis", "🎨 Color Analysis", "📐 Shape Metrics"])
    
    with tab1:
        uploaded = st.file_uploader("Upload Iris Flower Image", type=['jpg','jpeg','png','webp'])
        
        if uploaded:
            from PIL import Image
            import io
            img = Image.open(uploaded).convert('RGB')
            img_arr = np.array(img)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.markdown("**📊 Image Metadata:**")
                st.json({
                    "Filename": uploaded.name,
                    "Size": f"{img.width} × {img.height} px",
                    "Format": img.format or "JPEG/PNG",
                    "Mode": img.mode,
                    "File Size": f"{uploaded.size / 1024:.1f} KB"
                })
                
                # Color stats
                r, g, b = img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]
                st.markdown("**🎨 RGB Channel Stats:**")
                stats_df = pd.DataFrame({
                    'Channel': ['Red', 'Green', 'Blue'],
                    'Mean': [r.mean(), g.mean(), b.mean()],
                    'Std': [r.std(), g.std(), b.std()],
                    'Max': [r.max(), g.max(), b.max()]
                })
                stats_df[['Mean','Std','Max']] = stats_df[['Mean','Std','Max']].round(2)
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
            
            # Histogram
            st.markdown("**RGB Histogram:**")
            fig = go.Figure()
            for ch_name, ch_data, color in [('Red', r.flatten(), 'red'), 
                                              ('Green', g.flatten(), 'green'), 
                                              ('Blue', b.flatten(), 'blue')]:
                hist, bins = np.histogram(ch_data, bins=50)
                fig.add_trace(go.Scatter(x=bins[:-1], y=hist, fill='tozeroy',
                    name=ch_name, line_color=color, opacity=0.5))
            fig.update_layout(height=300, xaxis_title="Pixel Value", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.markdown("""
            <div style="background:#f0f2ff;border:2px dashed #667eea;border-radius:12px;
                        padding:3rem;text-align:center;color:#667eea;">
                <div style="font-size:4rem;">📷</div>
                <h3>Upload an Iris flower image above</h3>
                <p>Supported formats: JPG, PNG, WebP</p>
            </div>""", unsafe_allow_html=True)
    
    with tab2:
        if uploaded:
            from PIL import Image
            img = Image.open(uploaded).convert('RGB')
            img_arr = np.array(img)
            img_small = np.array(img.resize((100, 100)))
            
            # Dominant colors
            pixels = img_small.reshape(-1, 3).astype(float)
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=6, random_state=42, n_init=10)
            km.fit(pixels)
            centers = km.cluster_centers_.astype(int)
            counts = np.bincount(km.labels_)
            sorted_idx = np.argsort(-counts)
            
            st.markdown("**🎨 Dominant Colors:**")
            cols = st.columns(6)
            for i, idx in enumerate(sorted_idx):
                r, g, b = centers[idx]
                pct = counts[idx] / len(km.labels_) * 100
                with cols[i]:
                    st.markdown(f"""
                    <div style="background:rgb({r},{g},{b});height:80px;border-radius:8px;
                                margin-bottom:4px;"></div>
                    <small style="text-align:center;display:block">
                        #{r:02x}{g:02x}{b:02x}<br>{pct:.1f}%
                    </small>""", unsafe_allow_html=True)
            
            # Heatmap of brightness
            st.markdown("**💡 Brightness Heatmap:**")
            gray = img_arr.mean(axis=2)
            fig = px.imshow(gray, color_continuous_scale='Purples',
                           labels=dict(color="Brightness"))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please upload an image in the Image Analysis tab first.")
    
    with tab3:
        st.markdown("**📐 Shape Descriptor Simulation:**")
        st.markdown("""
        For real flower images, the following shape features would be extracted
        using image processing techniques:
        """)
        
        # Simulate shape metrics for demo
        demo_data = {
            'Feature': ['Aspect Ratio', 'Circularity', 'Solidity', 'Extent', 
                       'Euler Number', 'Perimeter/Area Ratio'],
            'Setosa': [0.42, 0.78, 0.89, 0.71, 1, 0.15],
            'Versicolor': [0.58, 0.65, 0.82, 0.68, 1, 0.18],
            'Virginica': [0.71, 0.59, 0.79, 0.65, 1, 0.21]
        }
        demo_df = pd.DataFrame(demo_data)
        st.dataframe(demo_df, hide_index=True, use_container_width=True)
        
        fig = px.bar(demo_df.melt(id_vars='Feature', var_name='Species', value_name='Value'),
                     x='Feature', y='Value', color='Species',
                     barmode='group', color_discrete_map=COLORS)
        fig.update_layout(height=380, xaxis_tickangle=-15)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: AI CHATBOT
# ════════════════════════════════════════════════════════════════════════════════
elif page == "💬 AI Chatbot":
    st.markdown('<div class="section-title">Iris Expert AI Chatbot</div>', unsafe_allow_html=True)
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "bot", "text": "👋 Namaste! Main **Iris Expert AI** hoon. Aap mujhse Iris flowers, ML models, dataset stats, ya predictions ke baare mein kuch bhi pooch sakte hain!\n\n💡 **Try asking:**\n- _Setosa ki characteristics kya hain?_\n- _Best model kaunsa hai?_\n- _Petal length ka mean kya hai?_\n- _Species predict karo: 5.1, 3.5, 1.4, 0.2_"}
        ]
    
    # Chatbot logic
    def iris_chatbot(user_msg):
        msg = user_msg.lower().strip()
        
        # Species info
        if any(w in msg for w in ['setosa', 'सेटोसा']):
            stats = df[df['Species']=='setosa'][FEATURES].describe().loc['mean']
            return (f"🌸 **Iris Setosa** ke baare mein:\n\n"
                    f"- Avg Sepal Length: **{stats['SepalLength']:.2f} cm**\n"
                    f"- Avg Sepal Width: **{stats['SepalWidth']:.2f} cm**\n"
                    f"- Avg Petal Length: **{stats['PetalLength']:.2f} cm**\n"
                    f"- Avg Petal Width: **{stats['PetalWidth']:.2f} cm**\n\n"
                    f"✅ Setosa sabse alag hoti hai — easily classifiable!")
        
        if any(w in msg for w in ['versicolor', 'वर्सिकलर']):
            stats = df[df['Species']=='versicolor'][FEATURES].describe().loc['mean']
            return (f"💙 **Iris Versicolor** ke baare mein:\n\n"
                    f"- Avg Sepal Length: **{stats['SepalLength']:.2f} cm**\n"
                    f"- Avg Sepal Width: **{stats['SepalWidth']:.2f} cm**\n"
                    f"- Avg Petal Length: **{stats['PetalLength']:.2f} cm**\n"
                    f"- Avg Petal Width: **{stats['PetalWidth']:.2f} cm**\n\n"
                    f"ℹ️ Versicolor aur Virginica thoda overlap karte hain.")
        
        if any(w in msg for w in ['virginica', 'वर्जिनिका']):
            stats = df[df['Species']=='virginica'][FEATURES].describe().loc['mean']
            return (f"💜 **Iris Virginica** ke baare mein:\n\n"
                    f"- Avg Sepal Length: **{stats['SepalLength']:.2f} cm**\n"
                    f"- Avg Sepal Width: **{stats['SepalWidth']:.2f} cm**\n"
                    f"- Avg Petal Length: **{stats['PetalLength']:.2f} cm**\n"
                    f"- Avg Petal Width: **{stats['PetalWidth']:.2f} cm**\n\n"
                    f"👑 Virginica sabse badi petals wali species hai!")
        
        # Prediction from chat
        import re
        nums = re.findall(r'\d+\.?\d*', msg)
        if len(nums) >= 4 and any(w in msg for w in ['predict', 'classify', 'batao', 'species', 'kya hai', 'बताओ']):
            sl, sw, pl, pw = float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])
            sample = scaler.transform([[sl, sw, pl, pw]])
            pred = trained_models['Random Forest'].predict(sample)[0]
            proba = trained_models['Random Forest'].predict_proba(sample)[0]
            classes = trained_models['Random Forest'].classes_
            conf = max(proba)
            proba_str = '\n'.join([f"  - {c}: {p:.1%}" for c, p in zip(classes, proba)])
            return (f"🔮 **Prediction Result:**\n\n"
                    f"Input: SL={sl}, SW={sw}, PL={pl}, PW={pw}\n\n"
                    f"🌸 **Predicted: Iris {pred.capitalize()}**\n"
                    f"Confidence: **{conf:.1%}**\n\n"
                    f"All probabilities:\n{proba_str}")
        
        # Model accuracy
        if any(w in msg for w in ['accuracy', 'model', 'best', 'performance', 'सटीकता']):
            result_str = '\n'.join([f"- **{m}**: {r['accuracy']:.1%}" for m, r in results.items()])
            best = max(results, key=lambda x: results[x]['accuracy'])
            return (f"🤖 **Model Accuracies:**\n\n{result_str}\n\n"
                    f"🏆 Best model: **{best}** with **{results[best]['accuracy']:.1%}** accuracy!")
        
        # Dataset stats
        if any(w in msg for w in ['dataset', 'data', 'rows', 'samples', 'size', 'kitne']):
            return (f"📊 **Dataset Information:**\n\n"
                    f"- Total samples: **{len(df)}**\n"
                    f"- Features: **{len(FEATURES)}** ({', '.join(FEATURES)})\n"
                    f"- Species: **{len(SPECIES)}** ({', '.join(SPECIES)})\n"
                    f"- Train/Test split: **80/20**\n"
                    f"- Source: Classic Fisher's Iris dataset")
        
        # Feature stats
        for feat in FEATURES:
            if feat.lower() in msg or feat.lower().replace('sepal','').replace('petal','').strip() in msg:
                stats = df[feat].describe()
                return (f"📈 **{feat} Statistics:**\n\n"
                        f"- Mean: **{stats['mean']:.3f} cm**\n"
                        f"- Std Dev: **{stats['std']:.3f}**\n"
                        f"- Min: **{stats['min']:.3f}** | Max: **{stats['max']:.3f}**\n"
                        f"- Q1: **{stats['25%']:.3f}** | Q3: **{stats['75%']:.3f}**")
        
        # Help
        if any(w in msg for w in ['help', 'kya', 'what', 'pooch', 'sakte', 'मदद']):
            return ("💡 **Main ye cheezein jaanta hoon:**\n\n"
                    "1️⃣ **Species info** — _setosa, versicolor, virginica_\n"
                    "2️⃣ **Predictions** — _'predict 5.1 3.5 1.4 0.2'_\n"
                    "3️⃣ **Model accuracy** — _'best model kaunsa hai'_\n"
                    "4️⃣ **Feature stats** — _'petal length ka mean'_\n"
                    "5️⃣ **Dataset info** — _'kitne samples hain'_\n\n"
                    "Kuch bhi poochh sakte ho! 🌸")
        
        # Greetings
        if any(w in msg for w in ['hello', 'hi', 'hey', 'namaste', 'नमस्ते']):
            return "👋 Namaste! Iris flowers ke baare mein kuch bhi poochhein. Main yahaan hoon! 🌸"
        
        # Default
        return ("🤔 Mujhe bilkul samjha nahi. Try karo:\n\n"
                "- _'setosa ki info do'_\n"
                "- _'predict 5.0 3.4 1.5 0.2'_\n"
                "- _'best model kaun sa hai'_\n"
                "- _'help'_")
    
    # Chat UI
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"""
                <div style="display:flex;justify-content:flex-end;margin:8px 0;">
                    <div class="chat-user">👤 {msg['text']}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display:flex;justify-content:flex-start;margin:8px 0;">
                    <div class="chat-bot">🌸 {msg['text']}</div>
                </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick suggestion buttons
    st.markdown("**💡 Quick Questions:**")
    qs = ["Setosa ki info", "Best model kaunsa?", "predict 5.1 3.5 1.4 0.2", "Dataset size?", "Virginica features"]
    cols = st.columns(len(qs))
    for col, q in zip(cols, qs):
        with col:
            if st.button(q, use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "text": q})
                response = iris_chatbot(q)
                st.session_state.chat_history.append({"role": "bot", "text": response})
                st.rerun()
    
    # Input
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        user_input = st.text_input("Type your question...", key="chat_input", 
                                   placeholder="e.g., setosa ki characteristics kya hain?",
                                   label_visibility="collapsed")
    with col_btn:
        send = st.button("Send 📨", use_container_width=True, type="primary")
    
    if (send or user_input) and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "text": user_input})
        response = iris_chatbot(user_input)
        st.session_state.chat_history.append({"role": "bot", "text": response})
        st.rerun()
    
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.chat_history = [
            {"role": "bot", "text": "👋 Chat clear ho gayi! Naya sawaal poochhein 🌸"}
        ]
        st.rerun()
