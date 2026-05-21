import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BigMart Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --ink:    #0a0a0f;
    --paper:  #f7f5f0;
    --cream:  #ede9df;
    --gold:   #c9a84c;
    --gold2:  #f0d080;
    --blue:   #1c3f6e;
    --red:    #c0392b;
    --green:  #1a6b4a;
    --muted:  #7a7a8a;
    --border: #e2ddd4;
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background: var(--paper);
}

[data-testid="stSidebar"] {
    background: #0a0a0f !important;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * { color: #d0d0e0 !important; }

[data-testid="stSidebar"] .stRadio > div { gap: 4px; }
[data-testid="stSidebar"] .stRadio label {
    background: #14141e !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    font-size: .85rem !important;
    font-weight: 500 !important;
    border: 1px solid #2a2a3a !important;
    display: block; width: 100%;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #1e1e30 !important;
    border-color: #c9a84c !important;
}

.main .block-container { padding: 0 2rem 3rem; max-width: 1400px; }

/* HERO */
.hero {
    background: #0a0a0f; border-radius: 24px;
    padding: 60px 56px; margin: 24px 0 32px;
    position: relative; overflow: hidden;
}
.hero::before {
    content:''; position:absolute; top:-80px; right:-80px;
    width:420px; height:420px;
    background:radial-gradient(circle,rgba(201,168,76,.18) 0%,transparent 70%);
    pointer-events:none;
}
.hero::after {
    content:''; position:absolute; bottom:-60px; left:30%;
    width:300px; height:300px;
    background:radial-gradient(circle,rgba(28,63,110,.3) 0%,transparent 70%);
    pointer-events:none;
}
.hero-tag {
    font-size:.72rem; font-weight:600; letter-spacing:.18em;
    text-transform:uppercase; color:#c9a84c; margin-bottom:16px; display:block;
}
.hero-title {
    font-family:'Bebas Neue',sans-serif; font-size:5rem;
    line-height:.95; color:white; margin:0 0 20px; letter-spacing:.02em;
}
.hero-title span { color:#c9a84c; }
.hero-desc {
    font-size:1rem; color:#a0a0b8; max-width:520px;
    line-height:1.7; margin-bottom:36px;
}
.hero-stats { display:flex; gap:40px; flex-wrap:wrap; }
.hero-stat-val {
    font-family:'Bebas Neue',sans-serif;
    font-size:2.4rem; color:white; line-height:1;
}
.hero-stat-label {
    font-size:.72rem; color:#606078; letter-spacing:.1em;
    text-transform:uppercase; margin-top:4px;
}
.hero-badge {
    position:absolute; top:32px; right:40px;
    background:rgba(201,168,76,.15); border:1px solid rgba(201,168,76,.3);
    border-radius:999px; padding:8px 20px;
    font-size:.78rem; color:#c9a84c; font-weight:600; letter-spacing:.06em;
}

/* Feature cards */
.feat-grid {
    display:grid; grid-template-columns:repeat(4,1fr);
    gap:16px; margin:0 0 32px;
}
.feat-card {
    background:white; border-radius:16px; padding:24px 20px;
    border:1px solid #e2ddd4; transition:transform .2s,box-shadow .2s;
}
.feat-card:hover { transform:translateY(-4px); box-shadow:0 12px 40px rgba(0,0,0,.1); }
.feat-icon { font-size:1.8rem; margin-bottom:12px; }
.feat-title { font-weight:700; font-size:.95rem; color:#0a0a0f; margin-bottom:6px; }
.feat-desc { font-size:.8rem; color:#7a7a8a; line-height:1.5; }

/* KPI cards */
.kpi-row { display:flex; gap:16px; margin-bottom:28px; flex-wrap:wrap; }
.kpi-card {
    flex:1; min-width:180px; background:white; border-radius:16px;
    padding:20px 22px; border:1px solid #e2ddd4; border-top:4px solid;
    position:relative; overflow:hidden;
}
.kpi-card::after {
    content:attr(data-icon); position:absolute; right:16px; bottom:12px;
    font-size:2.2rem; opacity:.07;
}
.kpi-label {
    font-size:.7rem; font-weight:600; letter-spacing:.1em;
    text-transform:uppercase; color:#7a7a8a; margin-bottom:6px;
}
.kpi-value {
    font-family:'Bebas Neue',sans-serif; font-size:2.1rem;
    color:#0a0a0f; line-height:1;
}
.kpi-sub { font-size:.75rem; color:#b0b0c0; margin-top:4px; }

/* Section title */
.sec-title {
    font-family:'Bebas Neue',sans-serif; font-size:1.6rem;
    letter-spacing:.04em; color:#0a0a0f; margin:32px 0 16px;
    display:flex; align-items:center; gap:10px;
}
.sec-title::after {
    content:''; flex:1; height:1px;
    background:#e2ddd4; margin-left:8px;
}

/* Model cards */
.model-card {
    background:white; border-radius:16px; padding:22px;
    border:1px solid #e2ddd4; text-align:center; transition:transform .2s;
}
.model-card:hover { transform:translateY(-3px); }
.model-card.best { border-color:#c9a84c; background:#fffdf5; }
.model-name { font-weight:700; font-size:.85rem; color:#7a7a8a; margin-bottom:8px; }
.model-r2 { font-family:'Bebas Neue',sans-serif; font-size:2.4rem; line-height:1; }
.model-rmse { font-size:.78rem; color:#7a7a8a; margin-top:6px; }
.best-crown { font-size:.75rem; color:#c9a84c; font-weight:700; margin-bottom:4px; }

/* Insight cards */
.insight-row { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-bottom:28px; }
.insight-card { background:white; border-radius:14px; padding:20px; border:1px solid #e2ddd4; }
.insight-card .i-val { font-family:'Bebas Neue',sans-serif; font-size:1.6rem; color:#0a0a0f; }
.insight-card .i-label { font-size:.78rem; color:#7a7a8a; }

/* Predict result */
.pred-result {
    background:#0a0a0f; border-radius:20px; padding:40px;
    text-align:center; position:relative; overflow:hidden;
}
.pred-result::before {
    content:''; position:absolute; top:-60px; left:50%; transform:translateX(-50%);
    width:300px; height:300px;
    background:radial-gradient(circle,rgba(201,168,76,.2) 0%,transparent 70%);
}
.pred-amount {
    font-family:'Bebas Neue',sans-serif; font-size:4rem;
    color:#f0d080; letter-spacing:.02em; position:relative;
}
.pred-label { font-size:.85rem; color:#7070a0; margin-top:8px; }
.pred-model-tag {
    display:inline-block; margin-top:14px;
    background:rgba(255,255,255,.08); border-radius:999px;
    padding:6px 16px; font-size:.75rem; color:#9090b8;
}

#MainMenu { visibility:hidden; }
footer { visibility:hidden; }
header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODELS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    df = pd.read_csv("clean_1_train.csv")
    cat_cols = ['Item_Identifier','Item_Fat_Content','Item_Type',
                'Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type']
    df_enc = df.copy()
    for c in cat_cols:
        df_enc[c] = LabelEncoder().fit_transform(df_enc[c].astype(str))

    drop_cols = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type']
    X = df_enc.drop(columns=drop_cols + ['Item_Outlet_Sales'])
    y = df_enc['Item_Outlet_Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return df, df_enc, X_train_s, X_test_s, y_train, y_test, scaler, X.columns.tolist()

@st.cache_resource
def train_models(Xtr, Xte, ytr, yte):
    res = {}
    for name, mdl, color in [
        ('Linear Regression', LinearRegression(), '#1c3f6e'),
        ('Random Forest',     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1), '#1a6b4a'),
        ('Decision Tree',     DecisionTreeRegressor(criterion='squared_error', random_state=42), '#8338ec'),
    ]:
        mdl.fit(Xtr, ytr); p = mdl.predict(Xte)
        res[name] = {'model':mdl,'pred':p,'r2':r2_score(yte,p),
                     'rmse':np.sqrt(mean_squared_error(yte,p)),'color':color}
    if XGBOOST_OK:
        xgb = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
        xgb.fit(Xtr, ytr); p = xgb.predict(Xte)
        res['XGBoost'] = {'model':xgb,'pred':p,'r2':r2_score(yte,p),
                          'rmse':np.sqrt(mean_squared_error(yte,p)),'color':'#c0392b'}
    return res

def chart(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafaf8')
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#ddd')
    ax.tick_params(colors='#777', labelsize=8)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:24px 8px 28px;text-align:center;'>
        <div style='font-family:"Bebas Neue",sans-serif;font-size:2rem;color:#c9a84c;letter-spacing:.06em;'>BIGMART</div>
        <div style='font-size:.65rem;letter-spacing:.2em;color:#444;margin-top:2px;'>ANALYTICS PLATFORM</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Home", "📊  Dashboard", "🔍  EDA", "🤖  Models", "🎯  Predict"
    ], label_visibility="collapsed")

    st.markdown("<hr style='margin:20px 0;border-color:#1e1e2e'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.7rem;color:#333;padding:0 8px;line-height:2;'>
        <div>📁 Dataset: <b style='color:#555'>Big Mart Sales</b></div>
        <div>📝 Records: <b style='color:#555'>8,519</b></div>
        <div>🔢 Features: <b style='color:#555'>11</b></div>
        <div>🤖 Models: <b style='color:#555'>4</b></div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading & training…"):
    df, df_enc, X_train, X_test, y_train, y_test, scaler, feature_cols = load_and_prepare()
    models = train_models(X_train, X_test, y_train, y_test)

best_name   = max(models, key=lambda m: models[m]['r2'])
total_sales = df['Item_Outlet_Sales'].sum()
avg_sales   = df['Item_Outlet_Sales'].mean()


# ═════════════════════════════════════════════════════════════════════════════
# HOME
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":

    st.markdown(f"""
    <div class='hero'>
        <span class='hero-badge'>🛒 Retail Intelligence</span>
        <span class='hero-tag'>Big Mart Sales · Machine Learning Dashboard</span>
        <div class='hero-title'>SALES<br><span>INTELLIGENCE</span><br>PLATFORM</div>
        <p class='hero-desc'>
            A complete analytics suite for Big Mart outlet performance —
            explore trends, compare ML models, and predict item-level sales
            using Linear Regression, Random Forest, Decision Tree &amp; XGBoost.
        </p>
        <div class='hero-stats'>
            <div>
                <div class='hero-stat-val'>₹{total_sales/1e6:.1f}M</div>
                <div class='hero-stat-label'>Total Sales</div>
            </div>
            <div>
                <div class='hero-stat-val'>8,519</div>
                <div class='hero-stat-label'>Records</div>
            </div>
            <div>
                <div class='hero-stat-val'>{len(models)}</div>
                <div class='hero-stat-label'>ML Models</div>
            </div>
            <div>
                <div class='hero-stat-val'>{models[best_name]['r2']*100:.1f}%</div>
                <div class='hero-stat-label'>Best R²</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='feat-grid'>
        <div class='feat-card'>
            <div class='feat-icon'>📊</div>
            <div class='feat-title'>Dashboard</div>
            <div class='feat-desc'>KPI overview, outlet comparisons, tier analysis and product charts.</div>
        </div>
        <div class='feat-card'>
            <div class='feat-icon'>🔍</div>
            <div class='feat-title'>EDA</div>
            <div class='feat-desc'>Distributions, boxplots, scatter plots and full correlation heatmap.</div>
        </div>
        <div class='feat-card'>
            <div class='feat-icon'>🤖</div>
            <div class='feat-title'>Models</div>
            <div class='feat-desc'>R² / RMSE comparison, actual vs predicted, residuals and feature importance.</div>
        </div>
        <div class='feat-card'>
            <div class='feat-icon'>🎯</div>
            <div class='feat-title'>Predict</div>
            <div class='feat-desc'>Enter item & outlet parameters and get instant predictions from all 4 models.</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Quick insights
    st.markdown("<div class='sec-title'>⚡ Quick Insights</div>", unsafe_allow_html=True)
    top_type = df.groupby('Outlet_Type')['Item_Outlet_Sales'].mean().idxmax()
    top_item = df.groupby('Item_Type')['Item_Outlet_Sales'].mean().idxmax()
    top_loc  = df.groupby('Outlet_Location_Type')['Item_Outlet_Sales'].sum().idxmax()
    oldest   = df['Outlet_Establishment_Year'].min()

    st.markdown(f"""
    <div class='insight-row'>
        <div class='insight-card'>
            <div class='i-val'>🏪 {top_type}</div>
            <div class='i-label'>Highest avg sales outlet type</div>
        </div>
        <div class='insight-card'>
            <div class='i-val'>🥦 {top_item}</div>
            <div class='i-label'>Top selling item category</div>
        </div>
        <div class='insight-card'>
            <div class='i-val'>📍 {top_loc}</div>
            <div class='i-label'>Location with most total sales</div>
        </div>
        <div class='insight-card'>
            <div class='i-val'>🗓 {oldest}</div>
            <div class='i-label'>Oldest outlet year</div>
        </div>
        <div class='insight-card'>
            <div class='i-val'>💰 ₹{avg_sales:,.0f}</div>
            <div class='i-label'>Average item outlet sales</div>
        </div>
        <div class='insight-card'>
            <div class='i-val'>🏆 {best_name}</div>
            <div class='i-label'>Best ML model</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Snapshot charts
    st.markdown("<div class='sec-title'>📈 Sales Snapshot</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        fig, ax = chart(5, 3.5)
        ot = df.groupby('Outlet_Type')['Item_Outlet_Sales'].mean().sort_values()
        ax.barh(ot.index, ot.values,
                color=['#1c3f6e','#1a6b4a','#c9a84c','#c0392b'][:len(ot)], height=0.5)
        ax.set_title('Avg Sales by Outlet Type', fontsize=9, fontweight='bold', color='#333', pad=8)
        st.pyplot(fig, use_container_width=True); plt.close()

    with c2:
        fig, ax = chart(5, 3.5)
        loc = df.groupby('Outlet_Location_Type')['Item_Outlet_Sales'].mean()
        ax.pie(loc.values, labels=loc.index, autopct='%1.0f%%',
               colors=['#1c3f6e','#c9a84c','#1a6b4a'],
               wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2),
               startangle=140, textprops={'fontsize':8})
        ax.set_title('Sales Share by Tier', fontsize=9, fontweight='bold', color='#333', pad=8)
        ax.set_facecolor('white')
        st.pyplot(fig, use_container_width=True); plt.close()

    with c3:
        fig, ax = chart(5, 3.5)
        ax.hist(df['Item_Outlet_Sales'], bins=35, color='#1c3f6e', edgecolor='white', alpha=.85)
        ax.set_title('Sales Distribution', fontsize=9, fontweight='bold', color='#333', pad=8)
        st.pyplot(fig, use_container_width=True); plt.close()

    # Leaderboard
    st.markdown("<div class='sec-title'>🏆 Model Leaderboard</div>", unsafe_allow_html=True)
    cols = st.columns(len(models))
    for idx, (mname, minfo) in enumerate(sorted(models.items(), key=lambda x: -x[1]['r2'])):
        is_best = mname == best_name
        with cols[idx]:
            st.markdown(f"""
            <div class='model-card {"best" if is_best else ""}'>
                {"<div class='best-crown'>👑 BEST MODEL</div>" if is_best else ""}
                <div class='model-name'>{mname}</div>
                <div class='model-r2' style='color:{minfo["color"]}'>{minfo["r2"]*100:.2f}%</div>
                <div style='font-size:.65rem;color:#ccc;margin:2px 0 6px'>R² SCORE</div>
                <div class='model-rmse'>RMSE: ₹{minfo["rmse"]:,.1f}</div>
            </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊  Dashboard":
    st.markdown("""
    <div style='padding:24px 0 8px'>
        <span style='font-family:"Bebas Neue",sans-serif;font-size:2.4rem;letter-spacing:.04em;color:#0a0a0f'>SALES DASHBOARD</span><br>
        <span style='color:#999;font-size:.9rem'>Outlet Performance &amp; Product Analytics</span>
    </div>""", unsafe_allow_html=True)

    num_items   = df['Item_Identifier'].nunique()
    num_outlets = df['Outlet_Identifier'].nunique()
    max_sales   = df['Item_Outlet_Sales'].max()

    st.markdown(f"""
    <div class='kpi-row'>
        <div class='kpi-card' style='border-top-color:#c9a84c' data-icon='₹'>
            <div class='kpi-label'>Total Revenue</div>
            <div class='kpi-value'>₹{total_sales/1e6:.2f}M</div>
            <div class='kpi-sub'>Across all outlets</div>
        </div>
        <div class='kpi-card' style='border-top-color:#1c3f6e' data-icon='~'>
            <div class='kpi-label'>Avg Item Sales</div>
            <div class='kpi-value'>₹{avg_sales:,.0f}</div>
            <div class='kpi-sub'>Per item per outlet</div>
        </div>
        <div class='kpi-card' style='border-top-color:#1a6b4a' data-icon='📦'>
            <div class='kpi-label'>Unique Items</div>
            <div class='kpi-value'>{num_items:,}</div>
            <div class='kpi-sub'>Product SKUs</div>
        </div>
        <div class='kpi-card' style='border-top-color:#c0392b' data-icon='🏪'>
            <div class='kpi-label'>Outlets</div>
            <div class='kpi-value'>{num_outlets}</div>
            <div class='kpi-sub'>Active stores</div>
        </div>
        <div class='kpi-card' style='border-top-color:#8338ec' data-icon='⬆'>
            <div class='kpi-label'>Peak Sales</div>
            <div class='kpi-value'>₹{max_sales:,.0f}</div>
            <div class='kpi-sub'>Single item record</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-title'>🏪 Outlet Performance</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        pivot = df.groupby('Outlet_Type')['Item_Outlet_Sales'].mean().sort_values()
        fig, ax = chart(7, 3.8)
        colors_ = ['#1c3f6e','#1a6b4a','#c9a84c','#c0392b']
        bars = ax.barh(pivot.index, pivot.values, color=colors_[:len(pivot)], height=0.5)
        for bar, val in zip(bars, pivot.values):
            ax.text(val+30, bar.get_y()+bar.get_height()/2,
                    f'₹{val:,.0f}', va='center', fontsize=8, color='#555')
        ax.set_title('Average Sales by Outlet Type', fontsize=10, fontweight='bold', color='#333', pad=10)
        ax.set_xlabel('Avg Sales (₹)', fontsize=8, color='#999')
        st.pyplot(fig, use_container_width=True); plt.close()

    with c2:
        sz = df.groupby('Outlet_Size')['Item_Outlet_Sales'].mean()
        fig, ax = chart(4.5, 3.8)
        c_map = {'Small':'#c9a84c','Medium':'#1c3f6e','High':'#1a6b4a'}
        ax.bar(sz.index, sz.values,
               color=[c_map.get(s,'#888') for s in sz.index], width=0.5)
        for i, (x, y) in enumerate(zip(sz.index, sz.values)):
            ax.text(i, y+20, f'₹{y:,.0f}', ha='center', fontsize=8, color='#555')
        ax.set_title('Avg Sales by Outlet Size', fontsize=10, fontweight='bold', color='#333', pad=10)
        ax.set_ylabel('₹', fontsize=8, color='#999')
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<div class='sec-title'>📦 Product Analytics</div>", unsafe_allow_html=True)
    item_sales = df.groupby('Item_Type')['Item_Outlet_Sales'].mean().sort_values(ascending=False)
    fig, ax = chart(12, 4)
    ax.bar(item_sales.index, item_sales.values,
           color=plt.cm.YlGnBu(np.linspace(0.3, 0.9, len(item_sales))), width=0.65)
    ax.set_ylabel('Avg Sales (₹)', fontsize=8, color='#999')
    ax.set_title('Average Sales per Item Type', fontsize=10, fontweight='bold', color='#333', pad=10)
    ax.tick_params(axis='x', rotation=45, labelsize=7.5)
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<div class='sec-title'>📅 Establishment Year Trend</div>", unsafe_allow_html=True)
    yr = df.groupby('Outlet_Establishment_Year')['Item_Outlet_Sales'].mean()
    fig, ax = chart(10, 3.5)
    ax.plot(yr.index, yr.values, color='#1c3f6e', lw=2.5, marker='o',
            markersize=6, markerfacecolor='#c9a84c', markeredgecolor='white', markeredgewidth=1.5)
    ax.fill_between(yr.index, yr.values, alpha=0.08, color='#1c3f6e')
    ax.set_xlabel('Year', fontsize=8, color='#999')
    ax.set_ylabel('Avg Sales (₹)', fontsize=8, color='#999')
    ax.set_title('Average Sales by Outlet Establishment Year', fontsize=10, fontweight='bold', color='#333', pad=10)
    st.pyplot(fig, use_container_width=True); plt.close()

    c1, c2 = st.columns(2)
    with c1:
        fat = df.groupby('Item_Fat_Content')['Item_Outlet_Sales'].mean()
        fig, ax = chart(5, 3)
        ax.bar(fat.index, fat.values,
               color=['#1a6b4a','#c9a84c','#1c3f6e','#c0392b'][:len(fat)], width=0.45)
        ax.set_title('Sales by Fat Content', fontsize=9, fontweight='bold', color='#333')
        ax.set_ylabel('₹', fontsize=8, color='#999')
        st.pyplot(fig, use_container_width=True); plt.close()

    with c2:
        loc = df.groupby('Outlet_Location_Type')['Item_Outlet_Sales'].mean()
        fig, ax = chart(5, 3)
        ax.bar(loc.index, loc.values,
               color=['#1c3f6e','#c9a84c','#c0392b'][:len(loc)], width=0.45)
        ax.set_title('Sales by Location Type', fontsize=9, fontweight='bold', color='#333')
        ax.set_ylabel('₹', fontsize=8, color='#999')
        st.pyplot(fig, use_container_width=True); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# EDA
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍  EDA":
    st.markdown("""
    <div style='padding:24px 0 8px'>
        <span style='font-family:"Bebas Neue",sans-serif;font-size:2.4rem;letter-spacing:.04em;color:#0a0a0f'>EXPLORATORY DATA ANALYSIS</span>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋  Overview", "📈  Distributions", "🔗  Correlations"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Type':   df.dtypes.astype(str).values,
                'Non-Null': df.count().values,
                'Unique': df.nunique().values,
                'Nulls':  df.isnull().sum().values,
            })
            st.markdown("**Column Info**")
            st.dataframe(info_df, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Descriptive Statistics**")
            st.dataframe(df.describe().round(3), use_container_width=True)
        st.markdown("**Data Preview**")
        st.dataframe(df.head(100), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = chart(6, 4)
            ax.hist(df['Item_Outlet_Sales'], bins=45, color='#1c3f6e', edgecolor='white', alpha=.9)
            ax.set_title('Sales Distribution', fontsize=10, fontweight='bold', color='#333')
            ax.set_xlabel('₹', fontsize=9); ax.set_ylabel('Count', fontsize=9)
            st.pyplot(fig, use_container_width=True); plt.close()
        with c2:
            fig, ax = chart(6, 4)
            ax.hist(df['Item_MRP'], bins=45, color='#c9a84c', edgecolor='white', alpha=.9)
            ax.set_title('Item MRP Distribution', fontsize=10, fontweight='bold', color='#333')
            ax.set_xlabel('₹', fontsize=9); ax.set_ylabel('Count', fontsize=9)
            st.pyplot(fig, use_container_width=True); plt.close()

        fig, ax = chart(11, 4.5)
        sns.boxplot(data=df, x='Outlet_Type', y='Item_Outlet_Sales',
                    palette={'Grocery Store':'#1c3f6e','Supermarket Type1':'#1a6b4a',
                             'Supermarket Type2':'#c9a84c','Supermarket Type3':'#c0392b'},
                    ax=ax, width=0.5, flierprops=dict(marker='o',markersize=2,alpha=.4))
        ax.set_title('Sales by Outlet Type (Boxplot)', fontsize=11, fontweight='bold', color='#333')
        ax.set_xlabel(''); ax.set_ylabel('Sales (₹)', fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = chart(6, 4)
            ax.scatter(df['Item_MRP'], df['Item_Outlet_Sales'], alpha=.2, s=8, color='#1c3f6e')
            ax.set_title('MRP vs Sales', fontsize=10, fontweight='bold', color='#333')
            ax.set_xlabel('Item MRP (₹)', fontsize=9); ax.set_ylabel('Sales (₹)', fontsize=9)
            st.pyplot(fig, use_container_width=True); plt.close()
        with c2:
            fig, ax = chart(6, 4)
            ax.scatter(df['Item_Visibility'], df['Item_Outlet_Sales'], alpha=.2, s=8, color='#1a6b4a')
            ax.set_title('Visibility vs Sales', fontsize=10, fontweight='bold', color='#333')
            ax.set_xlabel('Visibility', fontsize=9); ax.set_ylabel('Sales (₹)', fontsize=9)
            st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        num_cols = df_enc.select_dtypes(include=np.number).columns.tolist()
        corr = df_enc[num_cols].corr()
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.patch.set_facecolor('white')
        sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)),
                    annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax,
                    linewidths=0.5, annot_kws={'size':8}, square=True)
        ax.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold', color='#333', pad=14)
        st.pyplot(fig, use_container_width=True); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# MODELS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Models":
    st.markdown("""
    <div style='padding:24px 0 8px'>
        <span style='font-family:"Bebas Neue",sans-serif;font-size:2.4rem;letter-spacing:.04em;color:#0a0a0f'>MODEL EVALUATION</span>
    </div>""", unsafe_allow_html=True)

    cols = st.columns(len(models))
    for idx, (mname, minfo) in enumerate(sorted(models.items(), key=lambda x: -x[1]['r2'])):
        is_best = mname == best_name
        with cols[idx]:
            st.markdown(f"""
            <div class='model-card {"best" if is_best else ""}'>
                {"<div class='best-crown'>👑 BEST</div>" if is_best else ""}
                <div class='model-name'>{mname}</div>
                <div class='model-r2' style='color:{minfo["color"]}'>{minfo["r2"]*100:.4f}%</div>
                <div style='font-size:.65rem;color:#bbb;margin:2px 0 6px'>R² SCORE</div>
                <div class='model-rmse'>RMSE ₹{minfo["rmse"]:,.2f}</div>
                <div class='model-rmse'>MSE ₹{minfo["rmse"]**2:,.1f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-title'>📊 Performance Comparison</div>", unsafe_allow_html=True)
    names   = list(models.keys())
    colors_ = [models[m]['color'] for m in names]
    c1, c2  = st.columns(2)

    with c1:
        fig, ax = chart(6, 4)
        r2s = [models[m]['r2']*100 for m in names]
        bars = ax.bar(names, r2s, color=colors_, width=0.5)
        for bar, val in zip(bars, r2s):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.003,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=8, color='#555')
        ax.set_ylabel('R² (%)', fontsize=8, color='#999')
        ax.set_title('R² Score Comparison', fontsize=10, fontweight='bold', color='#333')
        ax.set_ylim([min(r2s)*.985, 101])
        ax.tick_params(axis='x', rotation=15, labelsize=8)
        st.pyplot(fig, use_container_width=True); plt.close()

    with c2:
        fig, ax = chart(6, 4)
        rmses = [models[m]['rmse'] for m in names]
        bars  = ax.bar(names, rmses, color=colors_, width=0.5)
        for bar, val in zip(bars, rmses):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f'₹{val:,.0f}', ha='center', va='bottom', fontsize=8, color='#555')
        ax.set_ylabel('RMSE (₹)', fontsize=8, color='#999')
        ax.set_title('RMSE Comparison', fontsize=10, fontweight='bold', color='#333')
        ax.tick_params(axis='x', rotation=15, labelsize=8)
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<div class='sec-title'>🎯 Actual vs Predicted</div>", unsafe_allow_html=True)
    sel     = st.selectbox("Select model", list(models.keys()))
    preds_  = models[sel]['pred']
    actuals = y_test.values
    color_  = models[sel]['color']

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = chart(6, 4.5)
        ax.scatter(actuals, preds_, alpha=.25, s=10, color=color_)
        lims = [min(actuals.min(), preds_.min())*.9, max(actuals.max(), preds_.max())*1.05]
        ax.plot(lims, lims, 'k--', lw=1.2, alpha=.5, label='Perfect fit')
        ax.set_xlabel('Actual (₹)', fontsize=9); ax.set_ylabel('Predicted (₹)', fontsize=9)
        ax.set_title(f'{sel} — Actual vs Predicted', fontsize=10, fontweight='bold', color='#333')
        ax.legend(fontsize=8)
        st.pyplot(fig, use_container_width=True); plt.close()

    with c2:
        fig, ax = chart(6, 4.5)
        ax.hist(actuals - preds_, bins=45, color=color_, edgecolor='white', alpha=.85)
        ax.axvline(0, color='#333', lw=1.5, linestyle='--')
        ax.set_xlabel('Residual (₹)', fontsize=9); ax.set_ylabel('Count', fontsize=9)
        ax.set_title(f'{sel} — Residuals', fontsize=10, fontweight='bold', color='#333')
        st.pyplot(fig, use_container_width=True); plt.close()

    if 'Random Forest' in models:
        st.markdown("<div class='sec-title'>🌳 Feature Importance (Random Forest)</div>",
                    unsafe_allow_html=True)
        rf_model = models['Random Forest']['model']
        feat_names = ['Item_Identifier','Item_Weight','Item_Visibility',
                      'Item_MRP','Outlet_Identifier','Outlet_Est_Year']
        imps = rf_model.feature_importances_
        fi   = pd.DataFrame({'Feature':feat_names[:len(imps)],'Imp':imps}).sort_values('Imp')
        fig, ax = chart(9, 3.5)
        ax.barh(fi['Feature'], fi['Imp'],
                color=plt.cm.YlGnBu(np.linspace(0.3, 0.9, len(fi))), height=0.5)
        for i, (_, row) in enumerate(fi.iterrows()):
            ax.text(row['Imp']+.0005, i, f'{row["Imp"]:.3f}', va='center', fontsize=8, color='#555')
        ax.set_xlabel('Importance', fontsize=8, color='#999')
        ax.set_title('Random Forest Feature Importance', fontsize=10, fontweight='bold', color='#333')
        st.pyplot(fig, use_container_width=True); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PREDICT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎯  Predict":
    st.markdown("""
    <div style='padding:24px 0 8px'>
        <span style='font-family:"Bebas Neue",sans-serif;font-size:2.4rem;letter-spacing:.04em;color:#0a0a0f'>SALES PREDICTOR</span><br>
        <span style='color:#999;font-size:.9rem'>Enter item &amp; outlet parameters to predict sales</span>
    </div>""", unsafe_allow_html=True)

    item_id_max   = int(df_enc['Item_Identifier'].max())
    outlet_id_max = int(df_enc['Outlet_Identifier'].max())

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📦 Item Parameters")
        item_id  = st.slider("Item Identifier (encoded)", 0, item_id_max, 156)
        item_wt  = st.slider("Item Weight (kg)", 2.0, 22.0, 9.3, 0.1)
        item_vis = st.slider("Item Visibility", 0.000, 0.350, 0.016, 0.001, format="%.3f")
        item_mrp = st.slider("Item MRP (₹)", 10.0, 270.0, 150.0, 0.5)

    with c2:
        st.markdown("#### 🏪 Outlet Parameters")
        outlet_id   = st.slider("Outlet Identifier (encoded)", 0, outlet_id_max, 9)
        outlet_year = st.selectbox("Outlet Establishment Year",
            sorted(df['Outlet_Establishment_Year'].unique().tolist(), reverse=True))
        sel_model = st.selectbox("Model",
            list(models.keys()), index=list(models.keys()).index(best_name))
        st.markdown(f"""
        <div style='background:#f0f8f4;border-radius:10px;padding:12px 16px;
                    border:1px solid #d0e8da;margin-top:12px;font-size:.82rem;color:#444;'>
            <b style='color:#1a6b4a'>Model:</b> {sel_model}<br>
            <b style='color:#1a6b4a'>R²:</b> {models[sel_model]['r2']*100:.2f}%
            &nbsp;|&nbsp;
            <b style='color:#1a6b4a'>RMSE:</b> ₹{models[sel_model]['rmse']:,.1f}
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮  PREDICT SALES", use_container_width=True, type="primary"):
        inp = np.array([[item_id, item_wt, item_vis, item_mrp, outlet_id, int(outlet_year)]])
        inp_scaled = scaler.transform(inp)
        prediction = models[sel_model]['model'].predict(inp_scaled)[0]

        st.markdown(f"""
        <div class='pred-result'>
            <div style='font-size:.75rem;letter-spacing:.2em;color:#5050a0;
                        text-transform:uppercase;margin-bottom:8px;'>Predicted Sales</div>
            <div class='pred-amount'>₹ {prediction:,.2f}</div>
            <div class='pred-label'>Item Outlet Sales Estimate</div>
            <div class='pred-model-tag'>{sel_model} · R² {models[sel_model]['r2']*100:.2f}%</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='sec-title' style='margin-top:28px'>🤖 All Models</div>",
                    unsafe_allow_html=True)
        comp = []
        for mname, minfo in models.items():
            p = minfo['model'].predict(inp_scaled)[0]
            comp.append({'Model':mname, 'Predicted Sales':f'₹{p:,.2f}',
                         'R²':f'{minfo["r2"]*100:.4f}%', 'RMSE':f'₹{minfo["rmse"]:,.2f}'})
        st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)

        # Prediction bar chart
        preds_all = {m: models[m]['model'].predict(inp_scaled)[0] for m in models}
        fig, ax = chart(8, 3)
        bars = ax.barh(list(preds_all.keys()), list(preds_all.values()),
                       color=[models[m]['color'] for m in preds_all], height=0.4)
        for bar, val in zip(bars, preds_all.values()):
            ax.text(val+5, bar.get_y()+bar.get_height()/2,
                    f'₹{val:,.0f}', va='center', fontsize=9, color='#333')
        ax.set_xlabel('Predicted Sales (₹)', fontsize=8, color='#999')
        ax.set_title('Prediction Comparison Across Models', fontsize=10, fontweight='bold', color='#333')
        st.pyplot(fig, use_container_width=True); plt.close()