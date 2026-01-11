import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import altair as alt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ShapeLens | Yasir Ahmad",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    /* Main Background and Font */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Card Style for Metrics */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1f2937;
    }
    
    /* Custom Info Box */
    .student-card {
        background-color: #eef2ff;
        border-left: 5px solid #6366f1;
        padding: 15px;
        border-radius: 5px;
        color: #312e81;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHING OPTIMIZATION ---
@st.cache_data
def process_image(_image, threshold_val, contour_color, text_color, thickness):
    # Convert PIL to Array
    img_array = np.array(_image.convert('RGB'))
    
    # Grayscale & Blur
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding
    _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_img = img_array.copy()
    data = []

    # Hex to RGB conversion for OpenCV
    c_color = tuple(int(contour_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    t_color = tuple(int(text_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Filter noise
        if area < 100:
            continue
            
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        vertices = len(approx)
        
        # Shape Logic
        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "Square" if 0.95 <= ar <= 1.05 else "Rectangle"
        elif vertices == 5:
            shape = "Pentagon"
        elif vertices > 5:
            shape = "Circle/Oval" if vertices > 8 else f"Polygon ({vertices})"
        else:
            shape = "Unknown"

        # Draw Contours & Text
        cv2.drawContours(output_img, [cnt], -1, c_color, thickness)
        
        # Calculate Moment for Center
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = approx[0][0][0], approx[0][0][1]

        cv2.putText(output_img, f"{i+1}", (cX - 10, cY + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, t_color, 2)
        
        data.append({
            "ID": i+1,
            "Shape": shape,
            "Vertices": vertices,
            "Area": round(area, 2),
            "Perimeter": round(perimeter, 2)
        })

    return output_img, thresh, data

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/10051/10051283.png", width=60)
    st.title("Settings")
    
    uploaded_file = st.file_uploader("📂 Upload Source Image", type=["jpg", "png", "jpeg"])
    
    st.markdown("### 🎛️ Parameters")
    threshold = st.slider("Threshold Sensitivity", 0, 255, 127, help="Adjust to isolate shapes from background.")
    thickness = st.slider("Contour Thickness", 1, 10, 3)
    
    col1, col2 = st.columns(2)
    with col1:
        contour_col = st.color_picker("Line Color", "#00FF00")
    with col2:
        text_col = st.color_picker("Text Color", "#FF0000")

    st.markdown("---")
    # Custom HTML Student Card
    st.markdown("""
    <div class="student-card">
        <b>👤 Submitted By:</b><br>
        <span style="font-size: 1.1em;">Yasir Ahmad</span><br>
        <span style="font-size: 0.9em; opacity: 0.8;">Reg: 22MIA1064</span><br>
        <span style="font-size: 0.9em; opacity: 0.8;">Course: CSE3089</span>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN CONTENT ---

# Hero Section
st.markdown("## 🔍 Shape & Contour Analysis")
st.markdown("**Digital Assignment 1: Computer Vision** | *Automated Geometric Detection System*")
st.divider()

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Process Image
    processed_img, thresh_img, stats = process_image(image, threshold, contour_col, text_col, thickness)
    df = pd.DataFrame(stats)

    # --- KPI METRICS ---
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Objects", len(stats))
    with m2:
        dominant_shape = df['Shape'].mode()[0] if not df.empty else "N/A"
        st.metric("Dominant Shape", dominant_shape)
    with m3:
        avg_area = f"{df['Area'].mean():.0f} px²" if not df.empty else "0"
        st.metric("Avg. Area", avg_area)
    with m4:
        st.metric("Vertices Detected", df['Vertices'].sum() if not df.empty else 0)

    st.markdown("### 🖼️ Visualization & Analysis")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["👁️ Result View", "⚙️ Binary Mask", "📊 Analytics"])
    
    with tab1:
        st.image(processed_img, caption="Annotated Output", use_container_width=True)
    
    with tab2:
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.image(image, caption="Original Input", use_container_width=True)
        with col_t2:
            st.image(thresh_img, caption="Binary Threshold (Computer View)", use_container_width=True)
            
    with tab3:
        if not df.empty:
            # Altair Chart
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('Shape', axis=alt.Axis(title='Shape Type')),
                y=alt.Y('count()', axis=alt.Axis(title='Count')),
                color=alt.Color('Shape', legend=None)
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("No shapes detected to chart.")

    # --- DATA TABLE ---
    st.markdown("### 📝 Detailed Data")
    if not df.empty:
        st.dataframe(
            df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Area": st.column_config.NumberColumn(format="%d px²"),
                "Perimeter": st.column_config.NumberColumn(format="%d px"),
            }
        )
        
        # Download Button centered
        col_dl_1, col_dl_2, col_dl_3 = st.columns([1, 2, 1])
        with col_dl_2:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Analysis Report (CSV)",
                data=csv,
                file_name="shape_analysis_yasir.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.warning("No contours found. Try adjusting the Threshold Sensitivity in the sidebar.")

else:
    # --- EMPTY STATE (On Load) ---
    st.container()
    col_empty_1, col_empty_2 = st.columns([2, 1])
    
    with col_empty_1:
        st.info("👋 Hi Yasir! The system is ready.")
        st.markdown("""
        **How to use this tool:**
        1. Upload an image containing geometric shapes from the **Sidebar**.
        2. Adjust the **Threshold** slider until shapes are clearly white against a black background.
        3. Analyze the results in the **Analytics** tab.
        """)
    with col_empty_2:
        # Placeholder SVG to make it look active
        st.markdown('<div style="text-align: center; opacity: 0.5;">📡 Waiting for input...</div>', unsafe_allow_html=True)
