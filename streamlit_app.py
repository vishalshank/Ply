import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- THE LOGIC ---
def count_dots(image, min_area, sensitivity, crop_left, crop_right, crop_top, crop_bottom, circularity_threshold):
    img_cv = np.array(image.convert('RGB'))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    height, width, _ = img_cv.shape
    
    # 1. CROP
    x1 = int(width * (crop_left / 100))
    x2 = int(width * (1 - crop_right / 100))
    y1 = int(height * (crop_top / 100))
    y2 = int(height * (1 - crop_bottom / 100))
    
    # Safety Check
    if x1 >= x2: x2 = x1 + 10
    if y1 >= y2: y2 = y1 + 10
    
    roi = img_cv[y1:y2, x1:x2]
    
    # 2. DETECT
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_bound_sat = sensitivity 
    lower_bound_val = 50 

    # Range 1 (Darker Reds)
    lower_red1 = np.array([0, lower_bound_sat, lower_bound_val])
    upper_red1 = np.array([10, 255, 255])
    # Range 2 (Wraparound Reds)
    lower_red2 = np.array([170, lower_bound_sat, lower_bound_val])
    upper_red2 = np.array([180, 255, 255])
    
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 3. COUNT
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    output_roi = roi.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        # Filter using your calibrated values
        if area > min_area and circularity > circularity_threshold:
            count += 1
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            cv2.circle(output_roi, center, int(radius), (0, 255, 0), 2)
            cv2.putText(output_roi, str(count), (int(x)-10, int(y)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 4. DRAW VISUALIZATION
    final_display = np.zeros_like(img_cv)
    final_display[y1:y2, x1:x2] = output_roi
    # Draw the Blue Box to show what was scanned
    cv2.rectangle(final_display, (x1, y1), (x2, y2), (255, 0, 0), 4)

    return count, cv2.cvtColor(final_display, cv2.COLOR_BGR2RGB)

# --- THE UI ---
st.set_page_config(page_title="Dot One Counter", layout="wide")
st.title("🏭 Dot One: Stock Counter")

# INPUT METHOD
input_method = st.radio("Select Input:", ["📷 Use Camera", "📂 Upload Image"], horizontal=True)
image_source = None

if input_method == "📷 Use Camera":
    image_source = st.camera_input("Take a photo of the plywood stack")
else:
    image_source = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if image_source is not None:
    image = Image.open(image_source)
    
    # SETTINGS (Defaults set to your screenshots)
    with st.expander("⚙️ Calibration Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Scan Area**")
            # Defaults updated to: Left 23, Right 19, Top 0, Bottom 10
            crop_left = st.slider("Left %", 0, 49, 23)
            crop_top = st.slider("Top %", 0, 49, 0)
            crop_right = st.slider("Right %", 0, 49, 19)
            crop_bottom = st.slider("Bottom %", 0, 49, 10)
        with col2:
            st.write("**Detection**")
            # Defaults updated to: Size 51, Sensitivity 57, Roundness 0.35
            min_area = st.slider("Min Size", 10, 500, 51)
            sensitivity = st.slider("Color Sensitivity", 20, 200, 57)
            circularity = st.slider("Roundness", 0.0, 1.0, 0.35)

    # Run Detection
    count, result_img = count_dots(image, min_area, sensitivity, crop_left, crop_right, crop_top, crop_bottom, circularity)
    
    # Display Result
    st.markdown(f"""
    <div style="text-align: center; border: 2px solid #4CAF50; padding: 10px; border-radius: 10px; background-color: #f0fdf4;">
        <h2 style="color: black; margin:0;">Count: {count}</h2>
    </div>
    """, unsafe_allow_html=True)
    st.image(result_img, use_container_width=True)
