import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
from recommendation import cnv, dme, drusen, normal
import tempfile

#--------------------------------
# TensorFlow Model Prediction
#--------------------------------
def model_prediction(test_image_path):
    model = tf.keras.models.load_model("Trained_Model.keras")
    img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    return np.argmax(predictions)

#--------------------------------
# Page Configuration
#--------------------------------
st.set_page_config(page_title="AI Powered Disease Prediction System", layout="wide", page_icon="🧠")

#--------------------------------
# Sidebar Navigation
#--------------------------------
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose Section",
    ["🏠 Home", "📊 Prediction"]
)

# Sidebar Footer Credits
st.sidebar.markdown("""
---
 **Developed By:**  
**Sonu Yadav**, **Vivek Singh**, **Shubhi Singh**
""")

#--------------------------------
# HOME PAGE
#--------------------------------
if app_mode == "🏠 Home":
    st.title(" AI Powered Disease Prediction System")
    st.caption("Developed by Sonu Yadav, Vivek Singh & Shubhi Singh")
    st.markdown("""
    ---
    ### **Overview**

    Welcome to the **AI Powered Disease Prediction System** — an intelligent platform designed to assist in the early detection of diseases through advanced machine learning and deep learning models.

    This system integrates two powerful modes of prediction:
    -  **Human Eye Disease Prediction** *(Image-Based using OCT Scans)*
    -  **Textual Information Based Prediction** *(Health Parameter Inputs)*

    Each component plays a vital role in improving diagnostic accuracy, supporting healthcare professionals, and providing faster, data-driven results.

    ---

    ##  Human Eye Disease Prediction

    **Optical Coherence Tomography (OCT)** is a non-invasive imaging technique used to capture detailed cross-sectional images of the retina.  
    These scans allow early detection of various retinal diseases that could lead to vision loss if untreated.

    Our model classifies retinal images into **four main categories**, helping identify potential issues efficiently and accurately.

    ###  Types of Retinal Diseases

    **1. Choroidal Neovascularization (CNV)**
    - Occurs when new, abnormal blood vessels grow under the retina.
    - Leads to leakage of blood or fluid, resulting in blurred or distorted vision.
    - Early detection through OCT scans can prevent permanent damage.

    **2. Diabetic Macular Edema (DME)**
    - A complication of diabetes that causes fluid buildup in the macula (the central part of the retina).
    - Results in swelling and vision distortion.
    - OCT imaging helps detect retinal thickening and fluid accumulation.

    **3. Drusen (Early Age-related Macular Degeneration - AMD)**
    - Characterized by yellow deposits (drusen) beneath the retina.
    - Indicates early stages of AMD, a leading cause of vision loss in older adults.
    - OCT scans reveal these deposits clearly, enabling early intervention.

    **4. Normal Retina**
    - Represents a healthy eye with a preserved foveal contour.
    - No signs of fluid, edema, or abnormal deposits.

    ---

    ###  Importance of OCT Scans
    - Provides **high-resolution retinal imaging** for early and accurate diagnosis.  
    - Enables **non-invasive monitoring** of disease progression.  
    - Supports ophthalmologists in **clinical decision-making** with visual evidence.

    ---

    ##  Textual Information Based Prediction

    In addition to image-based analysis, our platform will soon support **health data-driven predictions**, where users can input medical parameters for AI evaluation.

    ### 🔹 Diabetes Prediction — *Coming Soon *

    **Diabetes Mellitus** is a chronic condition that occurs when the body cannot properly regulate blood sugar levels.  
    Over time, high glucose levels can damage vital organs such as the eyes, kidneys, heart, and nerves.

    The upcoming **Diabetes Prediction Module** will analyze key health indicators such as:
    - Glucose level  
    - Blood pressure  
    - BMI (Body Mass Index)  
    - Age  
    - Insulin level  

    Using these inputs, the system will provide insights into the likelihood of developing diabetes — enabling early awareness and lifestyle adjustments.

    > ⚙️ *This feature is currently under development and will be available soon!*

    ---

    ### 🌟 Our Mission
    To harness the power of **Artificial Intelligence** for early disease detection, improved diagnostic accuracy, and enhanced healthcare accessibility for everyone.

    ---

    👨‍💻 **Project Developed By:**  
    **Sonu Yadav**, **Vivek Singh**, and **Shubhi Singh**

    ---
    """)

#--------------------------------
# PREDICTION SECTION
#--------------------------------
elif app_mode == "📊 Prediction":
    st.title("Disease Prediction Dashboard")
    st.caption("Developed by Sonu Yadav, Vivek Singh & Shubhi Singh")

    prediction_type = st.radio(
        "Select Prediction Mode",
        [" Image-Based Prediction", " Text-Based Prediction"]
    )

    # IMAGE-BASED PREDICTION
    if prediction_type == " Image-Based Prediction":
        st.subheader("OCT Retinal Disease Prediction")
        test_image = st.file_uploader(" Upload your OCT Image", type=["jpg", "jpeg", "png"])

        if test_image is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=test_image.name) as tmp_file:
                tmp_file.write(test_image.read())
                temp_file_path = tmp_file.name

        if st.button("🔍 Predict") and test_image is not None:
            with st.spinner("Analyzing your image... Please wait ⏳"):
                result_index = model_prediction(temp_file_path)
                class_name = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
            st.success(f"✅ Prediction Result: **{class_name[result_index]}**")

            with st.expander("📘 Learn More"):
                if result_index == 0:
                    st.write("**Detected:** CNV (Choroidal Neovascularization)")
                    st.image(test_image)
                    st.markdown(cnv)
                elif result_index == 1:
                    st.write("**Detected:** DME (Diabetic Macular Edema)")
                    st.image(test_image)
                    st.markdown(dme)
                elif result_index == 2:
                    st.write("**Detected:** Drusen (Early AMD)")
                    st.image(test_image)
                    st.markdown(drusen)
                elif result_index == 3:
                    st.write("**Detected:** Normal Retina")
                    st.image(test_image)
                    st.markdown(normal)

    # TEXT-BASED PREDICTION
    elif prediction_type == " Text-Based Prediction":
        st.subheader("Diabetes Prediction — Coming Soon 🚧")
        st.markdown("""
        We're developing a **text-based AI model** to predict diseases like **Diabetes** using clinical and lifestyle parameters.

        The upcoming feature will allow users to input data such as glucose level, BMI, and blood pressure to get real-time predictions.

        🧩 *Stay tuned — this feature is under active development!*
        """)
        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966484.png", width=200, caption="Coming Soon Feature")

    # Footer Credits
    st.markdown(
    """
    <hr style="border: 0.5px solid #D0D3D4; margin-top: 30px; margin-bottom: 10px;">
    <div style='text-align: center; font-size:20px; color: #7B7D7D;'>
        © 2025 AI Powered Disease Prediction System | Developed by <b>Sonu Yadav</b>, <b>Vivek Singh</b>, and <b>Shubhi Singh</b>
    </div>
    """,
    unsafe_allow_html=True
)