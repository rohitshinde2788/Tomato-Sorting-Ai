import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import pandas as pd


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "models/best_model.pth"
DATA_FILE = "tomato_data.json"

IMAGE_SIZE = 224

# Actual tomato categories
CATEGORIES = [
    "Damaged",
    "Old",
    "Ripe",
    "Unripe"
]

# AI-only class
NO_TOMATO_CLASS = "No_Tomato"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Tomato Sorting AI",
    page_icon="🍅",
    layout="wide"
)


# =========================================================
# CUSTOM UI
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #777;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DATA FUNCTIONS
# =========================================================

def create_empty_data():

    return {
        "Damaged": 0,
        "Old": 0,
        "Ripe": 0,
        "Unripe": 0,
        "No_Tomato": 0
    }


def load_data():

    if not os.path.exists(DATA_FILE):

        data = create_empty_data()

        with open(DATA_FILE, "w") as file:
            json.dump(
                data,
                file,
                indent=4
            )

        return data

    try:

        with open(DATA_FILE, "r") as file:
            data = json.load(file)

        # Make sure every key exists
        for category in create_empty_data():

            if category not in data:
                data[category] = 0

        return data

    except Exception:

        return create_empty_data()


def save_data(data):

    with open(DATA_FILE, "w") as file:

        json.dump(
            data,
            file,
            indent=4
        )


def add_count(category):

    data = load_data()

    if category in CATEGORIES:

        data[category] += 1

        save_data(data)


def add_no_tomato():

    data = load_data()

    data["No_Tomato"] += 1

    save_data(data)


def clear_data():

    data = create_empty_data()

    save_data(data)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    classes = checkpoint["classes"]

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return model, classes


# Check model exists
if not os.path.exists(MODEL_PATH):

    st.error(
        f"❌ Model not found: {MODEL_PATH}"
    )

    st.stop()


# Load model
model, classes = load_model()


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# =========================================================
# SESSION STATE
# =========================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "prediction_confidence" not in st.session_state:
    st.session_state.prediction_confidence = None

if "prediction_probabilities" not in st.session_state:
    st.session_state.prediction_probabilities = None

if "last_image_id" not in st.session_state:
    st.session_state.last_image_id = None


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🍅 Tomato Sorting AI Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Tomato Classification & Sorting Analytics'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD COUNT DATA
# =========================================================

data = load_data()


# Total = only actual tomatoes
total_count = sum(
    data[category]
    for category in CATEGORIES
)


# =========================================================
# TOP METRICS
# =========================================================

col1, col2, col3, col4, col5, col6 = st.columns(6)


with col1:

    st.metric(
        "🍅 Total",
        total_count
    )


with col2:

    st.metric(
        "🔴 Ripe",
        data["Ripe"]
    )


with col3:

    st.metric(
        "🟢 Unripe",
        data["Unripe"]
    )


with col4:

    st.metric(
        "🟠 Old",
        data["Old"]
    )


with col5:

    st.metric(
        "❌ Damaged",
        data["Damaged"]
    )


with col6:

    st.metric(
        "⚪ No Tomato",
        data["No_Tomato"]
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

info1, info2, info3, info4 = st.columns(4)


with info1:

    st.metric(
        "🤖 Model",
        "ResNet18"
    )


with info2:

    st.metric(
        "🎯 Classes",
        len(classes)
    )


with info3:

    st.metric(
        "⚡ Device",
        "GPU" if torch.cuda.is_available() else "CPU"
    )


with info4:

    st.metric(
        "🧠 AI Status",
        "Ready"
    )


# =========================================================
# CAMERA SECTION
# =========================================================

st.divider()

st.header(
    "📷 Tomato Image Input"
)


camera_col, result_col = st.columns(
    [1, 1]
)


# =========================================================
# LEFT SIDE - IMAGE INPUT
# =========================================================

with camera_col:

    st.subheader(
        "📸 Capture / Upload"
    )

    # Camera
    camera_image = st.camera_input(
        "Open Camera and Capture"
    )

    st.write("OR")

    # Upload image
    uploaded_image = st.file_uploader(
        "📁 Upload Tomato Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    # -----------------------------------------------------
    # Select image
    # -----------------------------------------------------

    image_source = None

    if uploaded_image is not None:

        image_source = uploaded_image

    elif camera_image is not None:

        image_source = camera_image


    # -----------------------------------------------------
    # Display image
    # -----------------------------------------------------

    if image_source is not None:

        image = Image.open(
            image_source
        ).convert("RGB")

        st.image(
            image,
            caption="Selected Image",
            use_container_width=True
        )

        # -------------------------------------------------
        # Image ID
        # -------------------------------------------------

        image_id = str(
            getattr(
                image_source,
                "name",
                "camera_image"
            )
        ) + str(
            getattr(
                image_source,
                "size",
                ""
            )
        )


        # -------------------------------------------------
        # Reset old prediction when image changes
        # -------------------------------------------------

        if st.session_state.last_image_id != image_id:

            st.session_state.prediction_result = None
            st.session_state.prediction_confidence = None
            st.session_state.prediction_probabilities = None

            st.session_state.last_image_id = image_id


        # -------------------------------------------------
        # PREDICT BUTTON
        # -------------------------------------------------

        predict_button = st.button(
            "🤖 Predict Tomato",
            type="primary",
            use_container_width=True
        )


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        if predict_button:

            image_tensor = transform(
                image
            ).unsqueeze(0).to(device)


            with torch.no_grad():

                output = model(
                    image_tensor
                )

                probabilities = torch.softmax(
                    output,
                    dim=1
                )

                confidence, prediction = torch.max(
                    probabilities,
                    1
                )


            predicted_class = classes[
                prediction.item()
            ]


            confidence_value = (
                confidence.item() * 100
            )


            # Save prediction
            st.session_state.prediction_result = predicted_class

            st.session_state.prediction_confidence = confidence_value

            st.session_state.prediction_probabilities = (
                probabilities[0]
                .detach()
                .cpu()
                .tolist()
            )


            st.rerun()


    else:

        st.info(
            "📷 Capture an image or "
            "📁 upload a tomato image."
        )


# =========================================================
# RIGHT SIDE - AI RESULT
# =========================================================

with result_col:

    st.subheader(
        "🤖 AI Prediction"
    )


    predicted_class = (
        st.session_state.prediction_result
    )


    # -----------------------------------------------------
    # No prediction yet
    # -----------------------------------------------------

    if predicted_class is None:

        st.info(
            "Press **🤖 Predict Tomato** "
            "after selecting an image."
        )


    else:

        confidence_value = (
            st.session_state.prediction_confidence
        )


        probabilities = (
            st.session_state.prediction_probabilities
        )


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        if predicted_class == NO_TOMATO_CLASS:

            st.warning(
                "⚪ No Tomato Detected"
            )

        else:

            st.success(
                f"🍅 Prediction: {predicted_class}"
            )


        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        st.metric(
            "Confidence",
            f"{confidence_value:.2f}%"
        )


        # -------------------------------------------------
        # CONFIDENCE STATUS
        # -------------------------------------------------

        if confidence_value >= 90:

            st.success(
                "🟢 High confidence prediction"
            )

        elif confidence_value >= 75:

            st.warning(
                "🟡 Medium confidence prediction"
            )

        else:

            st.error(
                "🔴 Low confidence prediction"
            )


        # -------------------------------------------------
        # COUNT BUTTON
        # -------------------------------------------------

        st.write("### 📦 Sorting / Counting")


        if predicted_class == NO_TOMATO_CLASS:

            if st.button(
                "⚪ Record No-Tomato Event",
                type="secondary",
                use_container_width=True
            ):

                add_no_tomato()

                st.success(
                    "No-Tomato event recorded."
                )

                st.rerun()


        else:

            if st.button(
                f"➕ Add {predicted_class} to Count",
                type="primary",
                use_container_width=True
            ):

                add_count(
                    predicted_class
                )

                st.success(
                    f"{predicted_class} count updated!"
                )

                st.rerun()


        # -------------------------------------------------
        # CLASS PROBABILITIES
        # -------------------------------------------------

        st.write(
            "### 📊 Class Probabilities"
        )


        for i, class_name in enumerate(classes):

            probability = (
                probabilities[i] * 100
            )


            st.write(
                f"**{class_name}**: "
                f"{probability:.2f}%"
            )


            st.progress(
                min(
                    int(probability),
                    100
                )
            )


# =========================================================
# ANALYTICS
# =========================================================

st.divider()

st.header(
    "📊 Sorting Analytics"
)


chart_data = pd.DataFrame({

    "Category": [
        "Ripe",
        "Unripe",
        "Old",
        "Damaged"
    ],

    "Count": [
        data["Ripe"],
        data["Unripe"],
        data["Old"],
        data["Damaged"]
    ]

})
st.bar_chart(
    chart_data.set_index(
        "Category"
    )
)
# =========================================================
# DATA TABLE
# =========================================================

st.subheader(
    "📋 Current Tomato Count"
)
st.dataframe(
    chart_data,
    use_container_width=True,
    hide_index=True
)
# =========================================================
# NO TOMATO INFORMATION
# =========================================================

st.subheader(
    "⚪ Camera / No-Tomato Events"
)
st.metric(
    "No-Tomato Events",
    data["No_Tomato"]
)
# ========================================================
# DATA MANAGEMENT
# =========================================================

st.divider()

st.subheader(
    "⚙️ Data Management"
)
if st.button(
    "🗑️ Clear All Counting Data",
    use_container_width=True
):

    clear_data()

    st.session_state.prediction_result = None
    st.session_state.prediction_confidence = None
    st.session_state.prediction_probabilities = None

    st.success(
        "All counting data has been cleared."
    )

    st.rerun()
# =========================================================
# SYSTEM STATUS
# =========================================================

st.divider()

st.subheader(
    "🖥️ System Status"
)
status1, status2 = st.columns(2)

with status1:

    st.info(
        f"🍅 Total tomatoes processed: "
        f"**{total_count}**"
    )
with status2:

    st.info(
        f"⚪ No-Tomato events: "
        f"**{data['No_Tomato']}**"
    )
st.caption(
    f"AI Device: {device}"
)
st.caption(
    f"AI Classes: {', '.join(classes)}"
)