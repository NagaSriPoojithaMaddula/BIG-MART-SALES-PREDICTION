import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="BigMart Sales Prediction")

st.title("🛒 BigMart Sales Prediction")
st.write("Enter product and outlet details to predict sales")

# ---------------- INPUT FIELDS ----------------

item_weight = st.number_input(
    "Item Weight",
    min_value=0.0,
    value=None,
    placeholder="Enter item weight"
)

item_visibility = st.number_input(
    "Item Visibility",
    min_value=0.0,
    value=None,
    placeholder="Enter item visibility"
)

item_mrp = st.number_input(
    "Item MRP",
    min_value=0.0,
    value=None,
    placeholder="Enter item MRP"
)

item_fat_content = st.selectbox(
    "Item Fat Content",
    ["Low Fat", "Regular"],
    index=None,
    placeholder="Select fat content"
)

item_type = st.selectbox(
    "Item Type",
    ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables",
     "Household", "Baking Goods", "Snack Foods"],
    index=None
)

outlet_location_type = st.selectbox(
    "Outlet Location Type",
    ["Tier 1", "Tier 2", "Tier 3"],
    index=None
)

outlet_type = st.selectbox(
    "Outlet Type",
    ["Grocery Store", "Supermarket Type1",
     "Supermarket Type2", "Supermarket Type3"],
    index=None
)

# ---------------- PREDICTION ----------------

if st.button("Predict Sales"):
    if None in [
        item_weight, item_visibility, item_mrp,
        item_fat_content, item_type,
        outlet_location_type, outlet_type
    ]:
        st.warning("⚠️ Please fill all the fields")
    else:
        # 🔹 Predicted sales (replace with model.predict)
        predicted_sales = round(item_mrp * 1.8, 2)

        # 🔹 Simulated actual sales (from dataset / testing data)
        actual_sales = round(item_mrp * 1.6, 2)

        st.success(f"💰 Predicted Sales: ₹ {predicted_sales}")
        st.info(f"📌 Actual Sales: ₹ {actual_sales}")
