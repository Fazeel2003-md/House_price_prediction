import gradio as gr
import pandas as pd
import joblib
from pathlib import Path
import os

# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"

model = joblib.load(MODEL_PATH)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_price(
    city,
    locality,
    locality_tier,
    property_type,
    bhk,
    bathrooms,
    balconies,
    built_up_area,
    carpet_area,
    floor_number,
    total_floors,
    floor_category,
    facing,
    furnishing_status,
    property_age,
    parking_spaces,
    security_score,
    gym_available,
    swimming_pool,
    power_backup,
    lift_available,
    maintenance_fee_monthly,
    distance_to_city_center_km,
    distance_to_metro_km,
    nearby_schools,
    nearby_hospitals,
    transaction_type
):

    try:

        # -------------------------------------------------
        # Convert input data into DataFrame
        # -------------------------------------------------

        input_data = {
            "city": city,
            "locality": locality,
            "locality_tier": locality_tier,
            "property_type": property_type,
            "bhk": bhk,
            "bathrooms": bathrooms,
            "balconies": balconies,
            "built_up_area": built_up_area,
            "carpet_area": carpet_area,
            "floor_number": floor_number,
            "total_floors": total_floors,
            "floor_category": floor_category,
            "facing": facing,
            "furnishing_status": furnishing_status,
            "property_age": property_age,
            "parking_spaces": parking_spaces,
            "security_score": security_score,
            "gym_available": gym_available,
            "swimming_pool": swimming_pool,
            "power_backup": power_backup,
            "lift_available": lift_available,
            "maintenance_fee_monthly": maintenance_fee_monthly,
            "distance_to_city_center_km": distance_to_city_center_km,
            "distance_to_metro_km": distance_to_metro_km,
            "nearby_schools": nearby_schools,
            "nearby_hospitals": nearby_hospitals,
            "transaction_type": transaction_type
        }

        input_df = pd.DataFrame([input_data])


        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        prediction = model.predict(input_df)[0]


        return f"₹ {float(prediction):,.2f} Lakhs"


    except Exception as e:

        return f"Prediction Error: {str(e)}"


# =========================================================
# GRADIO UI
# =========================================================

with gr.Blocks(
    title="House Price Prediction"
) as demo:


    gr.Markdown(
        """
        # 🏠 Indian House Price Prediction

        Enter the property details below to estimate
        the house price using a Machine Learning model.
        """
    )


    # =====================================================
    # SECTION 1 - LOCATION
    # =====================================================

    with gr.Group():

        gr.Markdown("## 📍 Location")

        with gr.Row():

            city = gr.Dropdown(
                choices=[
                    "Mumbai",
                    "Delhi",
                    "Bangalore",
                    "Hyderabad",
                    "Chennai",
                    "Pune",
                    "Other"
                ],
                label="City",
                value="Mumbai"
            )

            city_other = gr.Textbox(
                label="Other City",
                placeholder="If Other, type city name",
                visible=False
            )

            locality = gr.Textbox(
                label="Locality",
                placeholder="e.g. Andheri"
            )


        with gr.Row():

            locality_tier = gr.Dropdown(
                choices=[
                    "Tier 1",
                    "Tier 2",
                    "Tier 3"
                ],
                label="Locality Tier",
                value="Tier 1"
            )

            transaction_type = gr.Dropdown(
                choices=[
                    "Sale",
                    "Rent"
                ],
                label="Transaction Type",
                value="Sale"
            )


    # =====================================================
    # SECTION 2 - PROPERTY DETAILS
    # =====================================================

    with gr.Group():

        gr.Markdown("## 🏢 Property Details")

        with gr.Row():

            property_type = gr.Dropdown(
                choices=[
                    "Apartment",
                    "Villa",
                    "Independent House",
                    "Studio"
                ],
                label="Property Type",
                value="Apartment"
            )

            bhk = gr.Number(
                label="BHK",
                value=2,
                minimum=1
            )

            bathrooms = gr.Number(
                label="Bathrooms",
                value=2,
                minimum=1
            )

            balconies = gr.Number(
                label="Balconies",
                value=1,
                minimum=0
            )


        with gr.Row():

            built_up_area = gr.Number(
                label="Built-up Area (sqft)",
                value=1200,
                minimum=1
            )

            carpet_area = gr.Number(
                label="Carpet Area (sqft)",
                value=1000,
                minimum=1
            )


    # =====================================================
    # SECTION 3 - FLOOR DETAILS
    # =====================================================

    with gr.Group():

        gr.Markdown("## 🏙️ Floor & Building Details")

        with gr.Row():

            floor_number = gr.Number(
                label="Floor Number",
                value=5,
                minimum=0
            )

            total_floors = gr.Number(
                label="Total Floors",
                value=15,
                minimum=1
            )

            floor_category = gr.Dropdown(
                choices=[
                    "Low Floor",
                    "Mid Floor",
                    "High Floor"
                ],
                label="Floor Category",
                value="Mid Floor"
            )

            facing = gr.Dropdown(
                choices=[
                    "North",
                    "South",
                    "East",
                    "West"
                ],
                label="Facing",
                value="East"
            )


    # =====================================================
    # SECTION 4 - PROPERTY CONDITION
    # =====================================================

    with gr.Group():

        gr.Markdown("## 🛋️ Property Condition")

        with gr.Row():

            furnishing_status = gr.Dropdown(
                choices=[
                    "Unfurnished",
                    "Semi-Furnished",
                    "Fully Furnished"
                ],
                label="Furnishing Status",
                value="Semi-Furnished"
            )

            property_age = gr.Number(
                label="Property Age (Years)",
                value=5,
                minimum=0
            )

            parking_spaces = gr.Number(
                label="Parking Spaces",
                value=1,
                minimum=0
            )

            security_score = gr.Number(
                label="Security Score",
                value=8,
                minimum=0,
                maximum=10
            )


    # =====================================================
    # SECTION 5 - AMENITIES
    # =====================================================

    with gr.Group():

        gr.Markdown("## ⭐ Amenities")

        with gr.Row():

            gym_available = gr.Dropdown(
                choices=[
                    1,
                    0
                ],
                label="Gym Available",
                value=1
            )

            swimming_pool = gr.Dropdown(
                choices=[
                    1,
                    0
                ],
                label="Swimming Pool",
                value=1
            )

            power_backup = gr.Dropdown(
                choices=[
                    1,
                    0
                ],
                label="Power Backup",
                value=1
            )

            lift_available = gr.Dropdown(
                choices=[
                    1,
                    0
                ],
                label="Lift Available",
                value=1
            )


        maintenance_fee_monthly = gr.Number(
            label="Monthly Maintenance Fee (₹)",
            value=5000,
            minimum=0
        )


    # =====================================================
    # SECTION 6 - NEARBY FACILITIES
    # =====================================================

    with gr.Group():

        gr.Markdown("## 📍 Nearby Facilities")

        with gr.Row():

            distance_to_city_center_km = gr.Number(
                label="Distance to City Center (km)",
                value=8,
                minimum=0
            )

            distance_to_metro_km = gr.Number(
                label="Distance to Metro (km)",
                value=1.5,
                minimum=0
            )

            nearby_schools = gr.Number(
                label="Nearby Schools",
                value=5,
                minimum=0
            )

            nearby_hospitals = gr.Number(
                label="Nearby Hospitals",
                value=3,
                minimum=0
            )


    # =====================================================
    # PREDICT BUTTON
    # =====================================================

    predict_button = gr.Button(
        "🔮 Predict House Price",
        variant="primary"
    )


    # =====================================================
    # OUTPUT
    # =====================================================

    prediction_output = gr.Textbox(
        label="🏠 Predicted House Price",
        interactive=False
    )


    # =====================================================
    # CITY OTHER LOGIC
    # =====================================================

    def city_change(city_value):

        if city_value == "Other":
            return gr.update(visible=True)

        return gr.update(
            visible=False,
            value=""
        )


    city.change(
        fn=city_change,
        inputs=city,
        outputs=city_other
    )


    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    def final_prediction(
        city,
        city_other,
        locality,
        locality_tier,
        property_type,
        bhk,
        bathrooms,
        balconies,
        built_up_area,
        carpet_area,
        floor_number,
        total_floors,
        floor_category,
        facing,
        furnishing_status,
        property_age,
        parking_spaces,
        security_score,
        gym_available,
        swimming_pool,
        power_backup,
        lift_available,
        maintenance_fee_monthly,
        distance_to_city_center_km,
        distance_to_metro_km,
        nearby_schools,
        nearby_hospitals,
        transaction_type
    ):

        # If user selected Other,
        # use manually entered city

        if city == "Other":

            if not city_other.strip():

                return "Please enter the city name."

            city = city_other.strip()


        return predict_price(
            city,
            locality,
            locality_tier,
            property_type,
            bhk,
            bathrooms,
            balconies,
            built_up_area,
            carpet_area,
            floor_number,
            total_floors,
            floor_category,
            facing,
            furnishing_status,
            property_age,
            parking_spaces,
            security_score,
            gym_available,
            swimming_pool,
            power_backup,
            lift_available,
            maintenance_fee_monthly,
            distance_to_city_center_km,
            distance_to_metro_km,
            nearby_schools,
            nearby_hospitals,
            transaction_type
        )


    predict_button.click(
        fn=final_prediction,

        inputs=[
            city,
            city_other,
            locality,
            locality_tier,
            property_type,
            bhk,
            bathrooms,
            balconies,
            built_up_area,
            carpet_area,
            floor_number,
            total_floors,
            floor_category,
            facing,
            furnishing_status,
            property_age,
            parking_spaces,
            security_score,
            gym_available,
            swimming_pool,
            power_backup,
            lift_available,
            maintenance_fee_monthly,
            distance_to_city_center_km,
            distance_to_metro_km,
            nearby_schools,
            nearby_hospitals,
            transaction_type
        ],

        outputs=prediction_output
    )


# =========================================================
# LAUNCH
# =========================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
