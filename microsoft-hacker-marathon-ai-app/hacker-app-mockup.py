# Import libraries:
import streamlit as st  # Import streamlit (app Framework)
import pandas as pd  # Import Pandas for df / data wrangling
# from uszipcode import SearchEngine

# Notes:
# To keep Var data across user sessions (store in session state), use st.session_state."X" = "Var"
# Streamlit supports a user login authentication function. I haven't personally used it, but should be easy to integrate if you need to keep user info and build this out with a database/etc

# ---- Data.py ---

# # Retrieve all ZIP codes -> use this package for zip codes; I'm going to just use a dummy list
# all_zipcodes = search.by_pattern('', returns=200000, zipcode_type=None)
#
# # Extract ZIP code values
# zip_code_list = [zipcode.zipcode for zipcode in all_zipcodes]
#
# # Display the first 10 ZIP codes as a sample
# print(zip_code_list[:10])

# Dummy List of ZIP Codes
zip_codes_list = [
    "10001",  # New York, NY
    "30301",  # Atlanta, GA
    "60601",  # Chicago, IL
    "90001",  # Los Angeles, CA
    "75201",  # Dallas, TX
    "80202",  # Denver, CO
    "94102",  # San Francisco, CA
    "02108",  # Boston, MA
    "33101",  # Miami, FL
    "98101"   # Seattle, WA
]

# Dictionary mapping ZIP codes to a list of 5 restaurants
zip_to_restaurants_dict = {
    "10001": ["Joe's Pizza", "The Meatball Shop", "Shake Shack", "Eataly NYC", "Momofuku Noodle Bar"],
    "30301": ["Busy Bee Cafe", "South City Kitchen", "Fox Bros BBQ", "Mary Mac’s Tea Room", "The Varsity"],
    "60601": ["Girl & the Goat", "Portillo's", "Lou Malnati's", "Giordano’s", "The Purple Pig"],
    "90001": ["In-N-Out Burger", "Bestia", "Guelaguetza", "Howlin' Ray's", "The Boiling Crab"],
    "75201": ["Pecan Lodge", "Uchi Dallas", "Knife", "Nick & Sam’s", "Velvet Taco"],
    "80202": ["Snooze A.M. Eatery", "Root Down", "Rioja", "The Capital Grille", "D Bar Denver"],
    "94102": ["Tartine Bakery", "Zuni Café", "House of Prime Rib", "Liholiho Yacht Club", "La Taqueria"],
    "02108": ["Union Oyster House", "Mamma Maria", "Oleana", "The Capital Grille", "No. 9 Park"],
    "33101": ["Joe’s Stone Crab", "Versailles Restaurant", "Zuma", "Yardbird Southern Table", "Mandolin Aegean Bistro"],
    "98101": ["Canlis", "The Pink Door", "Tilikum Place Café", "Spinasse", "Matt’s in the Market"]
}

meal_types_list = ["Breakfast", "Lunch", "Dinner"]

# List of Meal Preferences
meal_preferences_dict = {
    "Beef": [
        "Beef Burger",
        "Steak",
        "Beef Tacos",
        "Beef Stroganoff",
        "Beef Stir-Fry"
    ],
    "Chicken": [
        "Grilled Chicken",
        "Chicken Caesar Salad",
        "Fried Chicken",
        "Chicken Alfredo",
        "Chicken Tikka Masala"
    ],
    "Seafood": [
        "Grilled Salmon",
        "Shrimp Scampi",
        "Fish Tacos",
        "Lobster Roll",
        "Crab Cakes"
    ],
    "Vegetarian": [
        "Vegetable Stir-Fry",
        "Caprese Salad",
        "Veggie Burger",
        "Margherita Pizza",
        "Stuffed Peppers"
    ],
    "Vegan": [
        "Vegan Buddha Bowl",
        "Tofu Stir-Fry",
        "Vegan Tacos",
        "Lentil Soup",
        "Vegan Pasta"
    ],
    "Salads": [
        "Greek Salad",
        "Cobb Salad",
        "Quinoa Salad",
        "Kale & Apple Salad",
        "Chickpea Salad"
    ],
    "Desserts": [
        "Cheesecake",
        "Chocolate Brownie",
        "Fruit Salad",
        "Ice Cream",
        "Tiramisu"
    ],
    "Drinks": [
        "Water",
        "Soda",
        "Iced Tea",
        "Coffee",
        "Smoothie"
    ]
}

# --- Methods.py ---
# Select Zip Code Method
def select_zip_code_section():
    return st.sidebar.selectbox("📍 Select Zip Code:", ["-- Select a Zip Code --"] + zip_codes_list)


# Select Restaurant Method
def select_restaurant_section(zip_code):
    restaurants = zip_to_restaurants_dict.get(zip_code, [])
    restaurant_options = ["-- Select a Restaurant --"] + restaurants
    return st.sidebar.selectbox("🍽️ Choose a Restaurant:", restaurant_options)


# Upload Image Method
def upload_menu_image_section(zip_code, restaurant_name):
    st.subheader(f"📍 ZIP Code: {zip_code}")
    st.markdown(f"**Selected Restaurant:** {restaurant_name}")

    uploaded_file = st.file_uploader(
        "📷 Upload a menu photo for this restaurant:",
        type=["jpg", "jpeg", "png", "pdf"]
    )

    # Show loading text briefly to mimic loading behavior
    data_load_state = st.text("Loading data...")
    data_load_state.empty()

    if uploaded_file is not None:
        st.success("✅ Menu uploaded successfully!")
        if uploaded_file.type != "application/pdf":
            st.image(uploaded_file, caption="Uploaded Menu", use_container_width=True)
        else:
            st.write("Uploaded a PDF menu.")


# Upload Menu Method
def upload_menu_image_section(zip_code, restaurant_name):
    st.subheader(f"📍 ZIP Code: {zip_code}")
    st.markdown(f"**Selected Restaurant:** {restaurant_name}")

    uploaded_file = st.file_uploader(
        "📷 Upload a menu photo for this restaurant:",
        type=["jpg", "jpeg", "png", "pdf"]
    )

    # Show loading text briefly to mimic loading behavior
    data_load_state = st.text("Loading data...")
    data_load_state.empty()

    if uploaded_file is not None:
        st.success("✅ Menu uploaded successfully!")
        if uploaded_file.type != "application/pdf":
            st.image(uploaded_file, caption="Uploaded Menu", use_container_width=True)
        else:
            st.write("Uploaded a PDF menu.")

    return uploaded_file  # Return the uploaded file object


# Select Meal Type Method
def select_meal_type():
    meal = st.radio("🍳 What meal is this menu for?", meal_types_list, index=None)  # use index param to start with no selection
    if meal:  # Only show the selection if one has been made
        st.write(f"You selected: **{meal}**")
    return meal


# User Input Budget Method
def user_input_budget():
    budget = st.slider("💵 What is your Budget Range (Adjust Slider Below)?",
                       min_value=0,
                       max_value=2000,
                       value=(500, 1000)  # Default selected range
                       )
    st.write(f"Your Budget Range: **{budget}**")
    return budget


# User Input Meal Preference Method
def select_meal_preferences():
    # Get the meal preference type
    meal_preferences = st.selectbox(
        "Select a meal preference:",
        ["-- Select a Meal Preference --"] + list(meal_preferences_dict.keys())
    )

    # Only show options and write the selection if a valid preference is selected
    if meal_preferences != "-- Select a Meal Preference --":
        meal_options = st.multiselect(
            "Choose your preference options (can choose more than one):",
            meal_preferences_dict[meal_preferences]
        )

        # Only show the selection text if options have been selected
        if meal_options:
            selected_text = st.write(f"You selected: {meal_preferences} & {', '.join(meal_options)}")
        else:
            selected_text = st.write(f"You selected: {meal_preferences}")
    else:
        # If no preference type is selected, don't show options
        meal_options = []
        selected_text = None

    return meal_preferences, meal_options, selected_text










# --- Main.py ---
# Set up App Layout
st.set_page_config(layout='wide')  # sets layout to wide w/ sidebar (check documentation for other streamlit layouts)

# Add Headers / Subheaders to Sidebar
st.sidebar.markdown(
    "<div style='text-align: center; padding: 20px;'>App Version: 1 &nbsp; <span style='color:#FF6347;'>&#x271A;</span></div>",
    unsafe_allow_html=True)  # adds App version and red medical cross icon with HTML & CSS code; nbsp adds a space
st.sidebar.header('Enter Zip Code & Restaurant')  # provide sidebar header

# Title for App
st.title("🍴 Restaurant Explorer")


# ---

zip_code = select_zip_code_section()

if zip_code != "-- Select a Zip Code --":
    restaurant = select_restaurant_section(zip_code)

    # Proceed to upload photo if restaurant selected
    if restaurant != "-- Select a Restaurant --":
        uploaded_file = upload_menu_image_section(zip_code, restaurant)

        # Proceed to meal selection only if an image was uploaded
        if uploaded_file is not None:
            meal = select_meal_type()  # Ask for the meal type after the image is uploaded
            # st.session_state.meal_type = meal  # Store the meal type in session state
            # (Use st.session_state if you want to keep track of user data across multiple app runs)

            # Proceed to Enter Budget selection only if a meal type was selected
            if meal is not None:
                budget = user_input_budget()  # Ask for the meal type after the image is uploaded
                # st.session_state.budget_range = budget  # Store the meal type in session state
                # (Use st.session_state if you want to keep track of user data across multiple app runs)

                # Proceed to meal prefs selection only if a budget range was selected
                if budget is not None:
                    meal_pref, meal_options, _ = select_meal_preferences()

                    # Check if a valid meal preference was selected (not the default)
                    if meal_pref != "-- Select a Meal Preference --":
                        # Continue with your app flow
                        pass



                else:
                    st.info("Please choose a meal preference to proceed.")
            else:
                st.info("Please choose a meal type to proceed.")
        else:
            st.info("Please upload a menu photo to proceed.")
    else:
        st.info("Please select a restaurant to proceed.")
else:
    st.info("Please select a ZIP code from the sidebar to begin.")
