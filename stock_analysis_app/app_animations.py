# this file holds the CSS Animations Class and the Error Messages Class that utilizes the created Animation Vars

# Import python packages
import streamlit as st

# ----------------------------------------------- CSS/HTML Animation Class -----------------------------------------------------------------


# Class to hold callable animation methods
class CSSAnimations:

    # Method for a spinning cogwheel animation in app
    @staticmethod
    def cog_wheel(size=30):
        """
        Returns both the CSS and HTML for the cogwheel animation.

        Parameters:
        size (int): The size of the cogwheel. Default is 30.

        Returns:
        tuple: (CSS code as string, HTML code as string)
        """
        # CSS for cog wheel animation
        cog_wheel_css = f"""
        <style>
        /* CSS for cog wheel animation */
        @keyframes rotate {{
          from {{
            transform: rotate(0deg);
          }}
          to {{
            transform: rotate(360deg);
          }}
        }}

        .cog-container {{
          display: flex;
          justify-content: center;
          align-items: center;
          height: 50%;
        }}

        .cog {{
          width: {size}px;  /* Adjustable size */
          height: {size}px; /* Adjustable size */
          border-radius: 50%;
          border: 5px solid transparent;
          border-top-color: #FF6347; /* Set to Streamlit's red color */
          animation: rotate 1s linear infinite;
        }}
        </style>
        """

        # HTML for the spinning cogwheel css to display it in the app
        cog_html = """
        <div class="cog-container">
          <div class="cog"></div>
        </div>
        """

        return cog_wheel_css + cog_html

    @staticmethod
    def warning_animation(size_factor=1.0):
        """
        Creates a minimal CSS/HTML animation of an exclamation point warning sign with Streamlit red colors.

        Parameters:
        size_factor (float): Multiplier for the size of the animation. Default is 1.0.
                             Values larger than 1.0 make the animation bigger, smaller than 1.0 make it smaller.

        Returns:
        tuple: A tuple containing (css_code, html_code) that can be used with st.markdown()
        """
        # Calculate size-adjusted values
        base_size = 120 * size_factor
        triangle_size = 100 * size_factor  # Keeping variable name but using for exclamation
        exclamation_height = 60 * size_factor
        exclamation_width = 12 * size_factor
        exclamation_dot_size = 12 * size_factor
        pulse_size = 105 * size_factor  # Smaller to prevent overlap
        font_size = 18 * size_factor

        # Define the CSS with size-adjusted values
        warning_animation_css = f"""
        <style>
            .warning-container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                margin: 20px auto;
                text-align: center;
                max-width: 100%;
            }}

            .warning-sign {{
                position: relative;
                width: {base_size}px;
                height: {base_size}px;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
            }}

            .exclamation-container {{
                position: relative;
                width: {exclamation_width * 2}px;
                height: {exclamation_height + exclamation_dot_size + 5 * size_factor}px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                z-index: 2;
            }}

            .exclamation {{
                width: {exclamation_width}px;
                height: {exclamation_height}px;
                background-color: #FF4B5C; /* Streamlit Red */
                border-radius: {4 * size_factor}px;
            }}

            .exclamation-dot {{
                width: {exclamation_dot_size}px;
                height: {exclamation_dot_size}px;
                background-color: #FF4B5C; /* Streamlit Red */
                border-radius: 50%;
                margin-top: {5 * size_factor}px;
            }}

            .pulse {{
                position: absolute;
                border: {3 * size_factor}px solid rgba(255, 75, 92, 0.5); /* Streamlit Red with opacity */
                width: {pulse_size}px;
                height: {pulse_size}px;
                border-radius: 50%;
                animation: pulse 1.5s ease-out infinite;
                opacity: 0;
                z-index: 1;
            }}

            @keyframes pulse {{
                0% {{
                    transform: scale(0.8);
                    opacity: 0.6;
                }}
                100% {{
                    transform: scale(1.5);
                    opacity: 0;
                }}
            }}

            .blink {{
                animation: blink 1s ease-in-out infinite alternate;
            }}

            @keyframes blink {{
                0% {{
                    opacity: 1;
                }}
                100% {{
                    opacity: 0.5;
                }}
            }}

            .error-message {{
                font-size: {font_size}px;
                font-weight: 600;
                line-height: 1.5;
                color: #FF4B5C; /* Streamlit Red */
                max-width: {600 * size_factor}px;
                text-align: center;
            }}
        </style>
        """

        # HTML to display the warning animation css to display it in the app
        warning_animation_html = """
        <div class="warning-container">
            <div class="warning-sign">
                <div class="pulse"></div>
                <div class="exclamation-container blink">
                    <div class="exclamation"></div>
                    <div class="exclamation-dot"></div>
                </div>
            </div>
        </div>
        """

        return warning_animation_css + warning_animation_html
