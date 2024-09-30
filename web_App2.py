from streamlit_option_menu import option_menu
from URL_Logic import predict_new_data
import streamlit as st
import pandas as pd

class URLPredictionApp:
    def __init__(self):
        self.title = "لا تكن ضحية للقرصنة"
        self.header = "مرحبا بك في موقع الحماية من الاحتيال"
        self.subheader = "لحماية خصوصيتك تحقق من المواقع قبل دخولها"

    def display_header(self):
        st.title(self.title)
        st.header(self.header)

        # عرض العنوان الفرعي في المنتصف
        st.markdown(
            f"""
            <div style='text-align: center;'>
                <h2>{self.subheader}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.text("URL Field")

    def predect(self, url=""):
        prediction, cleaned_data = predict_new_data(url)
        if prediction == 0:
            st.write(f"Prediction Results: {prediction} this URL is safe!".title(),  cleaned_data)
        elif prediction == 1:
            st.write(f"Prediction Results: {prediction} this URL is not safe!".title(), cleaned_data)
        else:
            st.write("No Class For This Inputs!", cleaned_data)

    def get_user_input(self):
        return st.text_input(" : ادخل الرابط ")

    def run_prediction_page(self):
        self.display_header()
        url_input = self.get_user_input()

        # تنسيق الزر ليكون في المنتصف مع تأثير hover
        st.markdown(
            """
            <style>
            /* تنسيق الزر */
            div.stButton > button {
                background-color: #007BFF;
                color: white;
                padding: 10px 20px;
                font-size: 16px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                margin: auto;
                display: block;
            }
            /* تأثير عند التمرير على الزر */
            div.stButton > button:hover {
                background-color: #0056b3;
                transform: scale(1.05);
            }
            </style>
            """, 
            unsafe_allow_html=True
        )

        if st.button("فحص"):
            if url_input:
                self.predect(url_input)
            else:
                input_error="يرجى ادراج الرابط"
                st.markdown(
                f"""
                    <div style='text-align: center;'>
                        <p>{input_error}</p>
                    </div>
                    """,
            unsafe_allow_html=True
        )

    def run_about_page(self):
        st.title("حول الموقع")
        st.write("""
        هذا الموقع يساعدك على التحقق من سلامة الروابط ومنع الاحتيال الإلكتروني. 
        يعمل باستخدام نماذج تعلم الآلة لتحليل الروابط وتحديد ما إذا كانت آمنة أو لا.
        """)

    def run(self):
        # Option Menu for page navigation
        page = option_menu(
            "Main Menu", ["الرئيسية", "حول الموقع"], 
            icons=["house", "info-circle"], 
            menu_icon="cast", default_index=0, orientation="horizontal"
        )
        
        if page == "الرئيسية":
            self.run_prediction_page()
        elif page == "حول الموقع":
            self.run_about_page()


# Main execution
if __name__ == "__main__":
    # Configure page layout for full screen and minimal margins
    st.set_page_config(layout="wide")  # Full screen layout
    
    # Custom CSS for RTL layout and top margin
    st.markdown(
        """
        <style>
        /* Reducing top margin and setting RTL layout */
        .block-container {
            padding-top: 2rem; /* هامش بسيط من الأعلى */
            padding-left: 1rem;
            padding-right: 1rem;
            direction: rtl; /* تحويل المحاذاة من اليمين إلى اليسار */
            text-align: right; /* محاذاة النصوص من اليمين */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    app = URLPredictionApp()
    app.run()
