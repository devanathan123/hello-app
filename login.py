import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth
from firebase_admin import exceptions as firebase_exceptions
import requests

# Load JSON key file from GitHub
def load_json_key():
    url = "https://raw.githubusercontent.com/devanathan123/stream_app/main/private_key.json"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for HTTP errors (e.g., 404, 500)
        key_data = response.json()
        return key_data
    except Exception as e:
        st.error(f"Error fetching JSON key: {e}")
        return None

# Check if Firebase Admin SDK is already initialized
if not firebase_admin._apps:
    # Initialize Firebase Admin SDK
    key_data = load_json_key()
    cred = credentials.Certificate(key_data)
    firebase_admin.initialize_app(cred)

# def redirect():
#     # Provide a link to sign up page
#     new_page = """
#     <script>
#     window.location.href = 'https://hello-app-sign-up.streamlit.app/';  // Change the URL as needed
#     </script>
#     """
#     st.markdown(new_page, unsafe_allow_html=True)
    

# Define Streamlit app layout
def main():
    st.title("Firebase Authentication with Streamlit")

    # Display login page by default
    login()

    st.markdown(st.button("New user? [Sign up here](https://hello-app-sign-up.streamlit.app/)"))

# Login page
def login():
    st.subheader("Login")

    # Input fields for email and password
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    # Button to trigger login
    if st.button("Login"):
        try:
            # Sign in the user with email and password
            user = auth.get_user_by_email(email)
            #firebase_admin.auth.verify_password(password, user.password_hash)
            st.success("Authentication successful. User ID: {}".format(user.uid))
            st.button(st.markdown("[Welcome to e-shopping](https://hello-app-video.streamlit.app/)"))
        except firebase_exceptions.FirebaseError as e:
            st.error("Authentication failed: {}".format(e))

if __name__ == "__main__":
    main()
