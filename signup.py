import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth
from firebase_admin import exceptions as firebase_exceptions
import requests
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



# Define Streamlit app layout
def main():
    st.title("Firebase Authentication with Streamlit")

    # Display login page by default
    signup()

    # Provide a link to sign up page
    st.markdown("Already Existing User? [Login here](https://hello-app-log-in.streamlit.app/)")


# Sign-up page
def signup():
    st.subheader("Sign Up")

    # Input fields for email, password, and confirm password
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    # Button to trigger sign-up
    if st.button("Sign Up"):
        if password == confirm_password:
            try:
                # Create a new user with email and password
                user = auth.create_user(email=email, password=password)
                st.success("Sign up successful. User ID: {}".format(user.uid))
                st.markdown("Welcome to e-shoppping [JOIN US](https://hello-app-video.streamlit.app/)")
            except firebase_exceptions.FirebaseError as e:
                st.error("Sign up failed: {}".format(e))
        else:
            st.error("Passwords do not match. Please retype the passwords.")

if __name__ == "__main__":
    main()
