# import firebase_admin
# from firebase_admin import credentials, firestore, auth
# from pathlib import Path
# import streamlit as st

# # Get the absolute path of the current file
# FILE = Path(__file__).resolve()
# # Get the parent directory of the current file
# ROOT = FILE.parent
# # Get the relative path of the root directory with respect to the current working directory
# ROOT = ROOT.relative_to(Path.cwd())

# JSON_DIR = ROOT / 'json_key'
# MODEL_KEY = JSON_DIR / 'private_key.json'

# # Initialize Firebase Admin SDK if not already initialized
# if not firebase_admin._apps:
#     cred = credentials.Certificate(MODEL_KEY)
#     firebase_admin.initialize_app(cred)

# def authenticate_user(email, password):
#     try:
#         # Sign in the user with email and password
#         user = auth.get_user_by_email(email)
#         firebase_admin.auth.verify_password(user, password)
#         return user
#     except Exception as e:
#         # Authentication failed
#         st.write("Authentication failed:", e)
#         return None
        
# def main():
#     st.title("Hi App!!")
#     st.write("Hi there! Welcome to my Streamlit web app.")
#     email = st.text_input("Enter e-mail:")
#     password = st.text_input("Enter password:", type="password")
#     if st.button("Verify"):
#         user = authenticate_user(email, password)
#         if user:
#             st.write("Authentication successful. User ID:", user.uid)
#         else:
#             st.write("Authentication Failed")

# if __name__ == "__main__":
#     main()

import pyrebase
import streamlit as st
import firebase_admin

FILE = Path(__file__).resolve()
ROOT = FILE.parent
ROOT = ROOT.relative_to(Path.cwd())

JSON_DIR = ROOT / 'json_key'
MODEL_KEY = JSON_DIR / 'private_key.json'

if not firebase_admin._apps:
    #cred = credentials.Certificate(MODEL_KEY)
    #firebase_admin.initialize_app(cred)
    cred = credentials.Certificate(MODEL_KEY)
    firebase=pyrebase.initialize_app(cred)
    auth=firebase.auth()

def signup():
    st.title("Hi App!!")
    st.write("Hi there! Welcome to my Streamlit web app.")
    email = st.text_input("Enter e-mail:")
    password = st.text_input("Enter password:", type="password")
    if st.button("Verify"):
        try:
            user = auth.create_user_with_email_and_password(email,password)
            st.write("User created successful. User ID:", user.uid)
        except:
            st.write("Failed")

def login():
    email = st.text_input("Enter e-mail:")
    password = st.text_input("Enter password:", type="password")
    if st.button("Verify"):
        try:
            login = auth.sign_in_with_email_and_password(email,password)
            st.write("User created successful. User ID:", user.uid)
        except:
            st.write("Failed")
    
def main():
    st.title("Hi App!!")
    st.write("Hi there! New User.")
    if st.button("Signup!"):
        signup()
    if st.button("Login!"):
        login()

if __name__ == "__main__":
    main()

