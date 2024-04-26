import firebase_admin
from firebase_admin import credentials, firestore, auth
from pathlib import Path
import streamlit as st

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

JSON_DIR = ROOT / 'josn_key'
MODEL_KEY = JSON_DIR / 'private_key.json'

# Initialize Firebase Admin SDK if not already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate(MODEL_KEY)
    firebase_admin.initialize_app(cred)

def Login(email,password):

    # Sign in the user with email and password
    user = auth.get_user_by_email(email)
    if user:        
        #firebase_admin.auth.verify_password(user, password)
        st.write("Authentication successful. User ID:", user.uid)
    else:
        st.write("FAILED")

def Signup(email,password,confirm):
    # Button to trigger sign-up
    
        if password == confirm:
            try:
                # Create a new user with email and password
                user = auth.create_user(email=email, password=password)
                st.success("Sign up successful. User ID: {}".format(user.uid))
            except firebase_exceptions.FirebaseError as e:
                st.error("Sign up failed: {}".format(e))
        else:
            st.error("Passwords do not match. Please retype the passwords.")

    
def main():
    st.title("Hi App!!")
    st.write("Hi there! Welcome to my Streamlit web app.")
    if st.button("Login"):
        email = st.text_input("Enter e-mail:")
        password = st.text_input("Enter password:", type="password")
        if st.button("Verify"):
            try:
                user = auth.get_user_by_email(email)
                #firebase_admin.auth.verify_password(user, password)
                st.write("Authentication successful. User ID:", user.uid)
            except:
                st.error("FAILED")
                
    if st.button("Signup"):
        email = st.text_input("Enter e-mail:")
        password = st.text_input("Enter password:", type="password")
        confirm = st.text_input("Confirm password:",type="password")
        if st.button("Create"):
            if password == confirm:
            try:
                # Create a new user with email and password
                user = auth.create_user(email=email, password=password)
                st.success("Sign up successful. User ID: {}".format(user.uid))
            except firebase_exceptions.FirebaseError as e:
                st.error("Sign up failed: {}".format(e))
        else:
            st.error("Passwords do not match. Please retype the passwords.")

            
if __name__ == "__main__":
    main()

#------------------------------------------------------------------------------
# import pyrebase
# import streamlit as st
# import firebase_admin

# FILE = Path(__file__).resolve()
# ROOT = FILE.parent
# ROOT = ROOT.relative_to(Path.cwd())

# JSON_DIR = ROOT / 'josn_key'
# MODEL_KEY = JSON_DIR / 'private_key.json'

# # firebaseConfig = {
# #   'apiKey': "AIzaSyD4BW3b9_7ZwO2pJ6Xp_imLc1YIkk-8y-U",
# #   'authDomain': "fir-ec695.firebaseapp.com",
# #   'projectId': "fir-ec695",
# #   'storageBucket': "fir-ec695.appspot.com",
# #   'messagingSenderId': "651558941637",
# #   'appId': "1:651558941637:web:be19b9b60cc38aadd111ac",
# #   'measurementId': "G-TH3ZJPWKM1"
# # }


# if not firebase_admin._apps:
#     #cred = credentials.Certificate(MODEL_KEY)
#     #firebase_admin.initialize_app(cred)
#     cred = credentials.Certificate(MODEL_KEY)
#     firebase=pyrebase.initialize_app(cred)
#     auth=firebase.auth()

# def signup():
#     st.title("Hi App!!")
#     st.write("Hi there! Welcome to my Streamlit web app.")
#     email = st.text_input("Enter e-mail:")
#     password = st.text_input("Enter password:", type="password")
#     if st.button("Verify"):
#         try:
#             user = auth.create_user_with_email_and_password(email,password)
#             st.write("User created successful. User ID:", user.uid)
#         except:
#             st.write("Failed")

# def login():
#     email = st.text_input("Enter e-mail:")
#     password = st.text_input("Enter password:", type="password")
#     if st.button("Verify"):
#         try:
#             login = auth.sign_in_with_email_and_password(email,password)
#             st.write("User created successful. User ID:", user.uid)
#         except:
#             st.write("Failed")
    
# def main():
#     st.title("Hi App!!")
#     st.write("Hi there! New User.")
#     if st.button("Signup!"):
#         signup()
#     if st.button("Login!"):
#         login()

# if __name__ == "__main__":
#     main()

