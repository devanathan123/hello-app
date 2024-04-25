# import firebase_admin
# from firebase_admin import credentials,firestore,auth
# import pyrebase
#
# firebaseConfig = {
#   'apiKey': "AIzaSyD4BW3b9_7ZwO2pJ6Xp_imLc1YIkk-8y-U",
#   'authDomain': "fir-ec695.firebaseapp.com",
#   'projectId': "fir-ec695",
#   'storageBucket': "fir-ec695.appspot.com",
#   'messagingSenderId': "651558941637",
#   'appId': "1:651558941637:web:be19b9b60cc38aadd111ac",
#   'measurementId': "G-TH3ZJPWKM1"
# }
#
# # cred = credentials.Certificate(r"C:\Users\HP\PycharmProjects\stream\josn_key\private_key.json")
# # firebase=firebase_admin.initialize_app(cred)
# #
# # db=firestore.client()
#
# firebase=pyrebase.initialize_app(firebaseConfig)
# auth=firebase.auth()
#
# #login Auth
# email=input("Enter your email")
# password=input("Enter your password")
# auth.sign_in_with_email_and_password(email,password)
# print("success")
########################################33

import firebase_admin
from firebase_admin import credentials,firestore,auth
from google.cloud.firestore_v1.base_query import FieldFilter,Or
from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

JSON_DIR = ROOT / 'json_key'
MODEL_KEY = JSON_DIR / 'private_key.json'

# Initialize Firebase Admin SDK
cred = credentials.Certificate(MODEL_KEY)
firebase_admin.initialize_app(cred)

def authenticate_user(email, password):
    try:
        # Sign in the user with email and password
        user = auth.get_user_by_email(email)
        auth.get_user(user.id,password=password)
        return user
    except:
        # Authentication failed
        print("Invalid user or pwd")
        return None

def get_alldoc(tb):
    docs=(
        db.collection(tb)
        .stream()
    )
    documents_list=[]
    for doc in docs:
        doc_data = doc.to_dict()
        doc_data['id'] = doc.id
        doc_data['docData'] = doc._data
        documents_list.append(doc_data)
        print(f"Name :{doc_data['id']}")
        print(f"Data :{doc_data['docData']}")
        print()

def get_product(tb,product):
    doc_ref=db.collection(tb).document(product)
    doc=doc_ref.get()
    if doc.exists:
        print(doc.to_dict())
    else:
        print("NOT FOUND")

def get_doc_detail(tb,value):
    try:
        doc_ref = db.collection(tb)
        query = doc_ref.where(filter=FieldFilter("detail","==",value))
        docs = query.stream()
        for doc in docs:
            data = doc.to_dict()
            print("Data :",data)
    except:
        print("Error")

def get_doc_detail_or(tb,value1,value2):
    try:
        doc_ref = db.collection(tb)
        query1 = FieldFilter("detail","==",value1)
        query2 = FieldFilter("detail","==",value2)

        or_filter = Or(filters=[query1,query2])
        docs = doc_ref.where(filter=or_filter).stream()

        for doc in docs:
            data = doc.to_dict()
            print("Data :",data)
    except:
        print("Error")

def update_value(tb,product,field,value):
    doc_ref = db.collection(tb)
    doc = doc_ref.document(product)
    doc.update({
        field:value
    })
#Example usage:
#login Auth
email=input("Enter your email")
password=input("Enter your password")
db=firestore.client()
user = authenticate_user(email, password)
if user:
    print(f"Authentication successful. User ID: {user.uid}")
else:
    print("Authentication failed.")

tb=input("Enter table")
get_alldoc(tb)

product=input("Enter product")

get_product(tb,product)

value=input("Enter the detail value1")
get_doc_detail(tb,value)

value1=input("Enter the detail value1")
value2=input("Enter the detail value2")
get_doc_detail_or(tb,value1,value2)

field = input("Enter the field")
new_v = input("Enter new value")
update_value(tb,product,field,new_v)
get_alldoc(tb)
