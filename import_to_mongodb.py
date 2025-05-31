import csv
from pymongo import MongoClient
import urllib.parse

# MongoDB credentials
username = "root"
password = "Chaitu895@"  # Replace with your actual password
host = "cluster0.zklixmv.mongodb.net"
database = "skillsync"

# URL-encode the username and password
encoded_username = urllib.parse.quote_plus(username)
encoded_password = urllib.parse.quote_plus(password)

# Construct the MongoDB URI
MONGO_URI = f"mongodb+srv://{encoded_username}:{encoded_password}@{host}/{database}?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client['skillsync']

# Function to import CSV into a MongoDB collection
def import_csv_to_mongo(csv_file, collection_name):
    collection = db[collection_name]
    collection.drop()  # Clear the collection before importing
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        documents = []
        for row in reader:
            # Convert numeric fields to appropriate types
            if 'user_id' in row:
                row['user_id'] = int(row['user_id'])
            if 'internship_id' in row:
                row['internship_id'] = int(row['internship_id'])
            if 'years_of_experience' in row:
                row['years_of_experience'] = int(row['years_of_experience'])
            if 'downloaded' in row:
                row['downloaded'] = int(row['downloaded'])
            documents.append(row)
        collection.insert_many(documents)
    print(f"Imported {csv_file} into {collection_name} collection.")

# Import all CSVs
import_csv_to_mongo('users.csv', 'users')
import_csv_to_mongo('internship_info.csv', 'internship_info')
import_csv_to_mongo('applications.csv', 'applications')
import_csv_to_mongo('resume_info.csv', 'resume_info')

client.close()