from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from pymongo import MongoClient
import urllib.parse
import os

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Download NLTK dependencies
nltk.download('punkt')
nltk.download('stopwords')

# MongoDB Connection
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

# Preprocessing function for skills
def preprocess_skills(skills):
    if not isinstance(skills, str) or skills.strip() == '':
        return []
    tokens = word_tokenize(skills.lower())
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    return tokens

# Fetch data from MongoDB
def fetch_data():
    # Fetch resumes
    resumes = list(db.resume_info.find())
    resume_df = pd.DataFrame(resumes)
    
    # Fetch internships
    internships = list(db.internship_info.find())
    internship_df = pd.DataFrame(internships)

    # Debug: Check DataFrames
    print("resume_df:", resume_df)
    print("internship_df:", internship_df)

    # Preprocessing
    resume_df.fillna('', inplace=True)
    internship_df.fillna('', inplace=True)
    resume_df['processed_Skills'] = resume_df['skills'].apply(preprocess_skills)
    internship_df['processed_Required_Skills'] = internship_df['skills_required'].apply(preprocess_skills)

    # Debug: Check processed skills
    print("resume_df['processed_Skills']:", resume_df['processed_Skills'])
    print("internship_df['processed_Required_Skills']:", internship_df['processed_Required_Skills'])

    # Create a set of unique skills, handling empty cases
    resume_skills = resume_df['processed_Skills'].explode().dropna().tolist()
    internship_skills = internship_df['processed_Required_Skills'].explode().dropna().tolist()
    all_skills = resume_skills + internship_skills

    # Debug: Check all_skills
    print("all_skills:", all_skills)

    global skill_to_index
    skill_to_index = {skill: idx for idx, skill in enumerate(set(all_skills)) if all_skills}  # Avoid empty set

    # Vectorization
    def skills_to_vector(skills):
        vector = [0] * len(skill_to_index)
        for skill in skills:
            if skill in skill_to_index:
                vector[skill_to_index[skill]] += 1
        return vector

    resume_df['Skill_vector'] = resume_df['processed_Skills'].apply(skills_to_vector)
    internship_df['Required_Skill_vector'] = internship_df['processed_Required_Skills'].apply(skills_to_vector)

    return resume_df, internship_df

# Load data after database initialization
resume_df, internship_df = fetch_data()

# Jaccard Similarity for matching
def jaccard_similarity(vec1, vec2):
    set1, set2 = set(vec1), set(vec2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        organization_name = request.form.get('organization_name', '')
        contact_details = request.form.get('contact_details', '')
        location = request.form.get('location', '')
        website_link = request.form.get('website_link', '')

        # Hash the password
        from werkzeug.security import generate_password_hash
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)

        # Check if email already exists
        if db.users.find_one({"email": email}):
            flash('Email already exists!', 'danger')
            return redirect(url_for('signup'))

        # Get the highest user_id and increment
        max_user = db.users.find_one(sort=[("user_id", -1)])
        new_user_id = (max_user['user_id'] + 1) if max_user and 'user_id' in max_user else 1

        # Insert new user
        user = {
            "user_id": new_user_id,
            "name": name,
            "email": email,
            "password": hashed_password,
            "role": role,
            "organization_name": organization_name,
            "contact_details": contact_details,
            "location": location,
            "website_link": website_link
        }
        db.users.insert_one(user)
        flash('Signup successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        from werkzeug.security import check_password_hash
        user = db.users.find_one({"email": email})

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['user_id']
            session['role'] = user['role']
            flash('Login successful!', 'success')
            if user['role'] == 'intern':
                return redirect(url_for('intern_dashboard'))
            else:
                return redirect(url_for('recruiter_dashboard'))
        else:
            flash('Invalid email or password!', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('role', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/intern_dashboard')
def intern_dashboard():
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern!', 'danger')
        return redirect(url_for('login'))

    user_id = session['user_id']
    resume = db.resume_info.find_one({"user_id": user_id})

    if not resume:
        flash('Please create your resume first!', 'warning')
        return redirect(url_for('create_resume'))

    user_skills = preprocess_skills(resume['skills'])
    user_vector = [0] * len(skill_to_index)
    for skill in user_skills:
        if skill in skill_to_index:
            user_vector[skill_to_index[skill]] = 1

    internships = []
    for idx, internship in internship_df.iterrows():
        similarity = jaccard_similarity(user_vector, internship['Required_Skill_vector'])
        if similarity > 0:
            internships.append({
                'id': internship['id'],
                'role': internship['role'],
                'company_name': internship['company_name'],
                'similarity': similarity
            })

    internships = sorted(internships, key=lambda x: x['similarity'], reverse=True)
    return render_template('intern_dashboard.html', internships=internships)

@app.route('/create_resume', methods=['GET', 'POST'])
def create_resume():
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern!', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        skills = request.form['skills']
        experience = request.form['experience']
        education = request.form['education']
        certifications = request.form['certifications']
        achievements = request.form['achievements']
        user_id = session['user_id']

        resume = {
            "user_id": user_id,
            "name_of_applicant": name,
            "email": email,
            "phone_number": phone,
            "skills": skills,
            "experience": experience,
            "education": education,
            "certifications": certifications,
            "achievements": achievements,
            "downloaded": 0
        }
        db.resume_info.insert_one(resume)
        flash('Resume created successfully!', 'success')
        return redirect(url_for('intern_dashboard'))
    return render_template('create_resume.html')

@app.route('/edit_resume', methods=['GET', 'POST'])
def edit_resume():
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern!', 'danger')
        return redirect(url_for('login'))

    user_id = session['user_id']
    resume = db.resume_info.find_one({"user_id": user_id})

    if not resume:
        flash('Please create your resume first!', 'warning')
        return redirect(url_for('create_resume'))

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        skills = request.form['skills']
        experience = request.form['experience']
        education = request.form['education']
        certifications = request.form['certifications']
        achievements = request.form['achievements']

        db.resume_info.update_one(
            {"user_id": user_id},
            {"$set": {
                "name_of_applicant": name,
                "email": email,
                "phone_number": phone,
                "skills": skills,
                "experience": experience,
                "education": education,
                "certifications": certifications,
                "achievements": achievements
            }}
        )
        flash('Resume updated successfully!', 'success')
        return redirect(url_for('intern_dashboard'))

    return render_template('edit_resume.html', resume=resume)

@app.route('/download_resume')
def download_resume():
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern!', 'danger')
        return redirect(url_for('login'))

    user_id = session['user_id']
    resume = db.resume_info.find_one({"user_id": user_id})

    if not resume:
        flash('No resume found!', 'danger')
        return redirect(url_for('create_resume'))

    db.resume_info.update_one({"user_id": user_id}, {"$set": {"downloaded": 1}})

    # Render as HTML since pdfkit may not work on Hugging Face Spaces
    return render_template('resume_template.html', resume=resume)

@app.route('/apply_internship/<int:internship_id>', methods=['POST'])
def apply_internship(internship_id):
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern!', 'danger')
        return redirect(url_for('login'))

    user_id = session['user_id']
    # Check if the user has already applied
    existing_application = db.applications.find_one({"user_id": user_id, "internship_id": internship_id})
    if existing_application:
        flash('You have already applied to this internship!', 'warning')
        return redirect(url_for('intern_dashboard'))

    application = {
        "user_id": user_id,
        "internship_id": internship_id,
        "applied_at": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    db.applications.insert_one(application)
    flash('Applied successfully!', 'success')
    return redirect(url_for('intern_dashboard'))

@app.route('/applied_internships')
def applied_internships():
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern!', 'danger')
        return redirect(url_for('login'))

    user_id = session['user_id']
    applications = list(db.applications.find({"user_id": user_id}))
    internship_ids = [app['internship_id'] for app in applications]
    internships = list(db.internship_info.find({"id": {"$in": internship_ids}}))
    return render_template('applied_internships.html', internships=internships)

@app.route('/recruiter_dashboard')
def recruiter_dashboard():
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter!', 'danger')
        return redirect(url_for('login'))

    user_id = session['user_id']
    internships = list(db.internship_info.find({"user_id": user_id}))
    return render_template('recruiter_dashboard.html', internships=internships)

@app.route('/register_internship', methods=['GET', 'POST'])
def register_internship():
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter!', 'danger')
        return redirect(url_for('login'))

    user_id = session['user_id']
    # Fetch recruiter details to display in the template
    recruiter = db.users.find_one({"user_id": user_id})
    if not recruiter:
        flash('Recruiter profile not found!', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        role = request.form['role']
        description = request.form['description_of_internship']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        duration = request.form['duration']
        type_of_internship = request.form['type_of_internship']
        skills_required = request.form['skills_required']
        location = request.form['location']
        years_of_experience = int(request.form['years_of_experience'])
        phone_number = request.form['phone_number']

        # Fetch company details from the recruiter's profile
        company_name = recruiter.get('organization_name', '')
        company_mail = recruiter.get('email', '')

        # Basic validation
        if not role or not description or not start_date or not end_date or not duration or not type_of_internship or not skills_required or not location or not phone_number:
            flash('All required fields must be filled!', 'danger')
            return render_template('register_internship.html', recruiter=recruiter)

        # Validate dates
        from datetime import datetime
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            if end <= start:
                flash('End date must be after start date!', 'danger')
                return render_template('register_internship.html', recruiter=recruiter)
        except ValueError:
            flash('Invalid date format!', 'danger')
            return render_template('register_internship.html', recruiter=recruiter)

        # Validate phone number (basic regex for format like +1-800-555-1234)
        import re
        phone_pattern = r'^\+\d{1,3}-\d{3}-\d{3}-\d{4}$'
        if not re.match(phone_pattern, phone_number):
            flash('Phone number must be in the format +1-800-555-1234!', 'danger')
            return render_template('register_internship.html', recruiter=recruiter)

        # Get the highest internship_id and increment
        max_internship = db.internship_info.find_one(sort=[("id", -1)])
        new_internship_id = (max_internship['id'] + 1) if max_internship and 'id' in max_internship else 1

        # Insert internship
        internship = {
            "id": new_internship_id,
            "role": role,
            "description_of_internship": description,
            "start_date": start_date,
            "end_date": end_date,
            "duration": duration,
            "type_of_internship": type_of_internship,
            "skills_required": skills_required,
            "location": location,
            "years_of_experience": years_of_experience,
            "phone_number": phone_number,
            "company_name": company_name,
            "company_mail": company_mail,
            "user_id": user_id,
            "posted_date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            "expected_salary": ""  # Not in form, set as empty
        }
        db.internship_info.insert_one(internship)
        flash('Internship registered successfully!', 'success')
        return redirect(url_for('recruiter_dashboard'))

    return render_template('register_internship.html', recruiter=recruiter)

@app.route('/applied_applicants/<int:internship_id>')
def applied_applicants(internship_id):
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter!', 'danger')
        return redirect(url_for('login'))

    applications = list(db.applications.find({"internship_id": internship_id}))
    user_ids = [app['user_id'] for app in applications]
    applicants = []
    for user_id in user_ids:
        resume = db.resume_info.find_one({"user_id": user_id})
        user = db.users.find_one({"user_id": user_id})
        if resume and user:
            applicants.append({
                "name_of_applicant": resume['name_of_applicant'],
                "email": user['email'],
                "skills": resume['skills'],
                "experience": resume['experience'],
                "education": resume['education']
            })
    internship = db.internship_info.find_one({"id": internship_id})
    return render_template('applied_applicants.html', applicants=applicants, internship=internship)

@app.route('/top_matched_applicants/<int:internship_id>')
def top_matched_applicants(internship_id):
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter!', 'danger')
        return redirect(url_for('login'))

    internship = internship_df[internship_df['id'] == internship_id].iloc[0]
    internship_vector = internship['Required_Skill_vector']

    applicants = []
    for idx, resume in resume_df.iterrows():
        similarity = jaccard_similarity(resume['Skill_vector'], internship_vector)
        if similarity > 0:
            user = db.users.find_one({"user_id": resume['user_id']})
            if user:
                applicants.append({
                    'name': resume['name_of_applicant'],
                    'email': user['email'],
                    'similarity': similarity
                })

    applicants = sorted(applicants, key=lambda x: x['similarity'], reverse=True)[:5]
    return render_template('top_matched_applicants.html', applicants=applicants, internship_id=internship_id)

@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'user_id' not in session:
        flash('Please login!', 'danger')
        return redirect(url_for('login'))

    user_id = session['user_id']
    user = db.users.find_one({"user_id": user_id})

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        db.users.update_one(
            {"user_id": user_id},
            {"$set": {"name": name, "email": email}}
        )
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('intern_dashboard' if session['role'] == 'intern' else 'recruiter_dashboard'))

    return render_template('edit_profile.html', user=user)

@app.route('/edit_organization_profile', methods=['GET', 'POST'])
def edit_organization_profile():
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter!', 'danger')
        return redirect(url_for('login'))

    user_id = session['user_id']
    user = db.users.find_one({"user_id": user_id})

    if request.method == 'POST':
        organization_name = request.form['organization_name']
        contact_details = request.form['contact_details']
        location = request.form['location']
        website_link = request.form['website_link']
        db.users.update_one(
            {"user_id": user_id},
            {"$set": {
                "organization_name": organization_name,
                "contact_details": contact_details,
                "location": location,
                "website_link": website_link
            }}
        )
        flash('Organization profile updated successfully!', 'success')
        return redirect(url_for('recruiter_dashboard'))

    return render_template('edit_organization_profile.html', user=user)

@app.route('/results')
def results():
    return render_template('results.html')

@app.teardown_appcontext
def close_db(exception):
    client.close()

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)