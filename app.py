import os
import re
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure key in production

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['skillsync_db']

# Define upload folder for resumes
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load or initialize internship and resume dataframes (assumed global variables)
# These are used for matching interns to internships based on skills
internship_df = pd.DataFrame()
resume_df = pd.DataFrame()
skill_to_index = {}

def preprocess_skills(skills):
    if not isinstance(skills, str):
        return []
    return [skill.strip().lower() for skill in skills.split(',') if skill.strip()]

def jaccard_similarity(vector1, vector2):
    if len(vector1) != len(vector2):
        return 0.0
    intersection = sum(a & b for a, b in zip(vector1, vector2))
    union = sum(a | b for a, b in zip(vector1, vector2))
    return intersection / union if union != 0 else 0.0

# Initialize data (simplified for this example)
def initialize_data():
    global internship_df, resume_df, skill_to_index
    # Fetch internships from MongoDB
    internships = list(db.internship_info.find())
    if internships:
        internship_df = pd.DataFrame(internships)
        # Preprocess skills for vectorization
        all_skills = set()
        for skills in internship_df['skills_required']:
            skills_list = preprocess_skills(skills)
            all_skills.update(skills_list)
        skill_to_index = {skill: idx for idx, skill in enumerate(sorted(all_skills))}
        # Create skill vectors for internships
        internship_df['Required_Skill_vector'] = internship_df['skills_required'].apply(
            lambda skills: [1 if skill in preprocess_skills(skills) else 0 for skill in skill_to_index]
        )
    else:
        internship_df = pd.DataFrame()

    # Fetch resumes from MongoDB
    resumes = list(db.resume_info.find())
    if resumes:
        resume_df = pd.DataFrame(resumes)
        resume_df['Skill_vector'] = resume_df['skills'].apply(
            lambda skills: [1 if skill in preprocess_skills(skills) else 0 for skill in skill_to_index]
        )
    else:
        resume_df = pd.DataFrame()

# Call initialization on app startup
initialize_data()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/intern_signup', methods=['GET', 'POST'])
def intern_signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        skills = request.form['skills']

        # Hash the password
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)

        # Check if email already exists
        if db.users.find_one({"email": email}):
            flash('Email already exists!', 'danger')
            return redirect(url_for('intern_signup'))

        # Get the highest user_id and increment
        max_user = db.users.find_one(sort=[("user_id", -1)])
        new_user_id = (max_user['user_id'] + 1) if max_user and 'user_id' in max_user else 1

        # Insert new user
        user = {
            "user_id": new_user_id,
            "name": name,
            "email": email,
            "password": hashed_password,
            "role": "intern",
            "skills": skills
        }
        db.users.insert_one(user)
        flash('Signup successful! Please login.', 'success')
        return redirect(url_for('intern_login', signup_success='true'))
    return render_template('intern_signup.html')

@app.route('/recruiter_signup', methods=['GET', 'POST'])
def recruiter_signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        company = request.form['company']

        # Hash the password
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)

        # Check if email already exists
        if db.users.find_one({"email": email}):
            flash('Email already exists!', 'danger')
            return redirect(url_for('recruiter_signup'))

        # Get the highest user_id and increment
        max_user = db.users.find_one(sort=[("user_id", -1)])
        new_user_id = (max_user['user_id'] + 1) if max_user and 'user_id' in max_user else 1

        # Insert new user
        user = {
            "user_id": new_user_id,
            "name": name,
            "email": email,
            "password": hashed_password,
            "role": "recruiter",
            "organization_name": company
        }
        db.users.insert_one(user)
        flash('Signup successful! Please login.', 'success')
        return redirect(url_for('recruiter_login', signup_success='true'))
    return render_template('recruiter_signup.html')

@app.route('/intern_login', methods=['GET', 'POST'])
def intern_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = db.users.find_one({"email": email, "role": "intern"})

        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['user_id'])
            session['user_name'] = user['name']
            session['role'] = user['role']
            flash('Login successful!', 'success')
            return redirect(url_for('intern_dashboard', login_success='true'))
        else:
            flash('Invalid email or password!', 'danger')
    return render_template('intern_login.html')

@app.route('/recruiter_login', methods=['GET', 'POST'])
def recruiter_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = db.users.find_one({"email": email, "role": "recruiter"})

        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['user_id'])
            session['user_name'] = user['name']
            session['role'] = user['role']
            flash('Login successful!', 'success')
            return redirect(url_for('recruiter_dashboard', login_success='true'))
        else:
            flash('Invalid email or password!', 'danger')
    return render_template('recruiter_login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_name', None)
    session.pop('role', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/intern_dashboard')
def intern_dashboard():
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern!', 'danger')
        return redirect(url_for('intern_login'))

    user_id = int(session['user_id'])  # Convert to int for consistency
    resume = db.resume_info.find_one({"user_id": user_id})

    if not resume:
        flash('Please create your resume first!', 'warning')
        return redirect(url_for('create_resume'))

    # Fetch applied internships for the user
    applications = list(db.applications.find({"user_id": user_id}))
    applied_internship_ids = [app['internship_id'] for app in applications]

    user_skills = preprocess_skills(resume['skills'])
    user_vector = [0] * len(skill_to_index)
    for skill in user_skills:
        if skill in skill_to_index:
            user_vector[skill_to_index[skill]] = 1

    internships = []
    if internship_df.empty:
        print("No internships available in internship_df.")
    else:
        for idx, internship in internship_df.iterrows():
            similarity = jaccard_similarity(user_vector, internship['Required_Skill_vector'])
            if similarity > 0:
                internships.append({
                    'id': internship.get('id', 0),
                    'role': internship.get('role', 'N/A'),
                    'company_name': internship.get('company_name', 'N/A'),
                    'description': internship.get('description_of_internship', 'N/A'),
                    'duration': internship.get('duration', 'N/A'),
                    'type_of_internship': internship.get('type_of_internship', 'N/A'),
                    'skills_required': internship.get('skills_required', 'N/A'),
                    'location': internship.get('location', 'N/A'),
                    'similarity_score': round(similarity * 100, 2)
                })

    internships = sorted(internships, key=lambda x: x['similarity_score'], reverse=True)
    login_success = request.args.get('login_success', 'false') == 'true'
    return render_template('intern_dashboard.html', internships=internships, applied_internship_ids=applied_internship_ids, login_success=login_success)

@app.route('/create_resume', methods=['GET', 'POST'])
def create_resume():
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern!', 'danger')
        return redirect(url_for('intern_login'))

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        skills = request.form['skills']
        experience = request.form['experience']
        education = request.form['education']
        certifications = request.form['certifications']
        achievements = request.form['achievements']
        user_id = int(session['user_id'])  # Convert to int

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
        return redirect(url_for('intern_login'))

    user_id = int(session['user_id'])  # Convert to int
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

@app.route('/upload_resume', methods=['GET', 'POST'])
def upload_resume():
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern!', 'danger')
        return redirect(url_for('intern_login'))

    if request.method == 'POST':
        if 'resume' not in request.files:
            flash('No file part!', 'danger')
            return redirect(url_for('upload_resume'))

        file = request.files['resume']
        if file.filename == '':
            flash('No selected file!', 'danger')
            return redirect(url_for('upload_resume'))

        if file and file.filename.endswith('.pdf'):
            user_id = int(session['user_id'])  # Convert to int
            filename = f"resume_{user_id}.pdf"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Update resume info with file path
            db.resume_info.update_one(
                {"user_id": user_id},
                {"$set": {"resume_path": file_path}},
                upsert=True
            )
            flash('Resume uploaded successfully!', 'success')
            return redirect(url_for('intern_dashboard'))
        else:
            flash('Only PDF files are allowed!', 'danger')
            return redirect(url_for('upload_resume'))

    return render_template('upload_resume.html')

@app.route('/match')
def match():
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern!', 'danger')
        return redirect(url_for('intern_login'))

    user_id = int(session['user_id'])  # Convert to int
    resume = db.resume_info.find_one({"user_id": user_id})

    if not resume:
        flash('Please create your resume first!', 'warning')
        return redirect(url_for('create_resume'))

    user_skills = preprocess_skills(resume['skills'])
    user_vector = [0] * len(skill_to_index)
    for skill in user_skills:
        if skill in skill_to_index:
            user_vector[skill_to_index[skill]] = 1

    matched_internships = []
    if not internship_df.empty:
        for idx, internship in internship_df.iterrows():
            similarity = jaccard_similarity(user_vector, internship['Required_Skill_vector'])
            if similarity > 0:
                matched_internships.append({
                    'id': internship.get('id', 0),
                    'role': internship.get('role', 'N/A'),
                    'company_name': internship.get('company_name', 'N/A'),
                    'description': internship.get('description_of_internship', 'N/A'),
                    'duration': internship.get('duration', 'N/A'),
                    'type_of_internship': internship.get('type_of_internship', 'N/A'),
                    'skills_required': internship.get('skills_required', 'N/A'),
                    'location': internship.get('location', 'N/A'),
                    'similarity_score': round(similarity * 100, 2)
                })

    matched_internships = sorted(matched_internships, key=lambda x: x['similarity_score'], reverse=True)
    return render_template('match.html', matched_internships=matched_internships)

@app.route('/download_resume')
def download_resume():
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern!', 'danger')
        return redirect(url_for('intern_login'))

    user_id = int(session['user_id'])  # Convert to int
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
        return redirect(url_for('intern_login'))

    user_id = int(session['user_id'])  # Convert to int
    # Check if the user has already applied
    existing_application = db.applications.find_one({"user_id": user_id, "internship_id": internship_id})
    if existing_application:
        flash('You have already applied to this internship!', 'warning')
        return redirect(url_for('intern_dashboard'))

    application = {
        "user_id": user_id,
        "internship_id": internship_id,
        "applied_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    db.applications.insert_one(application)
    flash('Applied successfully!', 'success')
    return redirect(url_for('intern_dashboard'))

@app.route('/applied_internships')
def applied_internships():
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern!', 'danger')
        return redirect(url_for('intern_login'))

    user_id = int(session['user_id'])  # Convert to int
    applications = list(db.applications.find({"user_id": user_id}))
    internship_ids = [app['internship_id'] for app in applications]
    internships = list(db.internship_info.find({"id": {"$in": internship_ids}}))
    return render_template('applied_internships.html', internships=internships)

@app.route('/recruiter_dashboard')
def recruiter_dashboard():
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter!', 'danger')
        return redirect(url_for('recruiter_login'))

    user_id = int(session['user_id'])  # Convert to int for consistency
    recruiter = db.users.find_one({"user_id": user_id})
    
    if not recruiter:
        flash('Recruiter profile not found. Please log in again.', 'danger')
        session.pop('user_id', None)
        session.pop('user_name', None)
        session.pop('role', None)
        return redirect(url_for('recruiter_login'))

    internships = list(db.internship_info.find({"user_id": user_id}))
    login_success = request.args.get('login_success', 'false') == 'true'
    return render_template('recruiter_dashboard.html', recruiter=recruiter, internships=internships, login_success=login_success)

@app.route('/register_internship', methods=['GET', 'POST'])
def register_internship():
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter!', 'danger')
        return redirect(url_for('recruiter_login'))

    user_id = int(session['user_id'])  # Convert to int
    # Fetch recruiter details to display in the template
    recruiter = db.users.find_one({"user_id": user_id})
    if not recruiter:
        flash('Recruiter profile not found!', 'danger')
        return redirect(url_for('recruiter_login'))

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
            "posted_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "expected_salary": ""  # Not in form, set as empty
        }
        db.internship_info.insert_one(internship)
        # Reinitialize data to include the new internship
        initialize_data()
        flash('Internship registered successfully!', 'success')
        return redirect(url_for('recruiter_dashboard'))

    return render_template('register_internship.html', recruiter=recruiter)

@app.route('/applied_applicants')
def applied_applicants():
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter!', 'danger')
        return redirect(url_for('recruiter_login'))

    user_id = int(session['user_id'])  # Convert to int
    # Fetch internships posted by this recruiter
    recruiter_internships = list(db.internship_info.find({"user_id": user_id}))
    internship_ids = [internship['id'] for internship in recruiter_internships]
    # Fetch applications for these internships
    applications = list(db.applications.find({"internship_id": {"$in": internship_ids}}))
    # Fetch intern details for each application
    applied_interns = []
    for app in applications:
        intern = db.resume_info.find_one({"user_id": app['user_id']})
        user = db.users.find_one({"user_id": app['user_id']})
        internship = db.internship_info.find_one({"id": app['internship_id']})
        if intern and user and internship:
            applied_interns.append({
                'intern_name': intern['name_of_applicant'],
                'intern_email': user['email'],
                'internship_title': internship['role'],
                'applied_at': app['applied_at']
            })
    return render_template('applied_applicants.html', applied_interns=applied_interns)

@app.route('/applied_applicants/<int:internship_id>')
def applied_applicants_specific(internship_id):
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter!', 'danger')
        return redirect(url_for('recruiter_login'))

    user_id = int(session['user_id'])  # Convert to int
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
        return redirect(url_for('recruiter_login'))

    if internship_df.empty:
        flash('No internships available to match applicants.', 'warning')
        return render_template('top_matched_applicants.html', applicants=[], internship_id=internship_id)

    if 'id' not in internship_df.columns:
        flash('Internship data is malformed. Please contact support.', 'danger')
        return render_template('top_matched_applicants.html', applicants=[], internship_id=internship_id)

    matching_internships = internship_df[internship_df['id'] == internship_id]
    if matching_internships.empty:
        flash('Internship not found.', 'danger')
        return render_template('top_matched_applicants.html', applicants=[], internship_id=internship_id)

    internship = matching_internships.iloc[0]
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
        return redirect(url_for('intern_login' if session.get('role') == 'intern' else 'recruiter_login'))

    user_id = int(session['user_id'])  # Convert to int
    user = db.users.find_one({"user_id": user_id})

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        # Check if email is already in use by another user
        existing_user = db.users.find_one({"email": email, "user_id": {"$ne": user_id}})
        if existing_user:
            flash('Email is already in use by another user!', 'error')
            return render_template('edit_profile.html', user=user)

        db.users.update_one(
            {"user_id": user_id},
            {"$set": {"name": name, "email": email}}
        )
        # Update session user_name if name changes
        session['user_name'] = name
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('intern_dashboard' if session['role'] == 'intern' else 'recruiter_dashboard'))

    return render_template('edit_profile.html', user=user)

@app.route('/edit_organization_profile', methods=['GET', 'POST'])
def edit_organization_profile():
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter!', 'danger')
        return redirect(url_for('recruiter_login'))

    user_id = int(session['user_id'])  # Convert to int
    user = db.users.find_one({"user_id": user_id})

    if not user:
        flash('Recruiter profile not found. Please log in again.', 'danger')
        session.pop('user_id', None)
        session.pop('user_name', None)
        session.pop('role', None)
        return redirect(url_for('recruiter_login'))

    if request.method == 'POST':
        organization_name = request.form['organization_name'].strip()
        contact_details = request.form['contact_details'].strip()
        location = request.form['location'].strip()
        website_link = request.form['website_link'].strip()

        # Basic validation
        if not organization_name:
            flash('Organization name is required!', 'error')
            return render_template('edit_organization_profile.html', recruiter=user)

        # Validate website link if provided
        if website_link:
            url_pattern = r'^(https?://)?([\da-z.-]+)\.([a-z.]{2,6})([/\w .-]*)*\/?$'
            if not re.match(url_pattern, website_link):
                flash('Invalid website link format!', 'error')
                return render_template('edit_organization_profile.html', recruiter=user)

        db.users.update_one(
            {"user_id": user_id},
            {"$set": {
                "organization_name": organization_name,
                "contact_details": contact_details or None,
                "location": location or None,
                "website_link": website_link or None
            }}
        )
        flash('Organization profile updated successfully!', 'success')
        return redirect(url_for('recruiter_dashboard'))

    return render_template('edit_organization_profile.html', recruiter=user)

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)