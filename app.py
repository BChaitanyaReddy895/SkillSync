from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import sqlite3
import os
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import re

# Configure logging to a file in a writable directory
log_dir = "/tmp/logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "app.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure NLTK data path to a writable directory
nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# SQLite Connection
DB_PATH = '/tmp/database.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    schema = '''
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        role TEXT NOT NULL,
        organization_name TEXT,
        contact_details TEXT,
        location TEXT,
        website_link TEXT,
        skills TEXT
    );
    CREATE TABLE IF NOT EXISTS resume_info (
        user_id INTEGER PRIMARY KEY,
        name_of_applicant TEXT,
        email TEXT,
        phone_number TEXT,
        skills TEXT,
        experience TEXT,
        education TEXT,
        certifications TEXT,
        achievements TEXT,
        resume_path TEXT,
        downloaded INTEGER DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    CREATE TABLE IF NOT EXISTS internship_info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL,
        description_of_internship TEXT,
        start_date TEXT,
        end_date TEXT,
        duration TEXT,
        type_of_internship TEXT,
        skills_required TEXT,
        location TEXT,
        years_of_experience INTEGER,
        phone_number TEXT,
        company_name TEXT,
        company_mail TEXT,
        user_id INTEGER,
        posted_date TEXT,
        expected_salary TEXT,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    CREATE TABLE IF NOT EXISTS applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        internship_id INTEGER,
        applied_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users(user_id),
        FOREIGN KEY (internship_id) REFERENCES internship_info(id)
    );
    '''
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(schema)
    conn.commit()
    conn.close()

# Initialize database if not exists
if not os.path.exists(DB_PATH):
    initialize_database()

# Global variables for matching
resume_df = pd.DataFrame()
internship_df = pd.DataFrame()
skill_to_index = {}

# Preprocessing function for skills
def preprocess_skills(skills):
    if not isinstance(skills, str) or skills.strip() == '':
        return []
    tokens = word_tokenize(skills.lower())
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    return tokens

# Fetch data from SQLite and initialize dataframes
def fetch_data():
    global resume_df, internship_df, skill_to_index
    try:
        conn = get_db_connection()
        resumes = conn.execute('SELECT * FROM resume_info').fetchall()
        internships = conn.execute('SELECT * FROM internship_info').fetchall()
        resume_df = pd.DataFrame([dict(row) for row in resumes])
        internship_df = pd.DataFrame([dict(row) for row in internships])
        conn.close()
        logging.info(f"Fetched {len(resume_df)} resumes from SQLite")
        logging.info(f"Fetched {len(internship_df)} internships from SQLite")
        # Preprocessing
        resume_df.fillna('', inplace=True)
        internship_df.fillna('', inplace=True)
        resume_df['processed_Skills'] = resume_df['skills'].apply(preprocess_skills) if not resume_df.empty else []
        internship_df['processed_Required_Skills'] = internship_df['skills_required'].apply(preprocess_skills) if not internship_df.empty else []
        # Create a set of unique skills, handling empty cases
        resume_skills = resume_df['processed_Skills'].explode().dropna().tolist() if not resume_df.empty else []
        internship_skills = internship_df['processed_Required_Skills'].explode().dropna().tolist() if not internship_df.empty else []
        all_skills = resume_skills + internship_skills
        logging.info(f"Total unique skills: {len(set(all_skills))}")
        skill_to_index = {skill: idx for idx, skill in enumerate(set(all_skills)) if all_skills}
        # Vectorization
        def skills_to_vector(skills):
            vector = [0] * len(skill_to_index)
            for skill in skills:
                if skill in skill_to_index:
                    vector[skill_to_index[skill]] += 1
            return vector
        resume_df['Skill_vector'] = resume_df['processed_Skills'].apply(skills_to_vector) if not resume_df.empty else []
        internship_df['Required_Skill_vector'] = internship_df['processed_Required_Skills'].apply(skills_to_vector) if not internship_df.empty else []
        return resume_df, internship_df
    except Exception as e:
        logging.error(f"Error fetching data: {str(e)}")
        raise e

# Load data after database initialization
resume_df, internship_df = fetch_data()

# Jaccard Similarity for matching (set-based, for accurate skill matching)
def jaccard_similarity(skills1, skills2):
    set1 = set(skills1)
    set2 = set(skills2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0

# Custom Jinja filter to display type
def get_type(value):
    return str(type(value).__name__)

app.jinja_env.filters['type'] = get_type
# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recruiter_login', methods=['GET', 'POST'])
def recruiter_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ? AND role = ?', (email, 'recruiter')).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['user_id']
            session['user_name'] = user['name']
            session['role'] = 'recruiter'
            flash('Login successful!', 'success')
            return redirect(url_for('recruiter_dashboard', login_success='true'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
    return render_template('recruiter_login.html')

@app.route('/intern_login', methods=['GET', 'POST'])
def intern_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ? AND role = ?', (email, 'intern')).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['user_id']
            session['user_name'] = user['name']
            session['role'] = 'intern'
            flash('Login successful!', 'success')
            return redirect(url_for('intern_dashboard', login_success='true'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
    return render_template('intern_login.html')

@app.route('/recruiter_signup', methods=['GET', 'POST'])
def recruiter_signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        company = request.form['company']
        conn = get_db_connection()
        existing = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if existing:
            conn.close()
            flash('Email already exists. Please use a different email.', 'danger')
        else:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
            max_user = conn.execute('SELECT MAX(user_id) as max_id FROM users').fetchone()
            new_user_id = (max_user['max_id'] + 1) if max_user and max_user['max_id'] else 1
            conn.execute('INSERT INTO users (user_id, name, email, password, role, organization_name) VALUES (?, ?, ?, ?, ?, ?)',
                         (new_user_id, name, email, hashed_password, 'recruiter', company))
            conn.commit()
            conn.close()
            flash('Signup successful! Please login.', 'success')
            return redirect(url_for('recruiter_login', signup_success='true'))
    return render_template('recruiter_signup.html')

@app.route('/intern_signup', methods=['GET', 'POST'])
def intern_signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        skills = request.form['skills']
        conn = get_db_connection()
        existing = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if existing:
            conn.close()
            flash('Email already exists. Please use a different email.', 'danger')
        else:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
            max_user = conn.execute('SELECT MAX(user_id) as max_id FROM users').fetchone()
            new_user_id = (max_user['max_id'] + 1) if max_user and max_user['max_id'] else 1
            conn.execute('INSERT INTO users (user_id, name, email, password, role, skills) VALUES (?, ?, ?, ?, ?, ?)',
                         (new_user_id, name, email, hashed_password, 'intern', skills))
            conn.commit()
            conn.close()
            flash('Signup successful! Please login.', 'success')
            return redirect(url_for('intern_login', signup_success='true'))
    return render_template('intern_signup.html')

@app.route('/recruiter_dashboard')
def recruiter_dashboard():
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter.', 'danger')
        return redirect(url_for('recruiter_login'))
    user_id = int(session['user_id'])
    conn = get_db_connection()
    recruiter = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
    internships = conn.execute('SELECT * FROM internship_info WHERE user_id = ?', (user_id,)).fetchall()
    conn.close()
    if not recruiter:
        flash('Recruiter profile not found. Please log in again.', 'danger')
        session.pop('user_id', None)
        session.pop('user_name', None)
        session.pop('role', None)
        return redirect(url_for('recruiter_login'))
    login_success = request.args.get('login_success', 'false') == 'true'
    return render_template('recruiter_dashboard.html', recruiter=recruiter, internships=internships, login_success=login_success)

@app.route('/intern_dashboard')
def intern_dashboard():
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern.', 'danger')
        return redirect(url_for('intern_login'))
    user_id = int(session['user_id'])
    conn = get_db_connection()
    intern = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
    resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
    applications = conn.execute('SELECT * FROM applications WHERE user_id = ?', (user_id,)).fetchall()
    conn.close()
    if not intern:
        flash('Intern profile not found. Please log in again.', 'danger')
        session.pop('user_id', None)
        session.pop('user_name', None)
        session.pop('role', None)
        return redirect(url_for('intern_login'))
    if not resume:
        flash('Please create your resume first!', 'warning')
        return redirect(url_for('create_resume'))
    applied_internship_ids = [app['internship_id'] for app in applications]
    user_skills = preprocess_skills(resume['skills'])
    internships = []
    if not internship_df.empty and 'processed_Required_Skills' in internship_df.columns:
        for idx, internship in internship_df.iterrows():
            similarity = jaccard_similarity(user_skills, internship['processed_Required_Skills'])
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
    else:
        logging.warning("internship_df is empty or missing processed_Required_Skills column")
    internships = sorted(internships, key=lambda x: x['similarity_score'], reverse=True)
    login_success = request.args.get('login_success', 'false') == 'true'
    return render_template('intern_dashboard.html', internships=internships, applied_internship_ids=applied_internship_ids, login_success=login_success)

@app.route('/register_internship', methods=['GET', 'POST'])
def register_internship():
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter.', 'danger')
        return redirect(url_for('recruiter_login'))
    user_id = int(session['user_id'])
    conn = get_db_connection()
    recruiter = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
    if not recruiter:
        conn.close()
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
        company_name = recruiter['organization_name'] if recruiter['organization_name'] else ''
        company_mail = recruiter['email'] if recruiter['email'] else ''
        if not role or not description or not start_date or not end_date or not duration or not type_of_internship or not skills_required or not location or not phone_number:
            conn.close()
            flash('All required fields must be filled!', 'danger')
            return render_template('register_internship.html', recruiter=recruiter)
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            if end <= start:
                conn.close()
                flash('End date must be after start date!', 'danger')
                return render_template('register_internship.html', recruiter=recruiter)
        except ValueError:
            conn.close()
            flash('Invalid date format!', 'danger')
            return render_template('register_internship.html', recruiter=recruiter)
        phone_pattern = r'^\+\d{1,3}-\d{3}-\d{3}-\d{4}$'
        if not re.match(phone_pattern, phone_number):
            conn.close()
            flash('Phone number must be in the format +1-800-555-1234!', 'danger')
            return render_template('register_internship.html', recruiter=recruiter)
        max_internship = conn.execute('SELECT MAX(id) as max_id FROM internship_info').fetchone()
        new_internship_id = (max_internship['max_id'] + 1) if max_internship and max_internship['max_id'] else 1
        conn.execute('''INSERT INTO internship_info (id, role, description_of_internship, start_date, end_date, duration, type_of_internship, skills_required, location, years_of_experience, phone_number, company_name, company_mail, user_id, posted_date, expected_salary) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (new_internship_id, role, description, start_date, end_date, duration, type_of_internship, skills_required, location, years_of_experience, phone_number, company_name, company_mail, user_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ""))
        conn.commit()
        conn.close()
        global resume_df, internship_df
        resume_df, internship_df = fetch_data()
        flash('Internship registered successfully!', 'success')
        return redirect(url_for('recruiter_dashboard'))
    conn.close()
    return render_template('register_internship.html', recruiter=recruiter)

@app.route('/upload_resume', methods=['GET', 'POST'])
def upload_resume():
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern.', 'danger')
        return redirect(url_for('intern_login'))
    if request.method == 'POST':
        if 'resume' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['resume']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            user_id = int(session['user_id'])
            filename = f"{user_id}_resume.pdf"
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            skills = request.form['skills']
            conn = get_db_connection()
            existing_resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
            if existing_resume:
                conn.execute('UPDATE resume_info SET resume_path = ?, skills = ? WHERE user_id = ?', (file_path, skills, user_id))
            else:
                conn.execute('INSERT INTO resume_info (user_id, resume_path, skills) VALUES (?, ?, ?)', (user_id, file_path, skills))
            conn.commit()
            conn.close()
            global resume_df, internship_df
            resume_df, internship_df = fetch_data()
            flash('Resume uploaded successfully!', 'success')
            return redirect(url_for('intern_dashboard'))
        else:
            flash('Allowed file types are pdf only', 'danger')
    return render_template('upload_resume.html')

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
        user_id = int(session['user_id'])
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
        conn = get_db_connection()
        existing_resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
        if existing_resume:
            conn.execute('''UPDATE resume_info SET name_of_applicant=?, email=?, phone_number=?, skills=?, experience=?, education=?, certifications=?, achievements=?, downloaded=? WHERE user_id=?''',
                (name, email, phone, skills, experience, education, certifications, achievements, 0, user_id))
        else:
            conn.execute('''INSERT INTO resume_info (user_id, name_of_applicant, email, phone_number, skills, experience, education, certifications, achievements, downloaded) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (user_id, name, email, phone, skills, experience, education, certifications, achievements, 0))
        conn.commit()
        conn.close()
        global resume_df, internship_df
        resume_df, internship_df = fetch_data()
        flash('Resume created successfully!', 'success')
        return redirect(url_for('intern_dashboard'))
    return render_template('create_resume.html')

@app.route('/edit_resume', methods=['GET', 'POST'])
def edit_resume():
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern!', 'danger')
        return redirect(url_for('intern_login'))
    user_id = int(session['user_id'])
    conn = get_db_connection()
    resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
    if not resume:
        conn.close()
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
        conn.execute('''UPDATE resume_info SET name_of_applicant=?, email=?, phone_number=?, skills=?, experience=?, education=?, certifications=?, achievements=? WHERE user_id=?''',
            (name, email, phone, skills, experience, education, certifications, achievements, user_id))
        conn.commit()
        conn.close()
        global resume_df, internship_df
        resume_df, internship_df = fetch_data()
        flash('Resume updated successfully!', 'success')
        return redirect(url_for('intern_dashboard'))
    conn.close()
    return render_template('edit_resume.html', resume=resume)

@app.route('/match', methods=['GET'])
def match():
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern!', 'danger')
        return redirect(url_for('intern_login'))
    user_id = int(session['user_id'])
    try:
        conn = get_db_connection()
        resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
        conn.close()
        if not resume:
            flash('Please create your resume first!', 'warning')
            return redirect(url_for('create_resume'))
        if 'skills' not in resume or not resume['skills']:
            flash('Please add skills to your resume to match internships!', 'warning')
            return redirect(url_for('edit_resume'))
        logging.info(f"Matching internships for user {user_id}, skills: {resume['skills']}")
        user_skills = preprocess_skills(resume['skills'])
        matched_internships = []
        if not internship_df.empty and 'processed_Required_Skills' in internship_df.columns:
            for idx, internship in internship_df.iterrows():
                similarity = jaccard_similarity(user_skills, internship['processed_Required_Skills'])
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
        else:
            logging.warning("internship_df is empty or missing processed_Required_Skills column")
            flash('No internships available to match at this time.', 'info')
            return render_template('match.html', matched_internships=[])
        logging.info(f"Found {len(matched_internships)} matched internships for user {user_id}")
        matched_internships = sorted(matched_internships, key=lambda x: x['similarity_score'], reverse=True)
        return render_template('match.html', matched_internships=matched_internships)
    except Exception as e:
        logging.error(f"Error in /match route for user {user_id}: {str(e)}")
        flash('An error occurred while matching internships. Please try again later.', 'danger')
        return redirect(url_for('intern_dashboard'))

@app.route('/top_matched_applicants/<int:internship_id>')
def top_matched_applicants(internship_id):
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter.', 'danger')
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
    internship_skills = internship['processed_Required_Skills']
    applicants = []
    for idx, resume in resume_df.iterrows():
        similarity = jaccard_similarity(resume['processed_Skills'], internship_skills)
        if similarity > 0:
            conn = get_db_connection()
            user = conn.execute('SELECT * FROM users WHERE user_id = ?', (resume['user_id'],)).fetchone()
            conn.close()
            if user:
                applicants.append({
                    'name': resume['name_of_applicant'],
                    'email': user['email'],
                    'similarity': round(similarity * 100, 2),
                    'resume_path': resume.get('resume_path', '')
                })
    applicants = sorted(applicants, key=lambda x: x['similarity'], reverse=True)[:5]
    return render_template('top_matched_applicants.html', applicants=applicants, internship_id=internship_id)

@app.route('/download_resume/<path:resume_path>')
def download_resume(resume_path):
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter.', 'danger')
        return redirect(url_for('recruiter_login'))
    return send_file(resume_path, as_attachment=True)

@app.route('/apply_internship/<int:internship_id>', methods=['POST'])
def apply_internship(internship_id):
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern!', 'danger')
        return redirect(url_for('intern_login'))
    user_id = int(session['user_id'])
    try:
        conn = get_db_connection()
        internship = conn.execute('SELECT * FROM internship_info WHERE id = ?', (internship_id,)).fetchone()
        if not internship:
            conn.close()
            flash('Internship not found.', 'danger')
            return redirect(url_for('intern_dashboard'))
        existing_application = conn.execute('SELECT * FROM applications WHERE user_id = ? AND internship_id = ?', (user_id, internship_id)).fetchone()
        if existing_application:
            conn.close()
            flash('You have already applied to this internship!', 'warning')
            return redirect(url_for('intern_dashboard'))
        conn.execute('INSERT INTO applications (user_id, internship_id, applied_at) VALUES (?, ?, ?)', (user_id, internship_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()
        flash('Applied successfully!', 'success')
        return redirect(url_for('intern_dashboard'))
    except Exception as e:
        logging.error(f"Error applying to internship {internship_id} for user {user_id}: {str(e)}")
        flash('An error occurred while applying. Please try again.', 'danger')
        return redirect(url_for('intern_dashboard'))

@app.route('/applied_internships')
def applied_internships():
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern!', 'danger')
        return redirect(url_for('intern_login'))
    user_id = int(session['user_id'])
    try:
        conn = get_db_connection()
        applications = conn.execute('SELECT * FROM applications WHERE user_id = ?', (user_id,)).fetchall()
        internship_ids = [app['internship_id'] for app in applications]
        if internship_ids:
            placeholders = ','.join(['?'] * len(internship_ids))
            internships = conn.execute(f'SELECT * FROM internship_info WHERE id IN ({placeholders})', internship_ids).fetchall()
        else:
            internships = []
        conn.close()
        return render_template('applied_internships.html', internships=internships)
    except Exception as e:
        logging.error(f"Error in /applied_internships for user {user_id}: {str(e)}")
        flash('An error occurred while fetching applied internships. Please try again.', 'danger')
        return redirect(url_for('intern_dashboard'))

@app.route('/applied_applicants')
def applied_applicants():
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter!', 'danger')
        return redirect(url_for('recruiter_login'))
    user_id = int(session['user_id'])
    conn = get_db_connection()
    recruiter_internships = conn.execute('SELECT * FROM internship_info WHERE user_id = ?', (user_id,)).fetchall()
    internship_ids = [internship['id'] for internship in recruiter_internships]
    if internship_ids:
        placeholders = ','.join(['?'] * len(internship_ids))
        applications = conn.execute(f'SELECT * FROM applications WHERE internship_id IN ({placeholders})', internship_ids).fetchall()
    else:
        applications = []
    applied_interns = []
    for app in applications:
        intern = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (app['user_id'],)).fetchone()
        user = conn.execute('SELECT * FROM users WHERE user_id = ?', (app['user_id'],)).fetchone()
        internship = conn.execute('SELECT * FROM internship_info WHERE id = ?', (app['internship_id'],)).fetchone()
        if intern and user and internship:
            applied_interns.append({
                'intern_name': intern['name_of_applicant'],
                'intern_email': user['email'],
                'internship_title': internship['role'],
                'applied_at': app['applied_at'],
                'resume_path': intern['resume_path'] if 'resume_path' in intern.keys() else ''
            })
    conn.close()
    return render_template('applied_applicants.html', applied_interns=applied_interns)

@app.route('/applied_applicants/<int:internship_id>')
def applied_applicants_specific(internship_id):
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter!', 'danger')
        return redirect(url_for('recruiter_login'))
    user_id = int(session['user_id'])
    conn = get_db_connection()
    applications = conn.execute('SELECT * FROM applications WHERE internship_id = ?', (internship_id,)).fetchall()
    user_ids = [app['user_id'] for app in applications]
    applicants = []
    for uid in user_ids:
        resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (uid,)).fetchone()
        user = conn.execute('SELECT * FROM users WHERE user_id = ?', (uid,)).fetchone()
        if resume and user:
            applicants.append({
                "name_of_applicant": resume['name_of_applicant'],
                "email": user['email'],
                "skills": resume['skills'],
                "experience": resume['experience'],
                "education": resume['education'],
                "resume_path": resume['resume_path'] if 'resume_path' in resume.keys() else ''
            })
    internship = conn.execute('SELECT * FROM internship_info WHERE id = ?', (internship_id,)).fetchone()
    conn.close()
    return render_template('applied_applicants.html', applicants=applicants, internship=internship)

@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'user_id' not in session:
        flash('Please login!', 'danger')
        return redirect(url_for('intern_login' if session.get('role') == 'intern' else 'recruiter_login'))
    user_id = int(session['user_id'])
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form.get('password', '')
        existing_user = conn.execute('SELECT * FROM users WHERE email = ? AND user_id != ?', (email, user_id)).fetchone()
        if existing_user:
            flash('Email is already in use by another user!', 'error')
            conn.close()
            return render_template('edit_profile.html', user=user)
        if password:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
            conn.execute('UPDATE users SET name = ?, email = ?, password = ? WHERE user_id = ?', (name, email, hashed_password, user_id))
        else:
            conn.execute('UPDATE users SET name = ?, email = ? WHERE user_id = ?', (name, email, user_id))
        conn.commit()
        conn.close()
        session['user_name'] = name
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('intern_dashboard' if session['role'] == 'intern' else 'recruiter_dashboard'))
    conn.close()
    return render_template('edit_profile.html', user=user)

@app.route('/edit_organization_profile', methods=['GET', 'POST'])
def edit_organization_profile():
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter!', 'danger')
        return redirect(url_for('recruiter_login'))
    user_id = int(session['user_id'])
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
    if not user:
        conn.close()
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
        if not organization_name:
            flash('Organization name is required!', 'error')
            conn.close()
            return render_template('edit_organization_profile.html', recruiter=user)
        if website_link:
            url_pattern = r'^(https?://)?([\da-z.-]+)\.([a-z.]{2,6})([/\w .-]*)*\/?$'
            if not re.match(url_pattern, website_link):
                flash('Invalid website link format!', 'error')
                conn.close()
                return render_template('edit_organization_profile.html', recruiter=user)
        conn.execute('UPDATE users SET organization_name = ?, contact_details = ?, location = ?, website_link = ? WHERE user_id = ?',
            (organization_name, contact_details or None, location or None, website_link or None, user_id))
        conn.commit()
        conn.close()
        flash('Organization profile updated successfully!', 'success')
        return redirect(url_for('recruiter_dashboard'))
    conn.close()
    return render_template('edit_organization_profile.html', recruiter=user)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    error_message = f"Exception: {str(e)}\n{traceback.format_exc()}"
    print(error_message)
    logging.error(error_message)
    return render_template("500.html", error_message=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)