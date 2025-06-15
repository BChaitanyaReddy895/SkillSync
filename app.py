
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
import shutil
import secrets

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
# Secure secret key configuration
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))
logging.info(f"Flask secret key configured: {len(app.secret_key)} bytes")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure NLTK data path to a writable directory
nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)
logging.info(f"NLTK data directory: {nltk_data_dir}, writable: {os.access(nltk_data_dir, os.W_OK)}")

# Database configuration
SOURCE_DB = 'database.db'
DB_PATH = '/tmp/database.db'

# Copy database.db to /tmp/database.db if needed
if not os.path.exists(DB_PATH) and os.path.exists(SOURCE_DB):
    shutil.copyfile(SOURCE_DB, DB_PATH)
logging.info(f"Database path: {DB_PATH}, exists: {os.path.exists(DB_PATH)}")

# SQLite Connection
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
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(schema)
    conn.commit()
    conn.close()
    logging.info("Database schema initialized")

def insert_test_data():
    conn = get_db_connection()
    # Check if test data already exists
    existing_user = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    if existing_user > 0:
        logging.info("Test data already exists, skipping insertion")
        conn.close()
        return
    
    # Insert test data
    test_data_sql = '''
    -- Clear existing data
    DELETE FROM applications;
    DELETE FROM internship_info;
    DELETE FROM resume_info;
    DELETE FROM users;

    -- Reset autoincrement counters
    DELETE FROM sqlite_sequence WHERE name IN ('users', 'internship_info', 'applications');

    -- Insert Interns (user_id: 1-4)
    INSERT INTO users (user_id, name, email, password, role, skills) VALUES
    (1, 'Alice Smith', 'alice.smith@example.com', '$pbkdf2-sha256$29000$1RrjXGutlZLy3lvLeW/t3Q$J3Q2X5n7Z9X8Y6W5V4T3R2Q1P0O9N8M7L6K5J4I3H2G1F0E', 'intern', 'python, java, sql, tensorflow'),
    (2, 'Bob Johnson', 'bob.johnson@example.com', '$pbkdf2-sha256$29000$1RrjXGutlZLy3lvLeW/t3Q$J3Q2X5n7Z9X8Y6W5V4T3R2Q1P0O9N8M7L6K5J4I3H2G1F0E', 'intern', 'javascript, react, node.js'),
    (3, 'Carol Lee', 'carol.lee@example.com', '$pbkdf2-sha256$29000$1RrjXGutlZLy3lvLeW/t3Q$J3Q2X5n7Z9X8Y6W5V4T3R2Q1P0O9N8M7L6K5J4I3H2G1F0E', 'intern', 'python, django, postgresql'),
    (4, 'David Brown', 'david.brown@example.com', '$pbkdf2-sha256$29000$1RrjXGutlZLy3lvLeW/t3Q$J3Q2X5n7Z9X8Y6W5V4T3R2Q1P0O9N8M7L6K5J4I3H2G1F0E', 'intern', 'c++, opencv, machine learning');

    -- Insert Recruiters (user_id: 5-7)
    INSERT INTO users (user_id, name, email, password, role, organization_name, contact_details, location, website_link) VALUES
    (5, 'Emma Wilson', 'emma.wilson@techcorp.com', '$pbkdf2-sha256$29000$1RrjXGutlZLy3lvLeW/t3Q$J3Q2X5n7Z9X8Y6W5V4T3R2Q1P0O9N8M7L6K5J4I3H2G1F0E', 'recruiter', 'TechCorp', '+1-800-555-1234', 'San Francisco, CA', 'https://techcorp.com'),
    (6, 'Frank Taylor', 'frank.taylor@innovatech.com', '$pbkdf2-sha256$29000$1RrjXGutlZLy3lvLeW/t3Q$J3Q2X5n7Z9X8Y6W5V4T3R2Q1P0O9N8M7L6K5J4I3H2G1F0E', 'recruiter', 'Innovatech', '+1-800-555-5678', 'New York, NY', 'https://innovatech.com'),
    (7, 'Grace Miller', 'grace.miller@datatech.com', '$pbkdf2-sha256$29000$1RrjXGutlZLy3lvLeW/t3Q$J3Q2X5n7Z9X8Y6W5V4T3R2Q1P0O9N8M7L6K5J4I3H2G1F0E', 'recruiter', 'DataTech', '+1-800-555-9012', 'Austin, TX', 'https://datatech.com');

    -- Insert Resumes
    INSERT INTO resume_info (user_id, name_of_applicant, email, phone_number, skills, experience, education, certifications, achievements, resume_path, downloaded) VALUES
    (1, 'Alice Smith', 'alice.smith@example.com', '+1-555-123-4567', 'python, java, sql, tensorflow', 'Software Intern at XYZ Corp (6 months)', 'B.S. Computer Science, MIT, 2024', 'AWS Certified Developer', 'Won Hackathon 2023', 'static/uploads/1_resume.pdf', 0),
    (2, 'Bob Johnson', 'bob.johnson@example.com', '+1-555-234-5678', 'javascript, react, node.js', 'Frontend Developer at ABC Inc (1 year)', 'B.S. Software Engineering, Stanford, 2023', 'React Professional Certification', 'Published NPM package', 'static/uploads/2_resume.pdf', 0),
    (3, 'Carol Lee', 'carol.lee@example.com', '+1-555-345-6789', 'python, django, postgresql', 'Backend Intern at DEF Ltd (4 months)', 'B.S. Computer Science, UC Berkeley, 2024', 'Django Developer Certification', 'Top 10 in CodeJam 2024', 'static/uploads/3_resume.pdf', 0),
    (4, 'David Brown', 'david.brown@example.com', '+1-555-456-7890', 'c++, opencv, machine learning', 'Research Assistant at GHI University (8 months)', 'M.S. AI, Caltech, 2025', 'TensorFlow Developer Certificate', 'Published CVPR 2024 paper', 'static/uploads/4_resume.pdf', 0);

    -- Insert Internships
    INSERT INTO internship_info (id, role, description_of_internship, start_date, end_date, duration, type_of_internship, skills_required, location, years_of_experience, phone_number, company_name, company_mail, user_id, posted_date, expected_salary) VALUES
    (1, 'Machine Learning Intern', 'Develop ML models for image recognition', '2025-07-01', '2025-12-31', '6 months', 'Full-time', 'python, tensorflow, machine learning', 'San Francisco, CA', 0, '+1-800-555-1234', 'TechCorp', 'emma.wilson@techcorp.com', 5, '2025-06-15', 'Unpaid'),
    (2, 'Frontend Developer Intern', 'Build responsive web interfaces', '2025-08-01', '2025-11-30', '4 months', 'Part-time', 'javascript, react, css', 'New York, NY', 0, '+1-800-555-5678', 'Innovatech', 'frank.taylor@innovatech.com', 6, '2025-06-15', '$15/hr'),
    (3, 'Backend Developer Intern', 'Develop APIs and database systems', '2025-07-15', '2026-01-15', '6 months', 'Full-time', 'python, django, sql', 'Austin, TX', 1, '+1-800-555-9012', 'DataTech', 'grace.miller@datatech.com', 7, '2025-06-15', 'Unpaid'),
    (4, 'AI Research Intern', 'Research on computer vision algorithms', '2025-09-01', '2026-02-28', '6 months', 'Full-time', 'c++, opencv, python', 'San Francisco, CA', 1, '+1-800-555-1234', 'TechCorp', 'emma.wilson@techcorp.com', 5, '2025-06-15', '$20/hr'),
    (5, 'Full Stack Intern', 'Work on both frontend and backend', '2025-07-01', '2025-12-31', '6 months', 'Full-time', 'javascript, node.js, postgresql', 'New York, NY', 0, '+1-800-555-5678', 'Innovatech', 'frank.taylor@innovatech.com', 6, '2025-06-15', 'Unpaid'),
    (6, 'Data Science Intern', 'Analyze data and build predictive models', '2025-08-01', '2025-12-31', '5 months', 'Part-time', 'python, sql, machine learning', 'Austin, TX', 0, '+1-800-555-9012', 'DataTech', 'grace.miller@datatech.com', 7, '2025-06-15', '$18/hr');

    -- Insert Applications
    INSERT INTO applications (id, user_id, internship_id, applied_at) VALUES
    (1, 1, 1, '2025-06-15 10:00:00'),
    (2, 1, 3, '2025-06-15 11:00:00'),
    (3, 1, 4, '2025-06-15 12:00:00'),
    (4, 2, 2, '2025-06-15 13:00:00'),
    (5, 2, 5, '2025-06-15 14:00:00'),
    (6, 3, 3, '2025-06-15 15:00:00'),
    (7, 3, 6, '2025-06-15 16:00:00'),
    (8, 4, 1, '2025-06-15 17:00:00'),
    (9, 4, 4, '2025-06-15 18:00:00');
    '''
    conn.executescript(test_data_sql)
    conn.commit()
    logging.info("Inserted comprehensive test data")
    conn.close()

# Initialize database and ensure test data
if not os.path.exists(DB_PATH):
    initialize_database()
    insert_test_data()
else:
    # Verify test data exists
    conn = get_db_connection()
    user_count = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    conn.close()
    if user_count == 0:
        logging.warning("Test data missing, reinitializing")
        initialize_database()
        insert_test_data()

# Global variables for matching
resume_df = pd.DataFrame()
internship_df = pd.DataFrame()
skill_to_index = {}

# Preprocessing function for skills
def preprocess_skills(skills):
    logging.info(f"Raw skills input: {skills}")
    if not isinstance(skills, str) or skills.strip() == '':
        logging.warning(f"Skills empty or invalid: {skills}")
        return []
    # Replace multiple separators with comma
    skills_clean = re.sub(r'[;|\\/]', ',', skills)
    tokens = [s.strip().lower() for s in skills_clean.split(',') if s.strip()]
    stop_words = set(stopwords.words('english'))
    processed = []
    for token in tokens:
        token = ''.join([c for c in token if c not in string.punctuation])
        if token and token not in stop_words:
            processed.append(token)
    logging.info(f"Processed skills: {processed}")
    return processed

# Fetch data from SQLite and initialize dataframes
def fetch_data():
    global resume_df, internship_df, skill_to_index
    try:
        conn = get_db_connection()
        resumes = conn.execute('SELECT * FROM resume_info').fetchall()
        internships = conn.execute('SELECT * FROM internship_info').fetchall()
        logging.info(f"Raw resumes: {len(resumes)}, Raw internships: {len(internships)}")
        resume_df = pd.DataFrame([dict(row) for row in resumes])
        internship_df = pd.DataFrame([dict(row) for row in internships])
        logging.info(f"resume_df shape: {resume_df.shape}, internship_df shape: {internship_df.shape}")
        conn.close()
        resume_df.fillna('', inplace=True)
        internship_df.fillna('', inplace=True)
        resume_df['processed_Skills'] = resume_df['skills'].apply(preprocess_skills) if not resume_df.empty else []
        internship_df['processed_Required_Skills'] = internship_df['skills_required'].apply(preprocess_skills) if not internship_df.empty else []
        resume_skills = resume_df['processed_Skills'].explode().dropna().tolist() if not resume_df.empty else []
        internship_skills = internship_df['processed_Required_Skills'].explode().dropna().tolist() if not internship_df.empty else []
        all_skills = resume_skills + internship_skills
        logging.info(f"Total unique skills: {len(set(all_skills))}")
        skill_to_index = {skill: idx for idx, skill in enumerate(set(all_skills)) if all_skills}
        def skills_to_vector(skills):
            vector = [0] * len(skill_to_index)
            for skill in skills:
                if skill in skill_to_index:
                    vector[skill_to_index[skill]] += 1
            return vector
        resume_df['Skill_vector'] = resume_df['processed_Skills'].apply(skills_to_vector) if not resume_df.empty else []
        internship_df['Required_Skill_vector'] = internship_df['processed_Required_Skills'].apply(skills_to_vector) if not internship_df.empty else []
        logging.info(f"fetch_data completed successfully")
        return resume_df, internship_df
    except Exception as e:
        logging.error(f"Error fetching data: {str(e)}")
        raise e

# Load data after database initialization
resume_df, internship_df = fetch_data()

# Jaccard Similarity for matching
def jaccard_similarity(skills1, skills2):
    set1 = set(skills1)
    set2 = set(skills2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    similarity = len(intersection) / len(union) if union else 0
    logging.info(f"Skills1: {set1}, Skills2: {set2}, Intersection: {intersection}, Union: {union}, Similarity: {similarity}")
    return similarity

# Custom Jinja filter to display type
def get_type(value):
    return str(type(value).__name__)

app.jinja_env.filters['type'] = get_type

# Routes
@app.route('/', strict_slashes=False)
def index():
    return render_template('index.html')

@app.route('/recruiter_login', methods=['GET', 'POST'], strict_slashes=False)
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
            session['email'] = user['email']
            session['role'] = 'recruiter'
            logging.info(f"Recruiter login successful: Email={email}, user_id={user['user_id']}")
            flash('Login successful!', 'success')
            return redirect(url_for('recruiter_dashboard', login_success=True))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
    return render_template('recruiter_login.html')

@app.route('/intern_login', methods=['GET', 'POST'], strict_slashes=False)
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
            logging.info(f"Intern login successful: Email={email}, user_id={user['user_id']}")
            flash('Login successful!', 'success')
            return redirect(url_for('intern_dashboard', login_success=True))
        else:
            flash('Invalid credentials.', 'danger')
    return render_template('intern_login.html')

@app.route('/recruiter_signup', methods=['GET', 'POST'], strict_slashes=False)
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
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            max_user = conn.execute('SELECT MAX(user_id) as max_id FROM users').fetchone()
            new_user_id = (max_user['max_id'] + 1) if max_user and max_user['max_id'] else 1
            conn.execute('INSERT INTO users (user_id, name, email, password, role, organization_name) VALUES (?, ?, ?, ?, ?, ?)', 
                        (new_user_id, name, email, hashed_password, 'recruiter', company))
            conn.commit()
            conn.close()
            flash('Signup successful! Please login.', 'success')
            return redirect(url_for('recruiter_login', signup_success=True))
    return render_template('recruiter_signup.html')

@app.route('/intern_signup', methods=['GET', 'POST'], strict_slashes=False)
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
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            max_user = conn.execute('SELECT MAX(user_id) as max_id FROM users').fetchone()
            new_user_id = (max_user['max_id'] + 1) if max_user and max_user['max_id'] else 1
            conn.execute('INSERT INTO users (user_id, name, email, password, role, skills) VALUES (?, ?, ?, ?, ?, ?)', 
                         (new_user_id, name, email, hashed_password, 'intern', skills))
            conn.commit()
            conn.close()
            flash('Signup successful! Please login.', 'success')
            return redirect(url_for('intern_login', signup_success=True))
    return render_template('intern_signup.html')

@app.route('/recruiter_dashboard', strict_slashes=False)
def recruiter_dashboard():
    logging.info(f"Recruiter dashboard access, session data: {dict(session)}")
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter.', 'danger')
        logging.info("Redirecting to recruiter_login due to missing session or invalid role")
        return redirect(url_for('index'))

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
    login_success = request.args.get('login_success', False) == True
    return render_template('recruiter_dashboard.html', recruiter=recruiter, internships=internships, login_success=login_success, user_name=session.get('user_name'))

@app.route('/intern_dashboard', strict_slashes=False)
def intern_dashboard():
    logging.info(f"Intern dashboard access, session data: {dict(session)}")
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern.', 'danger')
        return redirect(url_for('intern_login'))
    user_id = int(session['user_id'])
    try:
        global resume_df, internship_df
        resume_df, internship_df = fetch_data()
        logging.info(f"[DASHBOARD DEBUG] Refreshed data: resume_df shape: {resume_df.shape}, internship_df shape: {internship_df.shape}")
        
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
        logging.info(f"[DASHBOARD DEBUG] User {user_id} processed skills: {user_skills}")
        internships = []
        if not internship_df.empty and 'processed_Required_Skills' in internship_df.columns:
            logging.info(f"[DASHBOARD DEBUG] internship_df has {len(internship_df)} internships")
            for idx, internship in internship_df.iterrows():
                similarity = jaccard_similarity(user_skills, internship['processed_Required_Skills'])
                logging.info(f"[DASHBOARD DEBUG] Internship {internship.get('id', 0)} similarity: {similarity}")
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
            logging.warning("[DASHBOARD DEBUG] internship_df is empty or missing processed_Required_Skills column")
        internships = sorted(internships, key=lambda x: x['similarity_score'], reverse=True)
        logging.info(f"[DASHBOARD DEBUG] Found {len(internships)} matched internships for user {user_id}")
        login_success = request.args.get('login_success', False) == True
        return render_template('intern_dashboard.html', user_name=session.get('user_name'), internships=internships, applied_internship_ids=applied_internship_ids, login_success=login_success)
    except Exception as e:
        logging.error(f"Error in /intern_dashboard for user {user_id}: {str(e)}")
        flash('An error occurred while loading the dashboard. Please try again.', 'danger')
        return redirect(url_for('intern_login'))

@app.route('/register_internship', methods=['GET', 'POST'], strict_slashes=False)
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
        company_name = recruiter['organization_name'] or ''
        company_mail = recruiter['email'] or ''
        expected_salary = request.form.get('expected_salary', '')
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
        conn.execute('''INSERT INTO internship_info (id, role, description_of_internship, start_date, end_date, duration, type_of_internship,
                       skills_required, location, years_of_experience, phone_number, company_name, company_mail, user_id, posted_date, expected_salary)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (new_internship_id, role, description, start_date, end_date, duration, type_of_internship,
                      skills_required, location, years_of_experience, phone_number, company_name, company_mail, user_id,
                      datetime.now().strftime('%Y-%m-%d %H:%M:%S'), expected_salary))
        conn.commit()
        conn.close()
        global resume_df, internship_df
        resume_df, internship_df = fetch_data()
        flash('Internship registered successfully!', 'success')
        return redirect(url_for('recruiter_dashboard'))
    conn.close()
    return render_template('register_internship.html', recruiter=recruiter)

@app.route('/upload_resume', methods=['GET', 'POST'], strict_slashes=False)
def upload_resume():
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern.', 'danger')
        return redirect(url_for('intern_login'))
    if request.method == 'POST':
        if 'resume' not in request.files:
            flash('No file part found', 'danger')
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
            skills = request.form.get('skills')
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

@app.route('/create_resume', methods=['GET', 'POST'], strict_slashes=False)
def create_resume():
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern!', 'danger')
        return redirect(url_for('intern_login'))
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        skills = request.form['skills']
        experience = request.form.get('experience')
        education = request.form.get('education')
        certifications = request.form.get('certifications')
        achievements = request.form.get('achievements')
        user_id = int(session['user_id'])
        resume = {
            'user_id': user_id,
            'name_of_applicant': name,
            'email': email,
            'phone_number': phone,
            'skills': skills,
            'experience': experience,
            'education': education,
            'certifications': certifications,
            'achievements': achievements,
            'downloaded': 0
        }
        
        conn = get_db_connection()
        existing_resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
        if existing_resume:
            conn.execute('''
                UPDATE resume_info SET name_of_applicant = ?, email = ?, phone_number = ?, skills = ?,
                experience = ?, education = ?, certifications = ?, achievements = ?
                WHERE user_id = ?
            ''',
            (name, email, phone, skills, experience, education, certifications, achievements, user_id))
        else:
            conn.execute('''
                INSERT INTO resume_info (user_id, name_of_applicant, email, phone_number,
                skills, experience, education, certifications, achievements, downloaded)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (user_id, name, email, phone, skills, experience, education, certifications, achievements, 0))
        conn.commit()
        conn.close()
        global resume_df, internship_df
        resume_df, internship_df = fetch_data()
        flash('Resume created successfully!', 'success')
        return redirect(url_for('intern_dashboard'))
    return render_template('create_resume.html')

@app.route('/edit_resume', methods=['GET', 'POST'], strict_slashes=False)
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
        experience = request.form.get('experience')
        education = request.form.get('education')
        certifications = request.form.get('certifications')
        achievements = request.form.get('achievements')
        conn.execute('''
            UPDATE resume_info SET name_of_applicant = ?, email = ?, phone_number = ?, skills = ?,
                   experience = ?, education = ?, certifications = ?, achievements = ?
                   WHERE user_id = ?
        ''',
        (name, email, phone, skills, experience, education, certifications, achievements, user_id))
        conn.commit()
        conn.close()
        global resume_df, internship_df
        resume_df, internship_df = fetch_data()
        flash('Resume updated successfully!', 'success')
        return redirect(url_for('intern_dashboard'))
    conn.close()
    return render_template('edit_resume.html', resume=resume)

@app.route('/match', methods=['GET'], strict_slashes=False)
def match():
    logging.info(f"Match attempt, session data: {dict(session)}")
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern.', 'danger')
        return redirect(url_for('index'))
    user_id = int(session['user_id'])
    try:
        global resume_df, internship_df
        resume_df, internship_df = fetch_data()
        logging.info(f"[MATCH DEBUG] Refreshed data: resume_df shape: {resume_df.shape}, internship_df shape: {internship_df.shape}")
        
        conn = get_db_connection()
        resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
        conn.close()
        if not resume:
            flash('Please create your resume first!', 'warning')
            return redirect(url_for('create_resume'))
        if 'skills' not in resume or not resume['skills']:
            logging.warning(f"User {user_id} resume missing skills")
            flash('Please add skills to your resume to match internships!', 'warning')
            return redirect(url_for('edit_resume'))
        user_skills = preprocess_skills(resume['skills'])
        logging.info(f"[MATCH DEBUG] User {user_id} processed skills: {user_skills}")
        matched_internships = []
        if not internship_df.empty and 'processed_Required_Skills' in internship_df.columns:
            logging.info(f"[MATCH DEBUG] internship_df has {len(internship_df)} internships")
            for idx, internship in internship_df.iterrows():
                similarity = jaccard_similarity(user_skills, internship['processed_Required_Skills'])
                logging.info(f"[MATCH] Internship {internship.get('id', 0)} similarity: {similarity}")
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
            logging.warning("[MATCH] internship_df empty or missing processed_Required_Skills column")
            flash('No internships available to match at this time.', 'info')
            return render_template('match.html', matched_internships=[])
        logging.info(f"[MATCH] Found {len(matched_internships)} matched internships for user {user_id}")
        matched_internships = sorted(matched_internships, key=lambda x: x['similarity_score'], reverse=True)
        if not matched_internships:
            flash('No matching internships found. Try updating your skills.', 'info')
        return render_template('match.html', matched_internships=matched_internships)
    except Exception as e:
        logging.error(f"Error in /match for user {user_id}: {str(e)}")
        flash('An error occurred while matching internships. Please try again.', 'danger')
        return redirect(url_for('intern_dashboard'))

@app.route('/top_matched_applicants/<int:internship_id>', strict_slashes=False)
def top_matched_applicants(internship_id):
    logging.info(f"Top matched applicants for internship {internship_id}, session data: {dict(session)}")
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter.', 'danger')
        logging.info("Redirecting to recruiter_login due to missing session or invalid role")
        return redirect(url_for('recruiter_login'))
    user_id = int(session['user_id'])
    try:
        # Refresh data
        global resume_df, internship_df
        resume_df, internship_df = fetch_data()
        logging.info(f"[TOP MATCH DEBUG] Refreshed data: resume_df shape: {resume_df.shape}, internship_df shape: {internship_df.shape}")
        
        # Verify internship exists and belongs to recruiter
        conn = get_db_connection()
        internship = conn.execute('SELECT * FROM internship_info WHERE id = ? AND user_id = ?', (internship_id, user_id)).fetchone()
        if not internship:
            conn.close()
            flash('Internship not found or you do not have access.', 'danger')
            logging.warning(f"Internship {internship_id} not found or not owned by user {user_id}")
            return render_template('top_matched_applicants.html', matched_applicants=[], internship_title='Unknown')
        
        internship_title = internship['role']
        internship_skills = preprocess_skills(internship['skills_required'])
        logging.info(f"[TOP MATCH DEBUG] Internship {internship_id} processed skills: {internship_skills}")
        
        matched_applicants = []
        if not resume_df.empty and 'processed_Skills' in resume_df.columns:
            logging.info(f"[TOP MATCH DEBUG] resume_df has {len(resume_df)} resumes")
            for idx, resume in resume_df.iterrows():
                similarity = jaccard_similarity(resume['processed_Skills'], internship_skills)
                logging.info(f"[TOP MATCH DEBUG] Resume for user {resume['user_id']} similarity: {similarity}")
                if similarity > 0:
                    user = conn.execute('SELECT * FROM users WHERE user_id = ?', (resume['user_id'],)).fetchone()
                    if user:
                        matched_applicants.append({
                            'name': resume['name_of_applicant'],
                            'email': user['email'],
                            'skills': resume['skills'],
                            'similarity_score': round(similarity * 100, 2),
                            'resume_path': resume.get('resume_path', '')
                        })
        else:
            logging.warning("[TOP MATCH DEBUG] resume_df is empty or missing processed_Skills column")
        
        conn.close()
        matched_applicants = sorted(matched_applicants, key=lambda x: x['similarity_score'], reverse=True)[:5]
        logging.info(f"[TOP MATCH DEBUG] Found {len(matched_applicants)} matched applicants for internship {internship_id}")
        
        return render_template('top_matched_applicants.html', matched_applicants=matched_applicants, internship_title=internship_title, user_name=session.get('user_name'))
    except Exception as e:
        logging.error(f"Error in /top_matched_applicants for internship {internship_id}: {str(e)}")
        flash('An error occurred while fetching matched applicants. Please try again.', 'danger')
        return render_template('top_matched_applicants.html', matched_applicants=[], internship_title='Unknown', user_name=session.get('user_name'))

@app.route('/download_resume/<path:resume_path>', strict_slashes=False)
def download_resume(resume_path):
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter.', 'danger')
        return redirect(url_for('recruiter_login'))
    try:
        return send_file(resume_path, as_attachment=True)
    except FileNotFoundError:
        flash('Resume file not found.', 'danger')
        return redirect(url_for('recruiter_dashboard'))

@app.route('/apply_internship/<int:internship_id>', methods=['POST'], strict_slashes=False)
def apply_internship(internship_id):
    logging.info(f"Applying to internship {internship_id}, session data: {dict(session)}")
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern!', 'danger')
        return redirect(url_for('index'))
    user_id = int(session['user_id'])
    try:
        conn = get_db_connection()
        internship = conn.execute('SELECT * FROM internship_info WHERE id = ?', (internship_id,)).fetchone()
        if not internship:
            conn.close()
            flash('Internship not found.', 'danger')
            logging.warning(f"Internship {internship_id} not found")
            return redirect(url_for('intern_dashboard'))
        existing_application = conn.execute('SELECT * FROM applications WHERE user_id = ? AND internship_id = ?', (user_id, internship_id)).fetchone()
        if existing_application:
            conn.close()
            flash('You have already applied to this internship!', 'warning')
            logging.info(f"User {user_id} already applied to internship {internship_id}")
            return redirect(url_for('intern_dashboard'))
        applied_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn.execute('INSERT INTO applications (user_id, internship_id, applied_at) VALUES (?, ?, ?)', (user_id, internship_id, applied_at))
        conn.commit()
        logging.info(f"Application recorded: user_id={user_id}, internship_id={internship_id}, applied_at={applied_at}")
        conn.close()
        flash('Applied successfully!', 'success')
        return redirect(url_for('intern_dashboard'))
    except sqlite3.Error as db_error:
        logging.error(f"Database error applying to internship {internship_id} for user {user_id}: {str(db_error)}")
        flash('A database error occurred while applying. Please try again.', 'danger')
        conn.close()
        return redirect(url_for('intern_dashboard'))
    except Exception as e:
        logging.error(f"Unexpected error applying to internship {internship_id} for user {user_id}: {str(e)}")
        flash('An unexpected error occurred while applying. Please try again.', 'danger')
        conn.close()
        return redirect(url_for('intern_dashboard'))

@app.route('/applied_internships', strict_slashes=False)
def applied_internships():
    logging.info(f"Fetching applied internships, session data: {dict(session)}")
    if 'user_id' not in session or session['role'] != 'intern':
        flash('Please login as an intern!', 'danger')
        return redirect(url_for('intern_login'))
    user_id = int(session['user_id'])
    try:
        conn = get_db_connection()
        query = '''
        SELECT i.*, a.applied_at
        FROM applications a
        JOIN internship_info i ON a.internship_id = i.id
        WHERE a.user_id = ?
        '''
        applied_internships = conn.execute(query, (user_id,)).fetchall()
        conn.close()
        applied_internships_list = [
            {
                'role': internship['role'],
                'company_name': internship['company_name'],
                'type_of_internship': internship['type_of_internship'],
                'location': internship['location'],
                'description': internship['description_of_internship'],
                'applied_at': internship['applied_at']
            }
            for internship in applied_internships
        ]
        logging.info(f"[APPLIED DEBUG] Found {len(applied_internships_list)} applied internships for user {user_id}")
        return render_template('applied_internships.html', applied_internships=applied_internships_list, user_name=session.get('user_name'))
    except Exception as e:
        logging.error(f"Error in /applied_internships for user {user_id}: {str(e)}")
        flash('An error occurred while fetching applied internships. Please try again.', 'danger')
        return redirect(url_for('intern_dashboard'))

@app.route('/applied_applicants', strict_slashes=False)
def applied_applicants():
    logging.info(f"Fetching applied applicants, session data: {dict(session)}")
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter!', 'danger')
        return redirect(url_for('recruiter_login'))
    user_id = int(session['user_id'])
    try:
        conn = get_db_connection()
        query = '''
        SELECT r.name_of_applicant, u.email, i.role, a.applied_at, r.resume_path
        FROM applications a
        JOIN resume_info r ON a.user_id = r.user_id
        JOIN users u ON a.user_id = u.user_id
        JOIN internship_info i ON a.internship_id = i.id
        WHERE i.user_id = ?
        '''
        results = conn.execute(query, (user_id,)).fetchall()
        conn.close()
        
        applicants = [
            {
                'name': applicant['name_of_applicant'],
                'email': applicant['email'],
                'internship_title': applicant['role'],
                'applied_at': applicant['applied_at'],
                'resume_path': applicant['resume_path'] or ''
            }
            for applicant in results
        ]
        logging.info(f"[APPLIED APPLICANTS] Found {len(applicants)} applicants for user {user_id}")
        
        return render_template('applied_applicants.html', applicants=applicants, internship=None, user_name=session.get('user_name'))
    except Exception as e:
        logging.error(f"Error in /applied_applicants for user {user_id}: {str(e)}")
        flash('An error occurred while fetching applied applicants. Please try again.', 'danger')
        return redirect(url_for('recruiter_dashboard'))

@app.route('/applied_applicants/<int:internship_id>', strict_slashes=False)
def applied_applicants_specific(internship_id):
    logging.info(f"Fetching applicants for internship {internship_id}, session data: {dict(session)}")
    if 'user_id' not in session or session['role'] != 'recruiter':
        flash('Please login as a recruiter!', 'danger')
        return redirect(url_for('recruiter_login'))
    user_id = int(session['user_id'])
    try:
        conn = get_db_connection()
        # Verify internship belongs to recruiter
        internship = conn.execute('SELECT * FROM internship_info WHERE id = ? AND user_id = ?', (internship_id, user_id)).fetchone()
        if not internship:
            conn.close()
            flash('Internship not found or you do not have access.', 'danger')
            logging.warning(f"Internship {internship_id} not found or not owned by user {user_id}")
            return render_template('applied_applicants.html', applicants=[], internship=None, user_name=session.get('user_name'))
        
        query = '''
        SELECT r.name_of_applicant, u.email, r.skills, r.experience, r.education, r.resume_path, a.applied_at
        FROM applications a
        JOIN resume_info r ON a.user_id = r.user_id
        JOIN users u ON a.user_id = u.user_id
        WHERE a.internship_id = ?
        '''
        applicants = conn.execute(query, (internship_id,)).fetchall()
        conn.close()
        
        applicants_list = [
            {
                'name': applicant['name_of_applicant'],
                'email': applicant['email'],
                'skills': applicant['skills'],
                'experience': applicant['experience'],
                'education': applicant['education'],
                'resume_path': applicant['resume_path'] or '',
                'applied_at': applicant['applied_at']
            }
            for applicant in applicants
        ]
        logging.info(f"[APPLICANTS SPECIFIC] Found {len(applicants_list)} applicants for internship {internship_id}")
        
        return render_template('applied_applicants.html', applicants=applicants_list, internship=internship, user_name=session.get('user_name'))
    except Exception as e:
        logging.error(f"Error in /applied_applicants/{internship_id} for user {user_id}: {str(e)}")
        flash('An error occurred while fetching applicants. Please try again.', 'danger')
        return redirect(url_for('recruiter_dashboard'))

@app.route('/edit_profile', methods=['GET', 'POST'], strict_slashes=False)
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
            flash('Email is already in use by another user!', 'danger')
            conn.close()
            return render_template('edit_profile.html', user=user)
        if password:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
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

@app.route('/edit_organization_profile', methods=['GET', 'POST'], strict_slashes=False)
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
            flash('Organization name is required!', 'danger')
            conn.close()
            return render_template('edit_organization_profile.html', recruiter=user)
        if website_link:
            url_pattern = r'^(https?://)?([\da-z.-]+)\.([a-z.]{2,6})([/\w .-]*)*\/?$'
            if not re.match(url_pattern, website_link):
                flash('Invalid website link format!', 'danger')
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

@app.route('/logout', strict_slashes=False)
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    error_message = f"Exception: {str(e)}\n{traceback.format_exc()}"
    logging.error(error_message)
    return render_template('500.html', error_message=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 7860)))