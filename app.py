from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import sqlite3
import os
import logging
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import re
import secrets
import hashlib
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.exceptions import InvalidSignature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from functools import wraps
from collections import Counter
from bleach import clean

# Configure logging
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
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# CSRF protection
csrf = CSRFProtect(app)

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response

# Input sanitization
def sanitize_input(text):
    if not isinstance(text, str):
        return ''
    return clean(text, tags=[], strip=True)

# NLTK data path
nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)
logging.info(f"NLTK data directory: {nltk_data_dir}, writable: {os.access(nltk_data_dir, os.W_OK)}")

# Database configuration
DB_PATH = '/tmp/database.db'

# Public key for signature verification
PUBLIC_KEY_PEM = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA2zQcp+kSdpAgcey6Z8aG
8io1R8X5TpqJNlVOcxCb0cioW5RcPNs9ScfwyewZrAvcyWT0koo6Ir6yPaA2kNz5
tDuCh8ud5gqC+2flklLQ56vu7UDzlrLgcMAxAgwKgLku7vX2q5HltKXNAcH452fI
fkyxBAJIaaMfAcCNUkQf/LSmWsqZQY96N1AypbYVqDKlhaE8V/RY0dlqLTsj10u3
arhfJzpXYnpUv61dAkDh7ENJcOE5UAO87vZSFuL/eDJOSGLjOi7gf/km9fniyoEO
l28dSBTRhPPPzNvIGnYicQAPO7aBsLpUni2mAbP2aFPFL8a1TvpP32BOnIZASP8i
YQIDAQAB
-----END PUBLIC KEY-----
"""

# SQLite Connection
def get_db_connection():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logging.error(f"Database connection error: {str(e)}")
        return None

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
        enhanced_resume TEXT,
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
    CREATE TABLE IF NOT EXISTS user_progress (
        user_id INTEGER,
        task_type TEXT,
        task_description TEXT,
        completion_date TEXT,
        points INTEGER,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    CREATE TABLE IF NOT EXISTS peer_reviews (
        review_id INTEGER PRIMARY KEY AUTOINCREMENT,
        reviewer_id INTEGER,
        reviewed_user_id INTEGER,
        resume_id INTEGER,
        rating INTEGER,
        feedback TEXT,
        review_date TEXT,
        FOREIGN KEY (reviewer_id) REFERENCES users(user_id),
        FOREIGN KEY (reviewed_user_id) REFERENCES users(user_id)
    );
    CREATE TABLE IF NOT EXISTS internship_ratings (
        rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        internship_id INTEGER,
        rating INTEGER,
        feedback TEXT,
        rating_date TEXT,
        FOREIGN KEY (user_id) REFERENCES users(user_id),
        FOREIGN KEY (internship_id) REFERENCES internship_info(id)
    );
    CREATE TABLE IF NOT EXISTS credentials (
        credential_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        credential_hash TEXT NOT NULL,
        signature TEXT NOT NULL,
        issued_date TEXT,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    '''
    with get_db_connection() as conn:
        conn.executescript(schema)
        conn.commit()
        logging.info("Database schema initialized")

def insert_test_data():
    with get_db_connection() as conn:
        existing_user = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
        if existing_user > 0:
            logging.info("Test data already exists")
            return
        admin_password_hash = generate_password_hash('password')
        test_data_sql = f'''
        DELETE FROM applications;
        DELETE FROM internship_info;
        DELETE FROM resume_info;
        DELETE FROM users;
        DELETE FROM user_progress;
        DELETE FROM peer_reviews;
        DELETE FROM internship_ratings;
        DELETE FROM credentials;
        DELETE FROM sqlite_sequence;
        INSERT INTO users (user_id, name, email, password, role, skills) VALUES
        (1, 'Alice Smith', 'alice.smith@example.com', '{admin_password_hash}', 'intern', 'python, java, sql, tensorflow'),
        (2, 'Bob Johnson', 'bob.johnson@example.com', '{admin_password_hash}', 'intern', 'javascript, react, node.js'),
        (3, 'Carol Lee', 'carol.lee@example.com', '{admin_password_hash}', 'intern', 'python, django, postgresql'),
        (4, 'David Brown', 'david.brown@example.com', '{admin_password_hash}', 'intern', 'c++, opencv, machine learning'),
        (8, 'Admin User', 'admin@example.com', '{admin_password_hash}', 'admin', NULL);
        INSERT INTO users (user_id, name, email, password, role, organization_name, contact_details, location, website_link) VALUES
        (5, 'Emma Wilson', 'emma.wilson@techcorp.com', '{admin_password_hash}', 'recruiter', 'TechCorp', '+1-800-555-1234', 'San Francisco, CA', 'https://techcorp.com'),
        (6, 'Frank Taylor', 'frank.taylor@innovatech.com', '{admin_password_hash}', 'recruiter', 'Innovatech', '+1-800-555-5678', 'New York, NY', 'https://innovatech.com'),
        (7, 'Grace Miller', 'grace.miller@datatech.com', '{admin_password_hash}', 'recruiter', 'DataTech', '+1-800-555-9012', 'Austin, TX', 'https://datatech.com');
        INSERT INTO resume_info (user_id, name_of_applicant, email, phone_number, skills, experience, education, certifications, achievements, resume_path, downloaded) VALUES
        (1, 'Alice Smith', 'alice.smith@example.com', '+1-555-123-4567', 'python, java, sql, tensorflow', 'Software Intern at XYZ Corp (6 months)', 'B.S. Computer Science, MIT, 2024', 'AWS Certified Developer', 'Completed Hackathon 2023', 'static/uploads/1_resume.pdf', 0),
        (2, 'Bob Johnson', 'bob.johnson@example.com', '+1-555-234-5678', 'javascript, react, node.js', 'Frontend Developer at ABC Inc (1 year)', 'B.S. Software Engineering', 'Stanford, 2023', 'React Professional Certification', 'Published NPM package', 'static/uploads/2_resume.pdf', 0),
        (3, 'David Brown', 'david.brown@example.com', '+1-555-456-7890', 'c++, python, opencv', 'Research Assistant (8 months)', 'M.S. AI', 'Caltech, 2025', NULL, NULL, 'static/uploads/4_resume.csv', 0),
        (4, 'David Brown', 'david.brown@example.com', '+1-555-456-7890', 'c++, opencv, machine learning', 'Research Assistant at GHI University (8 months)', 'M.S. AI', 'Caltech, 2025', 'TensorFlow Developer Certificate', 'Published CVPR 2024 paper', 'static/uploads/4_resume.pdf', 0);
        INSERT INTO internship_info (id, role, description_of_internship, start_date, end_date, duration, type_of_internship, skills_required, location, years_of_experience, phone_number, company_name, company_mail, user_id, posted_date, expected_salary) VALUES
        (1, 'Machine Learning Intern', 'Develop ML models for image recognition', '2025-07-01', '2025-12-31', '6 months', 'Full-time', 'python, tensorflow, machine learning', 'San Francisco, CA', 0, '+1-800-555-1234', 'TechCorp', 'emma.wilson@techcorp.com', 5, '2025-06-15', 'Unpaid'),
        (2, 'Frontend Developer Intern', 'Build responsive web interfaces', '2025-08-01', '2025-11-30', '4 months', 'Part-time', 'javascript, react, css', 'New York, NY', 0, '+1-800-555-5678', 'Innovatech', 'frank.taylor@innovatech.com', 6, '2025-06-15', '$15/hr'),
        (3, 'Backend Developer Intern', 'Develop APIs and database systems', '2025-07-15', '2026-01-15', '6 months', 'Full-time', 'python, django, sql', 'Austin, TX', 1, '+1-800-555-9012', 'DataTech', 'grace.miller@datatech.com', 7, '2025-06-15', 'Unpaid'),
        (4, 'AI Research Intern', 'Research on computer vision algorithms', '2025-09-01', '2026-02-28', '6 months', 'Full-time', 'c++, opencv, python', 'San Francisco, CA', 1, '+1-800-555-1234', 'TechCorp', 'emma.wilson@techcorp.com', 5, '2025-06-15', '$20/hr'),
        (5, 'Full Stack Intern', 'Work on both frontend and backend', '2025-07-01', '2025-12-31', '6 months', 'Full-time', 'javascript, node.js, postgresql', 'New York, NY', 0, '+1-800-555-5678', 'Innovatech', 'frank.taylor@innovatech.com', 6, '2025-06-15', 'Unpaid'),
        (6, 'Data Science Intern', 'Analyze data and build predictive models', '2025-08-01', '2025-12-31', '5 months', 'Part-time', 'python, sql, machine learning', 'Austin, TX', 0, '+1-800-555-9012', 'DataTech', 'grace.miller@datatech.com', 7, '2025-06-15', '$18/hr');
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
        INSERT INTO user_progress (user_id, task_type, task_description, completion_date, points) VALUES
        (1, 'Resume Creation', 'Created ATS-friendly resume', '2025-06-15', 100),
        (1, 'Application', 'Applied to Machine Learning Intern', '2025-06-15', 50),
        (2, 'Resume Creation', 'Created ATS-friendly resume', '2025-06-15', 100);
        INSERT INTO peer_reviews (reviewer_id, reviewed_user_id, resume_id, rating, feedback, review_date) VALUES
        (2, 1, 1, 4, 'Great skills section, consider adding more project details.', '2025-06-15'),
        (3, 1, 1, 3, 'Good resume, but formatting can be improved.', '2025-06-15');
        INSERT INTO internship_ratings (user_id, internship_id, rating, feedback, rating_date) VALUES
        (1, 1, 5, 'Excellent learning opportunity.', '2025-06-15'),
        (2, 2, 4, 'Good project, but limited mentorship.', '2025-06-15');
        '''
        conn.executescript(test_data_sql)
        conn.commit()
        logging.info("Inserted comprehensive test data")

if not os.path.exists(DB_PATH):
    initialize_database()
    insert_test_data()
else:
    with get_db_connection() as conn:
        user_count = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
        if user_count == 0:
            logging.warning("Test data missing, reinitializing")
            initialize_database()
            insert_test_data()

# Global data
resume_df = pd.DataFrame()
internship_df = pd.DataFrame()
skill_to_index = {}

# NLP preprocessing
lemmatizer = WordNetLemmatizer()
def preprocess_skills(skills):
    if not isinstance(skills, str) or skills.strip() == '':
        return []
    skills_clean = re.sub(r'[;|\\/]', ',', skills)
    tokens = [s.strip().lower() for s in skills_clean.split(',') if s.strip()]
    stop_words = set(stopwords.words('english'))
    processed = []
    for token in tokens:
        token = ''.join([c for c in token if c not in string.punctuation])
        if token and token not in stop_words:
            processed.append(lemmatizer.lemmatize(token))
    return processed

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = sanitize_input(text)
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and t.isalnum()]
    return ' '.join(tokens)

# Input validation
def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

def validate_phone(phone):
    pattern = r'^\+?[\d\s-]{10,15}$'
    return bool(re.match(pattern, phone))

# Fetch data
def fetch_data():
    global resume_df, internship_df, skill_to_index
    try:
        with get_db_connection() as conn:
            resumes = conn.execute('SELECT * FROM resume_info').fetchall()
            internships = conn.execute('SELECT * FROM internship_info').fetchall()
            resume_df = pd.DataFrame([dict(row) for row in resumes])
            internship_df = pd.DataFrame([dict(row) for row in internships])
        resume_df.fillna('', inplace=True)
        internship_df.fillna('', inplace=True)
        resume_df['processed_Skills'] = resume_df['skills'].apply(preprocess_skills)
        internship_df['processed_Required_Skills'] = internship_df['skills_required'].apply(preprocess_skills)
        resume_skills = resume_df['processed_Skills'].explode().dropna().tolist()
        internship_skills = internship_df['processed_Required_Skills'].explode().dropna().tolist()
        all_skills = resume_skills + internship_skills
        skill_to_index = {skill: idx for idx, skill in enumerate(set(all_skills)) if all_skills}
        def skills_to_vector(skills):
            vector = [0] * len(skill_to_index)
            for skill in skills:
                if skill in skill_to_index:
                    vector[skill_to_index[skill]] += 1
            return vector
        resume_df['Skill_vector'] = resume_df['processed_Skills'].apply(skills_to_vector)
        internship_df['Required_Skill_vector'] = internship_df['processed_Required_Skills'].apply(skills_to_vector)
        return resume_df, internship_df
    except Exception as e:
        logging.error(f"Error fetching data: {str(e)}")
        raise e

resume_df, internship_df = fetch_data()

# Similarity functions
def jaccard_similarity(skills1, skills2):
    set1 = set(skills1)
    set2 = set(skills2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0

def cosine_sim(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Role-based access decorator
def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session or session['role'] != role:
                flash(f'Please login as a {role}.', 'danger')
                return redirect(url_for(f'{role}_login'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Routes
@app.route('/', strict_slashes=False)
def index():
    return render_template('index.html')

@app.route('/admin_login', methods=['GET', 'POST'], strict_slashes=False)
@limiter.limit("10 per minute")
def admin_login():
    if request.method == 'POST':
        email = sanitize_input(request.form.get('email', ''))
        password = request.form.get('password', '')
        if not validate_email(email):
            flash('Invalid email format.', 'danger')
            return render_template('admin_login.html')
        with get_db_connection() as conn:
            if not conn:
                flash('Database error.', 'danger')
                return render_template('admin_login.html')
            user = conn.execute('SELECT * FROM users WHERE email = ? AND role = ?', (email, 'admin')).fetchone()
            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['user_id']
                session['user_name'] = user['name']
                session['role'] = 'admin'
                logging.info(f"Admin login: Email={email}, user_id={user['user_id']}")
                flash('Login successful!', 'success')
                return redirect(url_for('issue_credential'))
            flash('Invalid credentials.', 'danger')
    return render_template('admin_login.html')

@app.route('/recruiter_login', methods=['GET', 'POST'], strict_slashes=False)
@limiter.limit("10 per minute")
def recruiter_login():
    if request.method == 'POST':
        email = sanitize_input(request.form.get('email', ''))
        password = request.form.get('password', '')
        if not validate_email(email):
            flash('Invalid email format.', 'danger')
            return render_template('recruiter_login.html')
        with get_db_connection() as conn:
            if not conn:
                flash('Database error.', 'danger')
                return render_template('recruiter_login.html')
            user = conn.execute('SELECT * FROM users WHERE email = ? AND role = ?', (email, 'recruiter')).fetchone()
            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['user_id']
                session['user_name'] = user['name']
                session['email'] = user['email']
                session['role'] = 'recruiter'
                logging.info(f"Recruiter login: Email={email}, user_id={user['user_id']}")
                flash('Login successful!', 'success')
                return redirect(url_for('recruiter_dashboard'))
            flash('Invalid credentials.', 'danger')
    return render_template('recruiter_login.html')

@app.route('/intern_login', methods=['GET', 'POST'], strict_slashes=False)
@limiter.limit("10 per minute")
def intern_login():
    if request.method == 'POST':
        email = sanitize_input(request.form.get('email', ''))
        password = request.form.get('password', '')
        if not validate_email(email):
            flash('Invalid email format.', 'danger')
            return render_template('intern_login.html')
        with get_db_connection() as conn:
            if not conn:
                flash('Database error.', 'danger')
                return render_template('intern_login.html')
            user = conn.execute('SELECT * FROM users WHERE email = ? AND role = ?', (email, 'intern')).fetchone()
            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['user_id']
                session['user_name'] = user['name']
                session['role'] = 'intern'
                logging.info(f"Intern login: Email={email}, user_id={user['user_id']}")
                flash('Login successful!', 'success')
                return redirect(url_for('intern_dashboard'))
            flash('Invalid credentials.', 'danger')
    return render_template('intern_login.html')

@app.route('/recruiter_signup', methods=['GET', 'POST'], strict_slashes=False)
@limiter.limit("5 per hour")
def recruiter_signup():
    if request.method == 'POST':
        name = sanitize_input(request.form.get('name', ''))
        email = sanitize_input(request.form.get('email', ''))
        password = request.form.get('password', '')
        company = sanitize_input(request.form.get('company', ''))
        if not all([name, email, password, company]) or len(password) < 8:
            flash('All fields are required and password must be at least 8 characters.', 'danger')
            return render_template('recruiter_signup.html')
        if not validate_email(email):
            flash('Invalid email format.', 'danger')
            return render_template('recruiter_signup.html')
        with get_db_connection() as conn:
            if not conn:
                flash('Database error.', 'danger')
                return render_template('recruiter_signup.html')
            existing = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            if existing:
                flash('Email already exists.', 'danger')
            else:
                hashed_password = generate_password_hash(password)
                max_user = conn.execute('SELECT MAX(user_id) as max_id FROM users').fetchone()
                new_user_id = (max_user['max_id'] + 1) if max_user['max_id'] else 1
                conn.execute('INSERT INTO users (user_id, name, email, password, role, organization_name) VALUES (?, ?, ?, ?, ?, ?)', 
                             (new_user_id, name, email, hashed_password, 'recruiter', company))
                conn.commit()
                flash('Signup successful! Please login.', 'success')
                return redirect(url_for('recruiter_login'))
    return render_template('recruiter_signup.html')

@app.route('/intern_signup', methods=['GET', 'POST'], strict_slashes=False)
@limiter.limit("5 per hour")
def intern_signup():
    if request.method == 'POST':
        name = sanitize_input(request.form.get('name', ''))
        email = sanitize_input(request.form.get('email', ''))
        password = request.form.get('password', '')
        skills = sanitize_input(request.form.get('skills', ''))
        education = sanitize_input(request.form.get('education', ''))
        certifications = sanitize_input(request.form.get('certifications', ''))
        if not all([name, email, password, skills]) or len(password) < 8:
            flash('Required fields missing or password too short.', 'danger')
            return render_template('intern_signup.html')
        if not validate_email(email):
            flash('Invalid email format.', 'danger')
            return render_template('intern_signup.html')
        with get_db_connection() as conn:
            if not conn:
                flash('Database error.', 'danger')
                return render_template('intern_signup.html')
            existing = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            if existing:
                flash('Email already exists.', 'danger')
            else:
                hashed_password = generate_password_hash(password)
                max_user = conn.execute('SELECT MAX(user_id) as max_id FROM users').fetchone()
                new_user_id = (max_user['max_id'] + 1) if max_user['max_id'] else 1
                conn.execute('INSERT INTO users (user_id, name, email, password, role, skills) VALUES (?, ?, ?, ?, ?, ?)', 
                             (new_user_id, name, email, hashed_password, 'intern', skills))
                conn.execute('INSERT INTO resume_info (user_id, name_of_applicant, email, skills, education, certifications) VALUES (?, ?, ?, ?, ?, ?)',
                             (new_user_id, name, email, skills, education, certifications))
                conn.commit()
                flash('Signup successful! Please login.', 'success')
                return redirect(url_for('intern_login'))
    return render_template('intern_signup.html')

@app.route('/admin_signup', methods=['GET', 'POST'], strict_slashes=False)
@limiter.limit("5 per hour")
def admin_signup():
    if request.method == 'POST':
        name = sanitize_input(request.form.get('name', ''))
        email = sanitize_input(request.form.get('email', ''))
        password = request.form.get('password', '')
        secret_code = request.form.get('secret_code', '')
        if not all([name, email, password, secret_code]) or len(password) < 8:
            flash('All fields are required and password must be at least 8 characters.', 'danger')
            return render_template('admin_signup.html')
        if not validate_email(email):
            flash('Invalid email format.', 'danger')
            return render_template('admin_signup.html')
        expected_secret_code = os.getenv('ADMIN_SECRET_CODE', 'default-secret-code')
        if secret_code != expected_secret_code:
            flash('Invalid secret code.', 'danger')
            return render_template('admin_signup.html')
        try:
            with get_db_connection() as conn:
                if not conn:
                    flash('Database error.', 'danger')
                    return render_template('admin_signup.html')
                hashed_password = generate_password_hash(password)
                conn.execute('INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)',
                             (name, email, hashed_password, 'admin'))
                conn.commit()
                flash('Admin sign up successful! Please login.', 'success')
                return redirect(url_for('admin_login'))
        except sqlite3.IntegrityError:
            flash('Email already registered.', 'danger')
        except Exception as e:
            logging.error(f"Admin signup error: {str(e)}")
            flash('Error during signup.', 'danger')
    return render_template('admin_signup.html')

@app.route('/recruiter_dashboard', strict_slashes=False)
@role_required('recruiter')
def recruiter_dashboard():
    user_id = session['user_id']
    with get_db_connection() as conn:
        if not conn:
            flash('Database error.', 'danger')
            return redirect(url_for('recruiter_login'))
        recruiter = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
        internships = conn.execute('SELECT * FROM internship_info WHERE user_id = ?', (user_id,)).fetchall()
    return render_template('recruiter_dashboard.html', recruiter=recruiter, internships=internships, user_name=session['user_name'])

@app.route('/intern_dashboard', strict_slashes=False)
@role_required('intern')
def intern_dashboard():
    user_id = session['user_id']
    global resume_df, internship_df
    resume_df, internship_df = fetch_data()
    with get_db_connection() as conn:
        if not conn:
            flash('Database error.', 'danger')
            return redirect(url_for('intern_login'))
        intern = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
        resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
        applications = conn.execute('SELECT * FROM applications WHERE user_id = ?', (user_id,)).fetchall()
        progress = conn.execute('SELECT * FROM user_progress WHERE user_id = ?', (user_id,)).fetchall()
    if not resume:
        flash('Please create your resume!', 'warning')
        return redirect(url_for('create_resume'))
    applied_internship_ids = [app['internship_id'] for app in applications]
    user_skills = preprocess_skills(resume['skills'])
    internships = []
    for idx, internship in internship_df.iterrows():
        similarity = jaccard_similarity(user_skills, internship['processed_Required_Skills'])
        if similarity > 0:
            internships.append({
                'id': internship['id'],
                'role': internship['role'],
                'company_name': internship['company_name'],
                'description': internship['description_of_internship'],
                'duration': internship['duration'],
                'type_of_internship': internship['type_of_internship'],
                'skills_required': internship['skills_required'],
                'location': internship['location'],
                'similarity_score': round(similarity * 100, 2)
            })
    internships = sorted(internships, key=lambda x: x['similarity_score'], reverse=True)
    total_points = sum(p['points'] for p in progress) if progress else 0
    level = total_points // 100
    return render_template('intern_dashboard.html', user_name=session['user_name'], internships=internships, applied_internship_ids=applied_internship_ids, total_points=total_points, level=level)

@app.route('/register_internship', methods=['GET', 'POST'], strict_slashes=False)
@role_required('recruiter')
def register_internship():
    user_id = session['user_id']
    with get_db_connection() as conn:
        if not conn:
            flash('Database error.', 'danger')
            return redirect(url_for('recruiter_login'))
        recruiter = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
        if request.method == 'POST':
            role = sanitize_input(request.form.get('role', ''))
            description = sanitize_input(request.form.get('description_of_internship', ''))
            start_date = sanitize_input(request.form.get('start_date', ''))
            end_date = sanitize_input(request.form.get('end_date', ''))
            duration = sanitize_input(request.form.get('duration', ''))
            type_of_internship = sanitize_input(request.form.get('type_of_internship', ''))
            skills_required = sanitize_input(request.form.get('skills_required', ''))
            location = sanitize_input(request.form.get('location', ''))
            years_of_experience = request.form.get('years_of_experience', '0')
            phone_number = sanitize_input(request.form.get('phone_number', ''))
            expected_salary = sanitize_input(request.form.get('expected_salary', ''))
            if not all([role, description, start_date, duration, type_of_internship, skills_required, location]):
                flash('Required fields missing.', 'danger')
                return render_template('register_internship.html', recruiter=recruiter)
            if not validate_phone(phone_number):
                flash('Invalid phone number format.', 'danger')
                return render_template('register_internship.html', recruiter=recruiter)
            try:
                years_of_experience = int(years_of_experience)
                company_name = recruiter['organization_name']
                company_mail = recruiter['email']
                max_internship = conn.execute('SELECT MAX(id) as max_id FROM internship_info').fetchone()
                new_internship_id = (max_internship['max_id'] + 1) if max_internship['max_id'] else 1
                conn.execute('''INSERT INTO internship_info (id, role, description_of_internship, start_date, end_date, duration, 
                               type_of_internship, skills_required, location, years_of_experience, phone_number, company_name, 
                               company_mail, user_id, posted_date, expected_salary)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                               (new_internship_id, role, description, start_date, end_date, duration, type_of_internship,
                               skills_required, location, years_of_experience, phone_number, company_name, company_mail, user_id,
                               datetime.now().strftime('%Y-%m-%d %H:%M:%S'), expected_salary))
                conn.commit()
                global resume_df, internship_df
                resume_df, internship_df = fetch_data()
                flash('Internship registered successfully!', 'success')
                return redirect(url_for('recruiter_dashboard'))
            except ValueError:
                flash('Invalid years of experience.', 'danger')
                return render_template('register_internship.html', recruiter=recruiter)
    return render_template('register_internship.html', recruiter=recruiter)

@app.route('/upload_resume', methods=['GET', 'POST'], strict_slashes=False)
@role_required('intern')
def upload_resume():
    if request.method == 'POST':
        file = request.files.get('resume')
        skills = sanitize_input(request.form.get('skills', ''))
        if not skills:
            flash('Skills are required.', 'danger')
            return render_template('upload_resume.html')
        if file and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            user_id = session['user_id']
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filename = f"{user_id}_resume.pdf"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            with get_db_connection() as conn:
                if not conn:
                    flash('Database error.', 'danger')
                    return render_template('upload_resume.html')
                existing_resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
                if existing_resume:
                    conn.execute('UPDATE resume_info SET resume_path = ?, skills = ? WHERE user_id = ?', (file_path, skills, user_id))
                else:
                    conn.execute('INSERT INTO resume_info (user_id, resume_path, skills) VALUES (?, ?, ?)', (user_id, file_path, skills))
                conn.execute('INSERT INTO user_progress (user_id, task_type, task_description, completion_date, points) VALUES (?, ?, ?, ?, ?)',
                             (user_id, 'Resume Upload', 'Uploaded resume', datetime.now().strftime('%Y-%m-%d'), 50))
                conn.commit()
                global resume_df, internship_df
                resume_df, internship_df = fetch_data()
                flash('Resume uploaded successfully!', 'success')
                return redirect(url_for('intern_dashboard'))
        flash('Allowed file types are PDF only.', 'danger')
    return render_template('upload_resume.html')

@app.route('/create_resume', methods=['GET', 'POST'], strict_slashes=False)
@role_required('intern')
def create_resume():
    if request.method == 'POST':
        user_id = session['user_id']
        name = sanitize_input(request.form.get('name', ''))
        email = sanitize_input(request.form.get('email', ''))
        phone = sanitize_input(request.form.get('phone', ''))
        skills = sanitize_input(request.form.get('skills', ''))
        experience = sanitize_input(request.form.get('experience', ''))
        education = sanitize_input(request.form.get('education', ''))
        certifications = sanitize_input(request.form.get('certifications', ''))
        achievements = sanitize_input(request.form.get('achievements', ''))
        if not all([name, email, phone, skills]):
            flash('Required fields missing.', 'danger')
            return render_template('create_resume.html')
        if not validate_email(email) or not validate_phone(phone):
            flash('Invalid email or phone format.', 'danger')
            return render_template('create_resume.html')
        with get_db_connection() as conn:
            if not conn:
                flash('Database error.', 'danger')
                return render_template('create_resume.html')
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
            conn.execute('INSERT INTO user_progress (user_id, task_type, task_description, completion_date, points) VALUES (?, ?, ?, ?, ?)',
                         (user_id, 'Resume Creation', 'Created ATS-friendly resume', datetime.now().strftime('%Y-%m-%d'), 100))
            conn.commit()
            global resume_df, internship_df
            resume_df, internship_df = fetch_data()
            flash('Resume created successfully!', 'success')
            return redirect(url_for('intern_dashboard'))
    return render_template('create_resume.html')

@app.route('/edit_resume', methods=['GET', 'POST'], strict_slashes=False)
@role_required('intern')
def edit_resume():
    user_id = session['user_id']
    with get_db_connection() as conn:
        if not conn:
            flash('Database error.', 'danger')
            return redirect(url_for('intern_login'))
        resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
        if not resume:
            flash('Please create your resume first!', 'warning')
            return redirect(url_for('create_resume'))
        if request.method == 'POST':
            name = sanitize_input(request.form.get('name', ''))
            email = sanitize_input(request.form.get('email', ''))
            phone = sanitize_input(request.form.get('phone', ''))
            skills = sanitize_input(request.form.get('skills', ''))
            experience = sanitize_input(request.form.get('experience', ''))
            education = sanitize_input(request.form.get('education', ''))
            certifications = sanitize_input(request.form.get('certifications', ''))
            achievements = sanitize_input(request.form.get('achievements', ''))
            if not all([name, email, phone, skills]):
                flash('Required fields missing.', 'danger')
                return render_template('edit_resume.html', resume=resume)
            if not validate_email(email) or not validate_phone(phone):
                flash('Invalid email or phone format.', 'danger')
                return render_template('edit_resume.html', resume=resume)
            conn.execute('''
                UPDATE resume_info SET name_of_applicant = ?, email = ?, phone_number = ?, skills = ?,
                       experience = ?, education = ?, certifications = ?, achievements = ?
                WHERE user_id = ?
            ''',
            (name, email, phone, skills, experience, education, certifications, achievements, user_id))
            conn.commit()
            global resume_df, internship_df
            resume_df, internship_df = fetch_data()
            flash('Resume updated successfully!', 'success')
            return redirect(url_for('intern_dashboard'))
    return render_template('edit_resume.html', resume=resume)

@app.route('/resume_enhance', methods=['GET', 'POST'], strict_slashes=False)
@role_required('intern')
def resume_enhance():
    user_id = session['user_id']
    with get_db_connection() as conn:
        if not conn:
            flash('Database error.', 'danger')
            return redirect(url_for('intern_login'))
        resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
        if not resume:
            flash('Please create your resume first!', 'warning')
            return redirect(url_for('create_resume'))
        if request.method == 'POST':
            job_description = sanitize_input(request.form.get('job_description', ''))
            if not job_description:
                flash('Job description required.', 'danger')
                return render_template('resume_enhance.html', resume=resume)
            resume_text = ' '.join([resume[field] for field in ['skills', 'experience', 'education', 'certifications', 'achievements'] if resume[field]])
            job_text = preprocess_text(job_description)
            resume_text = preprocess_text(resume_text)
            keywords = job_text.split()
            enhanced_skills = ', '.join(set(resume['skills'].split(', ') + keywords[:5]))
            enhanced_resume = f"Enhanced Skills: {enhanced_skills}\nExperience: {resume['experience']}\nEducation: {resume['education']}"
            conn.execute('UPDATE resume_info SET enhanced_resume = ? WHERE user_id = ?', (enhanced_resume, user_id))
            conn.execute('INSERT INTO user_progress (user_id, task_type, task_description, completion_date, points) VALUES (?, ?, ?, ?, ?)',
                         (user_id, 'Resume Enhancement', 'Enhanced resume with AI', datetime.now().strftime('%Y-%m-%d'), 75))
            conn.commit()
            global resume_df, internship_df
            resume_df, internship_df = fetch_data()
            flash('Resume enhanced successfully!', 'success')
            return redirect(url_for('intern_dashboard'))
    return render_template('resume_enhance.html', resume=resume)

@app.route('/skill_gap', strict_slashes=False)
@role_required('intern')
def skill_gap():
    user_id = session['user_id']
    with get_db_connection() as conn:
        if not conn:
            flash('Database error.', 'danger')
            return redirect(url_for('intern_login'))
        resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
        if not resume:
            flash('Please create your resume first!', 'warning')
            return redirect(url_for('create_resume'))
    user_skills = set(preprocess_skills(resume['skills']))
    skill_gaps = []
    course_recommendations = {
        'python': 'Coursera: Python for Everybody',
        'javascript': 'Udemy: JavaScript Bootcamp',
        'sql': 'Khan Academy: Intro to SQL'
    }
    for idx, internship in internship_df.iterrows():
        required_skills = set(internship['processed_Required_Skills'])
        gaps = required_skills - user_skills
        if gaps:
            skill_gaps.append({
                'role': internship['role'],
                'company': internship['company_name'],
                'missing_skills': list(gaps),
                'courses': [course_recommendations.get(skill, f'Search for {skill} courses') for skill in gaps]
            })
    return render_template('skill_gap.html', skill_gaps=skill_gaps)

@app.route('/voice_command', methods=['POST'], strict_slashes=False)
@role_required('intern')
def voice_command():
    command = sanitize_input(request.json.get('command', '').lower())
    if 'apply to' in command:
        match = re.search(r'internship (\d+)', command)
        if match:
            internship_id = int(match.group(1))
            with get_db_connection() as conn:
                if not conn:
                    return jsonify({'error': 'Database error.'}), 500
                internship = conn.execute('SELECT * FROM internship_info WHERE id = ?', (internship_id,)).fetchone()
                if internship:
                    existing_application = conn.execute('SELECT * FROM applications WHERE user_id = ? AND internship_id = ?', 
                                          (session['user_id'], internship_id)).fetchone()
                    if not existing_application:
                        conn.execute('INSERT INTO applications (user_id, internship_id, applied_at) VALUES (?, ?, ?)',
                                     (session['user_id'], internship_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                        conn.commit()
                        return jsonify({'message': f'Applied to internship {internship_id} successfully!'})
                    return jsonify({'message': 'You have already applied to this internship.'})
                return jsonify({'error': 'Internship not found.'})
    elif 'show resume' in command:
        return jsonify({'redirect': url_for('edit_resume')})
    return jsonify({'error': 'Command not recognized.'})

@app.route('/analytics/<int:internship_id>', strict_slashes=False)
@role_required('recruiter')
def analytics(internship_id):
    user_id = session['user_id']
    with get_db_connection() as conn:
        if not conn:
            flash('Database error.', 'danger')
            return redirect(url_for('recruiter_dashboard'))
        internship = conn.execute('SELECT * FROM internship_info WHERE id = ? AND user_id = ?', (internship_id, user_id)).fetchone()
        if not internship:
            flash('Internship not found.', 'danger')
            return redirect(url_for('recruiter_dashboard'))
        applications = conn.execute('SELECT * FROM applications WHERE internship_id = ?', (internship_id,)).fetchall()
        applicant_ids = [app['user_id'] for app in applications]
        resumes = conn.execute('SELECT * FROM resume_info WHERE user_id IN ({})'.format(','.join(['?']*len(applicant_ids))), applicant_ids).fetchall()
        ratings = conn.execute('SELECT * FROM internship_ratings WHERE internship_id = ?', (internship_id,)).fetchall()
    avg_rating = np.mean([r['rating'] for r in ratings]) if ratings else 0
    skill_counts = {}
    for resume in resumes:
        skills = preprocess_skills(resume['skills'])
        for skill in skills:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
    top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    application_trend = [
        {'date': date, 'count': count}
        for date, count in Counter(app['applied_at'][:10] for app in applications).items()
    ]
    internship_skills = preprocess_skills(internship['skills_required'])
    applicant_data = []
    for resume in resumes:
        user = conn.execute('SELECT * FROM users WHERE user_id = ?', (resume['user_id'],)).fetchone()
        if user:
            similarity = jaccard_similarity(preprocess_skills(resume['skills']), internship_skills)
            acceptance_prob = similarity * 100
            applicant_data.append({
                'name': resume['name_of_applicant'],
                'email': user['email'],
                'skills': resume['skills'],
                'acceptance_prob': round(acceptance_prob, 1)
            })
    analytics_data = {
        'total_applicants': len(applications),
        'avg_rating': round(avg_rating, 2),
        'top_skills': top_skills,
        'application_trend': application_trend,
        'applicants': applicant_data
    }
    return render_template('analytics.html', internship=analytics_data, analytics_data=analytics_data, internship_id=internship_id)

@app.route('/progress', strict_slashes=False)
@role_required('intern')
def progress():
    user_id = session['user_id']
    with get_db_connection() as conn:
        if not conn:
            flash('Database error.', 'danger')
            return redirect(url_for('intern_login'))
        progress = conn.execute('SELECT * FROM user_progress WHERE user_id = ?', (user_id,)).fetchall()
    total_points = sum(p['points'] for p in progress) if progress else 0
    level = total_points // 100
    achievements = [
        {'name': 'Resume Master', 'unlocked': total_points >= 100},
        {'name': 'Application Pro', 'unlocked': total_points >= 150}
    ]
    logging.info(f"Progress for user_id={user_id}: {len(progress)} records, total_points={total_points}")
    return render_template('progress.html', progress=progress, total_points=total_points, level=level, achievements=achievements)

@app.route('/interview_prep', strict_slashes=False)
@role_required('intern')
def interview_prep():
    questions = [
        {'id': 1, 'question': 'Tell me about yourself.', 'category': 'General'},
        {'id': 2, 'question': 'What are your strengths?', 'category': 'General'},
        {'id': 3, 'question': 'Explain a Python decorator.', 'category': 'Technical'}
    ]
    return render_template('interview_prep.html', questions=questions', user_name=session['user_name'])

@app.route('/peer_review', methods=['GET', 'POST'], strict_slashes=False)
@role_required('intern')
def peer_review():
    user_id = session['user_id']
    with get_db_connection() as conn:
        if not conn:
            flash('Database error.', 'danger')
            return redirect(url_for('intern_login'))
        resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
        if not resume:
            flash('Please create your resume first!', 'warning')
            return redirect(url_for('create_resume'))
        if request.method == 'POST':
            reviewed_user_id = request.form.get('reviewed_user_id', '')
            rating = request.form.get('rating', '')
            feedback = sanitize_input(request.form.get('feedback', ''))
            try:
                reviewed_user_id = int(reviewed_user_id)
                rating = int(rating)
                if not (1 <= rating <= 5):
                    flash('Rating must be between 1 and 5.', 'danger')
                    return render_template('peer_review.html', resume=resume, reviews=reviews, other_resumes=other_resumes)
                conn.execute('INSERT INTO peer_reviews (reviewer_id, reviewed_user_id, resume_id, rating, feedback, review_date) VALUES (?, ?, ?, ?, ?, ?)',
                             (user_id, reviewed_user_id, reviewed_user_id, rating, feedback, datetime.now().strftime('%Y-%m-%d')))
                conn.execute('INSERT INTO user_progress (user_id, task_type, task_description, completion_date, points) VALUES (?, ?, ?, ?, ?)',
                             (user_id, 'Peer Review', 'Provided resume feedback', datetime.now().strftime('%Y-%m-%d'), 25))
                conn.commit()
                flash('Review submitted successfully!', 'success')
            except ValueError:
                flash('Invalid input data.', 'danger')
        reviews = conn.execute('SELECT * FROM peer_reviews WHERE reviewed_user_id = ?', (user_id,)).fetchall()
        other_resumes = conn.execute('SELECT * FROM resume_info WHERE user_id != ?', (user_id,)).fetchall()
    return render_template('peer_review.html', resume=resume, reviews=reviews, other_resumes=other_resumes)

@app.route('/match', strict_slashes=False)
@role_required('intern')
def match():
    user_id = session['user_id']
    global resume_df, internship_df
    resume_df, internship_df = fetch_data()
    with get_db_connection() as conn:
        if not conn:
            flash('Database error.', 'danger')
            return redirect(url_for('intern_login'))
        resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
        if not resume:
            flash('Please create your resume!', 'warning')
            return redirect(url_for('create_resume'))
    user_skills = preprocess_skills(resume['skills'])
    matched_internships = []
    for idx, internship in internship_df.iterrows():
        similarity = jaccard_similarity(user_skills, internship['processed_Required_Skills'])
        if similarity > 0:
            matched_internships.append({
                'id': internship['id'],
                'role': internship['role'],
                'company_name': internship['company_name'],
                'description': internship['description_of_internship'],
                'duration': internship['duration'],
                'type_of_internship': internship['type_of_internship'],
                'skills_required': internship['skills_required'],
                'location': internship['location'],
                'similarity_score': round(similarity * 100, 2)
            })
    matched_internships = sorted(matched_internships, key=lambda x: x['similarity_score'], reverse=True)
    return render_template('match.html', matched_internships=matched_internships)

@app.route('/top_matched_applicants/<int:internship_id>', strict_slashes=False)
@role_required('recruiter')
def top_matched_applicants(internship_id):
    user_id = session['user_id']
    global resume_df, internship_df
    resume_df, internship_df = fetch_data()
    with get_db_connection() as conn:
        if not conn:
            flash('Database error.', 'danger')
            return redirect(url_for('recruiter_dashboard'))
        internship = conn.execute('SELECT * FROM internship_info WHERE id = ? AND user_id = ?', (internship_id, user_id)).fetchone()
        if not internship:
            flash('Internship not found.', 'danger')
            return render_template('top_matched_applicants.html', matched_applicants=[], internship_title='Unknown')
    internship_title = internship['role']
    internship_skills = preprocess_skills(internship['skills_required'])
    matched_applicants = []
    for idx, resume in resume_df.iterrows():
        similarity = jaccard_similarity(resume['processed_Skills'], internship_skills)
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
    matched_applicants = sorted(matched_applicants, key=lambda x: x['similarity_score'], reverse=True)[:5]
    return render_template('top_matched_applicants.html', matched_applicants=matched_applicants, internship_title=internship_title, user_name=session['user_name'])

@app.route('/download_resume/<path:resume_path>', strict_slashes=False)
@role_required('recruiter')
def download_resume(resume_path):
    try:
        resume_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], resume_path))
        if not resume_path.startswith(app.config['UPLOAD_FOLDER']):
            flash('Invalid file path.', 'danger')
            return redirect(url_for('recruiter_dashboard'))
        return send_file(resume_path, as_attachment=True)
    except FileNotFoundError:
        flash('Resume file not found.', 'danger')
        return redirect(url_for('recruiter_dashboard'))
    except Exception as e:
        logging.error(f"Download resume error: {str(e)}")
        flash('Error downloading resume.', 'danger')
        return redirect(url_for('recruiter_dashboard'))

@app.route('/apply_internship/<int:internship_id>', methods=['POST'], strict_slashes=False)
@role_required('intern')
def apply_internship(internship_id):
    user_id = session['user_id']
    with get_db_connection() as conn:
        if not conn:
            flash('Database error.', 'danger')
            return redirect(url_for('intern_dashboard'))
        internship = conn.execute('SELECT * FROM internship_info WHERE id = ?', (internship_id,)).fetchone()
        if not internship:
            flash('Internship not found.', 'danger')
            return redirect(url_for('intern_dashboard'))
        existing_application = conn.execute('SELECT * FROM applications WHERE user_id = ? AND internship_id = ?', (user_id, internship_id)).fetchone()
        if existing_application:
            flash('You have already applied to this internship!','!, 'warning')
            return redirect(url_for('intern_dashboard'))
        conn.execute('INSERT INTO applications (user_id, internship_id, applied_at) VALUES (?, ?, ?)', 
                     (user_id, internship_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.execute('INSERT INTO user_progress (user_id, task_type, task_description, completion_date, points) VALUES (?, ?, ?, ?, ?)',
                     (user_id, 'Application', f'Applied to {internship["role"]}'), datetime.now().strftime('%Y-%m-%d'), 50))
        conn.commit()
        flash('Applied successfully!', 'success')
    return redirect(url_for('intern_dashboard'))

@app.route('/applied_internships', strict_slashes=False)
@role_required('intern')
def applied_internships():
    user_id = session['user_id']
    with get_db_connection() as conn:
        if not conn:
            flash('Database error.', 'danger')
            return redirect(url_for('intern_login'))
        query = '''
        SELECT i.*, a.applied_at
        FROM applications a
        JOIN internship_info i ON a.internship_id = i.id
        WHERE a.user_id = ?
        '''
        applied_internships = conn.execute(query, (user_id,)).fetchall()
    applied_internships_list = [
        {
            'role': internship['role'],
            'company_name': internship['company_name'],
            'type_of_internship': internship['type_of_internship'],
            'location': internship['location'],
            'description': internship['description_of_internship'],
            'applied_at': internship['applied_at']
        ]
    for internship in applied_internships
    ]
    return render_template('applied_internships.html', applied_internships=applied_internships_list, user_name=session['user_name'])

@app.route('/applied_applicants', methods=['GET'], strict_slashes=False)
@role_required('recruiter')
def applied_applicants():
    user_id = session['user_id']
    with get_db_connection() as conn:
        if not conn:
            flash('Database error.', 'danger')
            return redirect(url_for('recruiter_login'))
        query = '''
        SELECT r.name_of_applicant, u.email, i.role, a.applied_at, r.resume_path
        FROM applications a
        JOIN resume_info r ON a.user_id = r.user_id
        JOIN users u ON a.user_id = u.user_id
        JOIN internship_info i ON a.internship_id = i.id
        WHERE i.user_id = ?
        '''
        results = conn.execute(query, (user_id,)).fetchall()
    applicants = [
        {
            'name': applicant['name_of_applicant'],
            'email': applicant['email'],
            'internship_title': applicant['role'],
            'applied_at': applicant['applied_at'],
            'resume_path': applicant['resume_path'] or ''
        ]
    '''
    for applicant in results
    return render_template('applied_applicants.html', applicants=applicants, internship=None, user_name=session['user_name'])

@app.route('/applied_applicants/<int:internship_id>', strict_slashes=False)
@role_required('recruiter')
def applied_applicants_specific(internship_id):
    user_id = session['user_id']
    with get_db_connection() as conn:
        if not conn:
            flash('Database error.', 'danger')
            return redirect(url_for('recruiter_login'))
        internship = conn.execute('SELECT * FROM internship_info WHERE id = ? AND user_id = ?', 
                         (internship_id, user_id)).fetchone()
        if not internship:
            flash('Internship not found.', 'danger')
            return render_template('applied_applicants.html', applicants=None, internship=None, user_name=session['user_name'])
        query = '''
        SELECT r.name_of_applicant, u.email, r.skills, r.experience, r.education, r.resume_path, a.applied_at
        FROM applications a
        JOIN resume_info r ON a.user_id = r.user_id
        JOIN users u ON a.user_id = u.user_id
        WHERE a.internship_id = ?
        '''
        applicants = conn.execute(query, (internship_id,)).fetchall()
    applicants_list = [
        {
            'name': applicant['name_of_applicant'],
            'email': applicant['email'],
            'skills': applicant['skills'],
            'experience': applicant['experience'],
            'education': applicant['education'],
            'resume_path': applicant['resume_path'] or '',
            'applied_at': applicant['applied_at']
        ]
    for applicant in applicants
    return render_template('applied_applicants.html', applicants=applicants_list, internship=internship, user_name=session['user_name'])

@app.route('/edit_profile', methods=['GET', 'POST'], strict_slashes=False)
def edit_profile():
    if 'user_id' in session:
        flash('Please login!', 'danger')
        return redirect(url_for('intern_login')) if session.get('role') == 'intern' else 'recruiter_login'))
    user_id = session['user_id']
    with get_db_connection() as conn:
        if not conn:
            flash('Database error.', 'danger')
            return render_template('edit_profile.html', user=None)
        user = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
        if request.method == 'POST':
            name = sanitize_input(request.form.get('name', ''))
            email = sanitize_input(request.form.get('email', ''))
            password = request.form.get('password', '')
            if not all([name, email]):
                flash('Name and email are required!', 'danger')
                return render_template('edit_profile.html', user=user)
            if not validate_email(email):
                flash('Invalid email format!', 'danger')
                return render_template('edit_profile.html', user=user)
            existing_user = conn.execute('SELECT * FROM users WHERE email = ? AND user_id != ?', 
                                 (email, user_id)).fetchone()
            if existing_user:
                flash('Email is already in use!', 'danger')
                return render_template('edit_profile.html', user=user)
            if password and len(password) >= 8:
                hashed_password = generate_password_hash(password_hash(password))
                conn.execute('UPDATE users SET name = ?, email = ?, password = ? WHERE user_id = ?', 
                             (name, email, hashed_password, user_id))
            else:
                conn.execute('UPDATE users SET name = ?, email = ? WHERE user_id = ?', 
                             (name, email, user_id))
            conn.commit()
            session['user_id_name'] = name
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('intern_for_dashboard' if session['role'] == 'intern' else 'recruiter_dashboard'))
    return render_template('edit_profile.html', user=user)

@app.route('/edit_organization_profile', methods=['GET', 'POST'], strict_slashes=False)
@role_required('recruiter')
def edit_organization_profile():
    user_id = session['user_id']
    with get_db_connection() as conn:
        if not conn:
            flash('Database error.', 'danger')
            return redirect(url_for('recruiter_login'))
        user = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone())
        if request.method == 'POST':
            organization_name = sanitize_input(request.form.get('organization_name', ''))
            contact_details = sanitize_input(request.form.get('contact_details', ''))
            location = sanitize_input(request.form.get('location', ''))
            website_link = sanitize_input(request.form.get('website_link', ''))
            if not organization_name:
                flash('Organization name is required!', 'danger')
                return render_template('edit_organization_profile.html', recruiter=user)
            if contact_details and not validate_phone(contact_details):
                flash('Invalid contact details format.', 'danger')
                return render_template('edit_organization_profile.html', recruiter=user)
            conn.execute('UPDATE users SET organization_name = ?, contact_details = ?, location = ?, website_link = ? WHERE user_id = ?',
                         (organization_name, contact_details or None, location or None, website_link or None, user_id))
            conn.commit()
            flash('Organization profile updated successfully!', 'success')
            return redirect(url_for('recruiter_dashboard'))
    return render_template('edit_organization_profile.html', recruiter=user)

@app.route('/logout', strict_slashes=False)
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

@app.route('/verify_credential', methods=['GET', 'POST'], strict_slashes=False)
def verify_credential():
    if 'user_id' not in session:
        flash('Please login.', 'danger')
        return redirect(url_for('index'))
    if request.method == 'POST':
        credential_hash_input = sanitize_input(request.form.get('credential_hash', '').strip())
        signature_input = sanitize_input(request.form.get('signature', '').strip())
        if not all([credential_hash_input, signature_input]):
            flash('All fields are required.', 'danger')
            return render_template('verify_credential.html')
        try:
            with get_db_connection() as conn:
                if not conn:
                    flash('Database error.', 'danger')
                    return render_template('verify_credential.html')
                credentials = conn.execute('SELECT * FROM credentials').fetchall()
                matched_credential = None
                for cred in credentials:
                    full_hash = cred['credential_hash']
                    full_signature = cred['signature']
                    truncated_hash = f"{full_hash[:10]}...{full_hash[-10:]}" if len(full_hash) > 20 else full_hash
                    truncated_signature = f"{full_signature[:10]}...{full_signature[-10:]}" if len(full_signature) > 20 else full_signature
                    if (credential_hash_input in (full_hash, truncated_hash)) and (signature_input in (full_signature, truncated_signature)):
                        matched_credential = cred
                        break
                if not matched_credential:
                    flash('Credential not found in database.', 'danger')
                    return render_template('verify_credential.html')
                public_key = serialization.load_pem_public_key(PUBLIC_KEY.encode())
                signature = bytes.fromhex(matched_credential['signature'])
                public_key.verify(
                    signature,
                    matched_credential['credential_hash'].encode(),
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                user = conn.execute('SELECT name FROM users WHERE user_id = ?', (matched_credential['user_id'],)).fetchone()
                session['verified_credential'] = {
                    'user_id': matched_credential['user_id'],
                    'user_name': user['name'] if user else 'Unknown',
                    'credential_hash': matched_credential['credential_hash'],
                    'signature': matched_credential['signature'],
                    'issued_date': matched_credential['issued_date']
                }
                flash('Credential verified successfully!', 'success')
        except InvalidSignature:
            flash('Invalid signature.', 'danger')
        except Exception as e:
            logging.error(f"Verification error: {str(e)}")
            flash('Error verifying credential.', 'danger')
    return render_template('verify_credential.html')

@app.route('/issue_credential', methods=['GET', 'POST'], strict_slashes=False)
@role_required('admin')
def issue_credential():
    if not app.secret_key:
        logging.error('FLASK_SECRET_KEY not set')
        flash('Server configuration error.', 'danger')
        return render_template('issue_credential')
    logging.info(f"Accessing issue_credential, session: {session.get('user_id', 'None')}, role: {session.get('role', 'None')}")
    with get_db_connection() as conn:
        if not conn:
            flash('Database error.', 'danger')
            return render_template('issue_credential.html')
        interns = conn.execute('SELECT user_id, name FROM users WHERE role = ?', ('intern',)).fetchall()
        interns_list = [{'user_id': intern['user_id'], 'name': intern['name']} for intern in interns]
    if request.method == 'POST':
        user_id = sanitize_input(request.form.get('user_id', ''))
        credential_details = sanitize_input(request.form.get('credential_details', '').strip())
        logging.info(f'POST request: user_id={user_id}, credential_details={credential_details}')
        if not all([user_id, credential_details]):
            flash('All fields are required.', 'danger')
            return render_template('issue_credential.html', interns=interns_list)
        try:
            user_id = int(user_id)
            with get_db_connection() as conn:
                if not conn:
                    flash('Database error.', 'danger')
                    return render_template('issue_credential.html', interns=interns_list)
                user = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
                if not user:
                    flash('Invalid user ID.', 'danger')
                    return render_template('issue_credential.html', interns=interns_list)
                credential_hash = hashlib.sha256(credential_details.encode()).hexdigest()
                private_key_pem = os.getenv('PRIVATE_KEY_PEM')
                if not private_key_pem:
                    flash('Private key not configured.', 'danger')
                    return render_template('issue_credential.html', interns=interns_list)
                private_key = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
                signature = private_key.sign(
                    credential_hash.encode(),
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                signature_hex = signature.hex()
                conn.execute('INSERT INTO credentials (user_id, credential_hash, signature, issued_date) VALUES (?, ?, ?, ?)',
                             (user_id, credential_hash, signature_hex, datetime.now().strftime('%Y-%m-%d')))
                conn.commit()
                session['issued_credential'] = {
                    'user_id': user_id,
                    'credential_details': credential_details,
                    'credential_hash': credential_hash,
                    'signature': signature_hex,
                    'issued_date': datetime.now().strftime('%Y-%m-%d')
                )
                logging.info(f'Credential issued: user_id={user_id}, hash={credential_hash}')
                flash(f'Credential issued: Hash={credential_hash[:10]}...{credential_hash[-10:]}, Signature={signature_hex[:10]}...{signature_hex[-10:]}', 'success')
            except ValueError:
                flash('User ID must be a number.', 'danger')
                return render_template('issue_credential.html', interns=interns_list)
            except Exception as e:
                logging.error(f"Issue credential error: {str(e)}")
                flash('Error issuing credential.', 'danger')
                return render_template('issue_credential.html', interns=interns_list)
    else:
        session.pop('issued_credential', None)
        logging.info('GET request: Cleared issued_credential session')
    return render_template('issue_credential.html', interns=interns_list)

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    error_message = f"Exception: {str(e)}\n{traceback.format_exc()}"
    logging.error(error_message)
    return render_template('500.html', error_message=str(e)), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))