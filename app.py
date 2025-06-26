from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, jsonify
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
from datetime import datetime, timedelta
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
from transformers import pipeline
from rake_nltk import Rake
import requests
from markupsafe import Markup
import pdfplumber

# Set Hugging Face cache directory to a writable location
os.environ['TRANSFORMERS_CACHE'] = os.getenv('TRANSFORMERS_CACHE', '/tmp/hf_cache')

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
logging.info(f"Flask secret key configured: {len(app.secret_key)} bytes")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    ALTER TABLE resume_info ADD COLUMN soft_skills TEXT;
CREATE TABLE IF NOT EXISTS interview_feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    question TEXT,
    response TEXT,
    feedback TEXT,
    date TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
CREATE TABLE IF NOT EXISTS mentors (
    mentor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    industry TEXT,
    skills TEXT,
    availability TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
CREATE TABLE IF NOT EXISTS mentorship_requests (
    request_id INTEGER PRIMARY KEY AUTOINCREMENT,
    intern_id INTEGER,
    mentor_id INTEGER,
    message TEXT,
    status TEXT DEFAULT 'pending',
    request_date TEXT,
    FOREIGN KEY (intern_id) REFERENCES users(user_id),
    FOREIGN KEY (mentor_id) REFERENCES mentors(mentor_id)
);
    '''
    conn = get_db_connection()
    if conn:
        conn.executescript(schema)
        conn.commit()
        conn.close()
        logging.info("Database schema initialized")

def insert_test_data():
    conn = get_db_connection()
    if not conn:
        logging.error("Failed to get database connection for test data insertion")
        return
    existing_user = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    if existing_user > 0:
        logging.info("Test data already exists")
        conn.close()
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
    (1, 'Alice Smith', 'alice.smith@example.com', '+1-555-123-4567', 'python, java, sql, tensorflow', 'Software Intern at XYZ Corp (6 months)', 'B.S. Computer Science, MIT, 2024', 'AWS Certified Developer', 'Won Hackathon 2023', 'static/uploads/1_resume.pdf', 0),
    (2, 'Bob Johnson', 'bob.johnson@example.com', '+1-555-234-5678', 'javascript, react, node.js', 'Frontend Developer at ABC Inc (1 year)', 'B.S. Software Engineering, Stanford, 2023', 'React Professional Certification', 'Published NPM package', 'static/uploads/2_resume.pdf', 0),
    (3, 'Carol Lee', 'carol.lee@example.com', '+1-555-345-6789', 'python, django, postgresql', 'Backend Intern at DEF Ltd (4 months)', 'B.S. Computer Science, UC Berkeley, 2024', 'Django Developer Certification', 'Top 10 in CodeJam 2024', 'static/uploads/3_resume.pdf', 0),
    (4, 'David Brown', 'david.brown@example.com', '+1-555-456-7890', 'c++, opencv, machine learning', 'Research Assistant at GHI University (8 months)', 'M.S. AI, Caltech, 2025', 'TensorFlow Developer Certificate', 'Published CVPR 2024 paper', 'static/uploads/4_resume.pdf', 0);
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
    conn.close()

if not os.path.exists(DB_PATH):
    initialize_database()
    insert_test_data()
else:
    conn = get_db_connection()
    if conn:
        user_count = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
        conn.close()
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
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and t.isalnum()]
    return ' '.join(tokens)

# Fetch data
def fetch_data():
    global resume_df, internship_df, skill_to_index
    try:
        conn = get_db_connection()
        if not conn:
            raise Exception("Failed to connect to database")
        resumes = conn.execute('SELECT * FROM resume_info').fetchall()
        internships = conn.execute('SELECT * FROM internship_info').fetchall()
        resume_df = pd.DataFrame([dict(row) for row in resumes])
        internship_df = pd.DataFrame([dict(row) for row in internships])
        conn.close()
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

# Helper: Extract skills from resume PDF
SKILL_KEYWORDS = [
    'python', 'java', 'c++', 'javascript', 'sql', 'html', 'css', 'react', 'node', 'django', 'flask', 'machine learning',
    'deep learning', 'tensorflow', 'pytorch', 'nlp', 'data analysis', 'excel', 'powerpoint', 'communication', 'leadership',
    'teamwork', 'problem solving', 'project management', 'cloud', 'aws', 'azure', 'git', 'linux', 'docker', 'kubernetes',
    'postgresql', 'mongodb', 'opencv', 'pandas', 'numpy', 'scikit-learn', 'r', 'matlab', 'public speaking', 'creativity'
]
def extract_skills_from_text(text):
    text = text.lower()
    found = set()
    for skill in SKILL_KEYWORDS:
        if re.search(r'\b' + re.escape(skill) + r'\b', text):
            found.add(skill)
    return ', '.join(sorted(found))

def extract_skills_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text() + '\n'
        return extract_skills_from_text(text)
    except Exception as e:
        return ''

# Routes
@app.route('/', strict_slashes=False)
def index():
    return render_template('index.html')

@app.route('/admin_login', methods=['GET', 'POST'], strict_slashes=False)
def admin_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        if not conn:
            flash('Database error.', 'danger')
            return render_template('admin_login.html')
        user = conn.execute('SELECT * FROM users WHERE email = ? AND role = ?', (email, 'admin')).fetchone()
        conn.close()
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
def recruiter_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        if not conn:
            flash('Database error.', 'danger')
            return render_template('recruiter_login.html')
        user = conn.execute('SELECT * FROM users WHERE email = ? AND role = ?', (email, 'recruiter')).fetchone()
        conn.close()
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
def intern_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        if not conn:
            flash('Database error.', 'danger')
            return render_template('intern_login.html')
        user = conn.execute('SELECT * FROM users WHERE email = ? AND role = ?', (email, 'intern')).fetchone()
        conn.close()
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
def recruiter_signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        company = request.form['company']
        conn = get_db_connection()
        if not conn:
            flash('Database error.', 'danger')
            return render_template('recruiter_signup.html')
        existing = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if existing:
            conn.close()
            flash('Email already exists.', 'danger')
        else:
            hashed_password = generate_password_hash(password)
            max_user = conn.execute('SELECT MAX(user_id) as max_id FROM users').fetchone()
            new_user_id = (max_user['max_id'] + 1) if max_user['max_id'] else 1
            conn.execute('INSERT INTO users (user_id, name, email, password, role, organization_name) VALUES (?, ?, ?, ?, ?, ?)', 
                         (new_user_id, name, email, hashed_password, 'recruiter', company))
            conn.commit()
            conn.close()
            flash('Signup successful! Please login.', 'success')
            return redirect(url_for('recruiter_login'))
    return render_template('recruiter_signup.html')

@app.route('/intern_signup', methods=['GET', 'POST'], strict_slashes=False)
def intern_signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        skills = request.form['skills']
        education = request.form.get('education', '')
        certifications = request.form.get('certifications', '')
        conn = get_db_connection()
        if not conn:
            flash('Database error.', 'danger')
            return render_template('intern_signup.html')
        existing = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if existing:
            conn.close()
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
            conn.close()
            flash('Signup successful! Please login.', 'success')
            return redirect(url_for('intern_login'))
    return render_template('intern_signup.html')

@app.route('/admin_signup', methods=['GET', 'POST'], strict_slashes=False)
def admin_signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        secret_code = request.form['secret_code']
        expected_secret_code = os.getenv('ADMIN_SECRET_CODE', 'default-secret-code')
        if secret_code != expected_secret_code:
            flash('Invalid secret code.', 'danger')
            return render_template('admin_signup.html')
        try:
            conn = get_db_connection()
            if not conn:
                flash('Database error.', 'danger')
                return render_template('admin_signup.html')
            conn.execute('INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)',
                         (name, email, password, 'admin'))
            conn.commit()
            flash('Admin sign up successful! Please login.', 'success')
            return redirect(url_for('admin_login'))
        except sqlite3.IntegrityError:
            flash('Email already registered.', 'danger')
        except Exception as e:
            logging.error(f"Admin signup error: {str(e)}")
            flash('Error during signup.', 'danger')
        finally:
            if conn:
                conn.close()
    return render_template('admin_signup.html')

@app.route('/recruiter_dashboard', strict_slashes=False)
@role_required('recruiter')
def recruiter_dashboard():
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('recruiter_login'))
    recruiter = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
    internships = conn.execute('SELECT * FROM internship_info WHERE user_id = ?', (user_id,)).fetchall()
    conn.close()
    return render_template('recruiter_dashboard.html', recruiter=recruiter, internships=internships, user_name=session['user_name'])

@app.route('/intern_dashboard', strict_slashes=False)
@role_required('intern')
def intern_dashboard():
    user_id = session['user_id']
    global resume_df, internship_df
    resume_df, internship_df = fetch_data()
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('intern_login'))
    intern = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
    resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
    applications = conn.execute('SELECT * FROM applications WHERE user_id = ?', (user_id,)).fetchall()
    progress = conn.execute('SELECT * FROM user_progress WHERE user_id = ?', (user_id,)).fetchall()
    conn.close()
    if not resume:
        flash('Please create your resume!', 'warning')
        return redirect(url_for('create_resume'))
    applied_internship_ids = [app['internship_id'] for app in applications]
    user_skills = preprocess_skills(resume['skills'])
    internships = []
    one_month_ago = datetime.now() - timedelta(days=30)
    for idx, internship in internship_df.iterrows():
        posted_date = internship.get('posted_date')
        if posted_date:
            try:
                posted_dt = datetime.strptime(posted_date[:10], '%Y-%m-%d')
                if posted_dt < one_month_ago:
                    continue
            except Exception:
                continue
        similarity = jaccard_similarity(user_skills, internship['processed_Required_Skills'])
        if similarity > 0:
            internships.append({
                **internship,
                'similarity_score': int(similarity * 100)
            })
    internships = sorted(internships, key=lambda x: x['similarity_score'], reverse=True)
    total_points = sum(p['points'] for p in progress) if progress else 0
    level = total_points // 100
    # Check if resume is uploaded for apply button logic
    resume_uploaded = bool(resume and resume.get('resume_path'))
    return render_template('intern_dashboard.html', user_name=session['user_name'], internships=internships, applied_internship_ids=applied_internship_ids, total_points=total_points, level=level, resume_uploaded=resume_uploaded)

@app.route('/register_internship', methods=['GET', 'POST'], strict_slashes=False)
@role_required('recruiter')
def register_internship():
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('recruiter_login'))
    recruiter = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
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
        company_name = recruiter['organization_name']
        company_mail = recruiter['email']
        expected_salary = request.form.get('expected_salary', '')
        max_internship = conn.execute('SELECT MAX(id) as max_id FROM internship_info').fetchone()
        new_internship_id = (max_internship['max_id'] + 1) if max_internship['max_id'] else 1
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
@role_required('intern')
def upload_resume():
    if request.method == 'POST':
        file = request.files['resume']
        skills = request.form.get('skills')
        if file and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            filename = f"{session['user_id']}_resume.pdf"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Extract skills from PDF
            extracted_skills = extract_skills_from_pdf(file_path)
            # Use extracted skills if found, else fallback to user input
            final_skills = extracted_skills if extracted_skills else skills
            # Update resume_info and users table
            conn = get_db_connection()
            if conn:
                conn.execute('UPDATE resume_info SET resume_path = ?, skills = ? WHERE user_id = ?', (file_path, final_skills, session['user_id']))
                conn.execute('UPDATE users SET skills = ? WHERE user_id = ?', (final_skills, session['user_id']))
                conn.commit()
                conn.close()
            flash('Resume uploaded and skills updated successfully!', 'success')
            return redirect(url_for('intern_dashboard'))
        flash('Allowed file types are PDF only.', 'danger')
    return render_template('upload_resume.html')

@app.route('/create_resume', methods=['GET', 'POST'], strict_slashes=False)
@role_required('intern')
def create_resume():
    user_id = session['user_id']
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone() if conn else None
    ats_resume = None
    ats_resume_path = None
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        skills = request.form['skills']
        experience = request.form.get('experience')
        education = request.form.get('education')
        certifications = request.form.get('certifications')
        achievements = request.form.get('achievements')
        # Build ATS resume text
        ats_resume = f"""Name: {name}\nEmail: {email}\nPhone: {phone}\nSkills: {skills}\nExperience: {experience}\nEducation: {education}\nCertifications: {certifications}\nAchievements: {achievements}"""
        # Save ATS resume as .txt for download
        ats_resume_path = f"static/uploads/{user_id}_ats_resume.txt"
        with open(ats_resume_path, 'w', encoding='utf-8') as f:
            f.write(ats_resume)
        # Update resume_info and users table
        if conn:
            existing_resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
            if existing_resume:
                conn.execute('''
                    UPDATE resume_info SET name_of_applicant=?, email=?, phone_number=?, skills=?, experience=?, education=?, certifications=?, achievements=?, resume_path=? WHERE user_id=?
                ''', (name, email, phone, skills, experience, education, certifications, achievements, ats_resume_path, user_id))
            else:
                conn.execute('''
                    INSERT INTO resume_info (user_id, name_of_applicant, email, phone_number, skills, experience, education, certifications, achievements, resume_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, name, email, phone, skills, experience, education, certifications, achievements, ats_resume_path))
            # Update user skills in users table
            conn.execute('UPDATE users SET skills = ? WHERE user_id = ?', (skills, user_id))
            conn.commit()
            conn.close()
        flash('ATS resume created and skills updated successfully!', 'success')
        return render_template('create_resume.html', user=user, ats_resume=ats_resume, ats_resume_path=ats_resume_path)
    return render_template('create_resume.html', user=user, ats_resume=ats_resume, ats_resume_path=ats_resume_path)

@app.route('/edit_resume', methods=['GET', 'POST'], strict_slashes=False)
@role_required('intern')
def edit_resume():
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('intern_login'))
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
        # Update user skills in users table
        conn.execute('UPDATE users SET skills = ? WHERE user_id = ?', (skills, user_id))
        conn.commit()
        conn.close()
        global resume_df, internship_df
        resume_df, internship_df = fetch_data()
        flash('Resume updated successfully!', 'success')
        return redirect(url_for('intern_dashboard'))
    conn.close()
    return render_template('edit_resume.html', resume=resume)

@app.route('/resume_enhance', methods=['GET', 'POST'], strict_slashes=False)
@role_required('intern')
def resume_enhance():
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('intern_login'))
    resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
    if not resume:
        conn.close()
        flash('Please create your resume first!', 'warning')
        return redirect(url_for('create_resume'))
    if request.method == 'POST':
        job_description = request.form['job_description']
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
        conn.close()
        global resume_df, internship_df
        resume_df, internship_df = fetch_data()
        flash('Resume enhanced successfully!', 'success')
        return redirect(url_for('intern_dashboard'))
    conn.close()
    return render_template('resume_enhance.html', resume=resume)

# Helper: Get real-world course recommendations for missing skills
# Uses Coursera and Udemy APIs (or public endpoints)
def get_course_links(skill):
    courses = []
    # Coursera API (public search)
    try:
        resp = requests.get(f'https://www.coursera.org/search?query={skill}')
        if resp.status_code == 200:
            courses.append(f'<a href="https://www.coursera.org/search?query={skill}" target="_blank">Coursera: {skill.title()}</a>')
    except Exception:
        pass
    # Udemy API (public search)
    try:
        resp = requests.get(f'https://www.udemy.com/courses/search/?q={skill}')
        if resp.status_code == 200:
            courses.append(f'<a href="https://www.udemy.com/courses/search/?q={skill}" target="_blank">Udemy: {skill.title()}</a>')
    except Exception:
        pass
    # Add more providers as needed
    return courses if courses else [f'No course found for {skill}']

@app.route('/skill_gap', strict_slashes=False)
@role_required('intern')
def skill_gap():
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('intern_login'))
    resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
    if not resume:
        conn.close()
        flash('Please create your resume first!', 'warning')
        return redirect(url_for('create_resume'))
    user_skills = set(preprocess_skills(resume['skills']))
    skill_gaps = []
    one_month_ago = datetime.now() - timedelta(days=30)
    for idx, internship in internship_df.iterrows():
        posted_date = internship.get('posted_date')
        if posted_date:
            try:
                posted_dt = datetime.strptime(posted_date[:10], '%Y-%m-%d')
                if posted_dt < one_month_ago:
                    continue
            except Exception:
                continue
        required_skills = set(internship['processed_Required_Skills'])
        gaps = required_skills - user_skills
        if gaps:
            courses = []
            for skill in gaps:
                courses.extend(get_course_links(skill))
            skill_gaps.append({
                'role': internship['role'],
                'company': internship['company_name'],
                'missing_skills': list(gaps),
                'courses': courses
            })
    conn.close()
    # Pass safe HTML for course links
    for gap in skill_gaps:
        gap['courses'] = [Markup(link) for link in gap['courses']]
    return render_template('skill_gap.html', skill_gaps=skill_gaps)

@app.route('/voice_command', methods=['POST'], strict_slashes=False)
@role_required('intern')
def voice_command():
    command = request.json.get('command', '').lower()
    if 'apply to' in command:
        match = re.search(r'internship (\d+)', command)
        if match:
            internship_id = int(match.group(1))
            conn = get_db_connection()
            if not conn:
                return jsonify({'error': 'Database error.'}), 500
            internship = conn.execute('SELECT * FROM internship_info WHERE id = ?', (internship_id,)).fetchone()
            if internship:
                existing_application = conn.execute('SELECT * FROM applications WHERE user_id = ? AND internship_id = ?', (session['user_id'], internship_id)).fetchone()
                if not existing_application:
                    conn.execute('INSERT INTO applications (user_id, internship_id, applied_at) VALUES (?, ?, ?)',
                                 (session['user_id'], internship_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                    conn.commit()
                    conn.close()
                    return jsonify({'message': f'Applied to internship {internship_id} successfully!'})
                conn.close()
                return jsonify({'message': 'You have already applied to this internship.'})
            conn.close()
            return jsonify({'error': 'Internship not found.'})
    elif 'show resume' in command:
        return jsonify({'redirect': url_for('edit_resume')})
    return jsonify({'error': 'Command not recognized.'})

@app.route('/analytics/<int:internship_id>', strict_slashes=False)
@role_required('recruiter')
def analytics(internship_id):
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('recruiter_dashboard'))
    internship = conn.execute('SELECT * FROM internship_info WHERE id = ? AND user_id = ?', (internship_id, user_id)).fetchone()
    if not internship:
        conn.close()
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
    # Calculate applicant-specific analytics
    internship_skills = preprocess_skills(internship['skills_required'])
    applicant_data = []
    for resume in resumes:
        user = conn.execute('SELECT * FROM users WHERE user_id = ?', (resume['user_id'],)).fetchone()
        if user:
            similarity = jaccard_similarity(preprocess_skills(resume['skills']), internship_skills)
            acceptance_prob = similarity * 100  # Simple heuristic: similarity as likelihood
            applicant_data.append({
                'name': resume['name_of_applicant'],
                'email': user['email'],
                'skills': resume['skills'],
                'acceptance_prob': acceptance_prob
            })
    analytics_data = {
        'total_applicants': len(applications),
        'avg_rating': round(avg_rating, 2),
        'top_skills': top_skills,
        'application_trend': application_trend,
        'applicants': applicant_data
    }
    conn.close()
    return render_template('analytics.html', internship=internship, analytics_data=analytics_data, internship_id=internship_id)

@app.route('/progress', strict_slashes=False)
@role_required('intern')
def progress():
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('intern_login'))
    progress = conn.execute('SELECT * FROM user_progress WHERE user_id = ?', (user_id,)).fetchall()
    total_points = sum(p['points'] for p in progress if p['points'] is not None) if progress else 0
    level = total_points // 100
    achievements = [
        {'name': 'Resume Master', 'unlocked': total_points >= 100},
        {'name': 'Application Pro', 'unlocked': total_points >= 150},
        {'name': 'Skill Builder', 'points': 200, 'unlocked': total_points >= 200},
        {'name': 'Mentorship Seeker', 'points': 250, 'unlocked': total_points >= 250}
    ]
    conn.close()
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
    return render_template('interview_prep.html', questions=questions, user_name=session['user_name'])

@app.route('/peer_review', methods=['GET', 'POST'], strict_slashes=False)
@role_required('intern')
def peer_review():
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('intern_login'))
    resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
    if not resume:
        conn.close()
        flash('Please create your resume first!', 'warning')
        return redirect(url_for('create_resume'))
    if request.method == 'POST':
        reviewed_user_id = int(request.form['reviewed_user_id'])
        rating = int(request.form['rating'])
        feedback = request.form['feedback']
        conn.execute('INSERT INTO peer_reviews (reviewer_id, reviewed_user_id, resume_id, rating, feedback, review_date) VALUES (?, ?, ?, ?, ?, ?)',
                     (user_id, reviewed_user_id, reviewed_user_id, rating, feedback, datetime.now().strftime('%Y-%m-%d')))
        conn.execute('INSERT INTO user_progress (user_id, task_type, task_description, completion_date, points) VALUES (?, ?, ?, ?, ?)',
                     (user_id, 'Peer Review', 'Provided resume feedback', datetime.now().strftime('%Y-%m-%d'), 25))
        conn.commit()
        flash('Review submitted successfully!', 'success')
    reviews = conn.execute('SELECT * FROM peer_reviews WHERE reviewed_user_id = ?', (user_id,)).fetchall()
    other_resumes = conn.execute('SELECT * FROM resume_info WHERE user_id != ?', (user_id,)).fetchall()
    conn.close()
    return render_template('peer_review.html', resume=resume, reviews=reviews, other_resumes=other_resumes)

@app.route('/match', strict_slashes=False)
@role_required('intern')
def match():
    user_id = session['user_id']
    global resume_df, internship_df
    resume_df, internship_df = fetch_data()
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('intern_login'))
    resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
    conn.close()
    if not resume:
        flash('Please create your resume!', 'warning')
        return redirect(url_for('create_resume'))
    user_skills = preprocess_skills(resume['skills'])
    soft_skills = resume['soft_skills'] or 'Not assessed'
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
                'similarity_score': round(similarity * 100, 2),
                'soft_skills': soft_skills
            })
    matched_internships = sorted(matched_internships, key=lambda x: x['similarity_score'], reverse=True)
    return render_template('match.html', matched_internships=matched_internships)

@app.route('/top_matched_applicants/<int:internship_id>', strict_slashes=False)
@role_required('recruiter')
def top_matched_applicants(internship_id):
    user_id = session['user_id']
    global resume_df, internship_df
    resume_df, internship_df = fetch_data()
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('recruiter_dashboard'))
    internship = conn.execute('SELECT * FROM internship_info WHERE id = ? AND user_id = ?', (internship_id, user_id)).fetchone()
    if not internship:
        conn.close()
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
                progress = conn.execute('SELECT SUM(points) as total_points FROM user_progress WHERE user_id = ?', (resume['user_id'],)).fetchone()
                total_points = progress['total_points'] or 0
                matched_applicants.append({
                    'name': resume['name_of_applicant'],
                    'email': user['email'],
                    'skills': resume['skills'],
                    'soft_skills': resume['soft_skills'] or 'Not assessed',
                    'similarity_score': round(similarity * 100, 2),
                    'resume_path': resume.get('resume_path', ''),
                    'total_points': total_points
                })
    conn.close()
    matched_applicants = sorted(matched_applicants, key=lambda x: x['similarity_score'], reverse=True)[:5]
    return render_template('top_matched_applicants.html', matched_applicants=matched_applicants, internship_title=internship_title, user_name=session['user_name'])

@app.route('/leaderboard', strict_slashes=False)
def leaderboard():
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('index'))
    query = '''
    SELECT u.name, SUM(p.points) as total_points
    FROM user_progress p
    JOIN users u ON p.user_id = u.user_id
    WHERE u.role = 'intern'
    GROUP BY p.user_id
    ORDER BY total_points DESC
    LIMIT 10
    '''
    leaderboard = conn.execute(query).fetchall()
    conn.close()
    return render_template('leaderboard.html', leaderboard=leaderboard)
    
@app.route('/download_resume/<path:resume_path>', strict_slashes=False)
@role_required('recruiter')
def download_resume(resume_path):
    try:
        return send_file(resume_path, as_attachment=True)
    except FileNotFoundError:
        flash('Resume file not found.', 'danger')
        return redirect(url_for('recruiter_dashboard'))

@app.route('/apply_internship/<int:internship_id>', methods=['POST'], strict_slashes=False)
@role_required('intern')
def apply_internship(internship_id):
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('intern_dashboard'))
    resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
    if not resume or not resume.get('resume_path'):
        conn.close()
        flash('Please upload your resume before applying to internships!', 'danger')
        return redirect(url_for('intern_dashboard'))
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
    conn.execute('INSERT INTO applications (user_id, internship_id, applied_at) VALUES (?, ?, ?)', 
                 (user_id, internship_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.execute('INSERT INTO user_progress (user_id, task_type, task_description, completion_date, points) VALUES (?, ?, ?, ?, ?)',
                 (user_id, 'Application', f'Applied to {internship["role"]}', datetime.now().strftime('%Y-%m-%d'), 50))
    conn.commit()
    conn.close()
    flash('Applied successfully!', 'success')
    return redirect(url_for('intern_dashboard'))

@app.route('/applied_internships', strict_slashes=False)
@role_required('intern')
def applied_internships():
    user_id = session['user_id']
    conn = get_db_connection()
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
    return render_template('applied_internships.html', applied_internships=applied_internships_list, user_name=session['user_name'])

@app.route('/applied_applicants', strict_slashes=False)
@role_required('recruiter')
def applied_applicants():
    user_id = session['user_id']
    conn = get_db_connection()
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
    return render_template('applied_applicants.html', applicants=applicants, internship=None, user_name=session['user_name'])

@app.route('/applied_applicants/<int:internship_id>', strict_slashes=False)
@role_required('recruiter')
def applied_applicants_specific(internship_id):
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('recruiter_login'))
    internship = conn.execute('SELECT * FROM internship_info WHERE id = ? AND user_id = ?', (internship_id, user_id)).fetchone()
    if not internship:
        conn.close()
        flash('Internship not found.', 'danger')
        return render_template('applied_applicants.html', applicants=[], internship=None, user_name=session['user_name'])
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
    return render_template('applied_applicants.html', applicants=applicants_list, internship=internship, user_name=session['user_name'])

@app.route('/edit_profile', methods=['GET', 'POST'], strict_slashes=False)
def edit_profile():
    if 'user_id' not in session:
        flash('Please login!', 'danger')
        return redirect(url_for('intern_login' if session.get('role') == 'intern' else 'recruiter_login'))
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return render_template('edit_profile.html', user=None)
    user = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form.get('password', '')
        existing_user = conn.execute('SELECT * FROM users WHERE email = ? AND user_id != ?', (email, user_id)).fetchone()
        if existing_user:
            flash('Email is already in use!', 'danger')
            conn.close()
            return render_template('edit_profile.html', user=user)
        if password:
            hashed_password = generate_password_hash(password)
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
@role_required('recruiter')
def edit_organization_profile():
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('recruiter_login'))
    user = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
    if request.method == 'POST':
        organization_name = request.form['organization_name'].strip()
        contact_details = request.form['contact_details'].strip()
        location = request.form['location'].strip()
        website_link = request.form['website_link'].strip()
        if not organization_name:
            flash('Organization name is required!', 'danger')
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

@app.route('/verify_credential', methods=['GET', 'POST'], strict_slashes=False)
def verify_credential():
    if 'user_id' not in session:
        flash('Please login.', 'danger')
        return redirect(url_for('index'))
    if request.method == 'POST':
        credential_hash_input = request.form['credential_hash'].strip()
        signature_input = request.form['signature'].strip()
        try:
            conn = get_db_connection()
            if not conn:
                flash('Database error.', 'danger')
                return render_template('verify_credential.html')
            # Fetch all credentials to check for matches
            credentials = conn.execute('SELECT * FROM credentials').fetchall()
            matched_credential = None
            for cred in credentials:
                full_hash = cred['credential_hash']
                full_signature = cred['signature']
                # Check if input matches full or truncated (first 10 + last 10)
                truncated_hash = f"{full_hash[:10]}...{full_hash[-10:]}" if len(full_hash) > 20 else full_hash
                truncated_signature = f"{full_signature[:10]}...{full_signature[-10:]}" if len(full_signature) > 20 else full_signature
                if (credential_hash_input in (full_hash, truncated_hash)) and (signature_input in (full_signature, truncated_signature)):
                    matched_credential = cred
                    break
            if not matched_credential:
                conn.close()
                flash('Credential not found in database.', 'danger')
                return render_template('verify_credential.html')
            # Verify signature using full values
            public_key = serialization.load_pem_public_key(PUBLIC_KEY_PEM.encode())
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
            # Fetch user details for display
            user = conn.execute('SELECT name FROM users WHERE user_id = ?', (matched_credential['user_id'],)).fetchone()
            conn.close()
            # Store credential in session for display
            session['verified_credential'] = {
                'user_id': matched_credential['user_id'],
                'user_name': user['name'] if user else 'Unknown',
                'credential_hash': matched_credential['credential_hash'],
                'signature': matched_credential['signature'],
                'issued_date': matched_credential['issued_date']
            }
            flash('Credential verified successfully!', 'success')
        except InvalidSignature:
            conn.close()
            flash('Invalid signature.', 'danger')
        except Exception as e:
            if conn:
                conn.close()
            logging.error(f"Verification error: {str(e)}")
            flash('Error verifying credential.', 'danger')
    return render_template('verify_credential.html')

@app.route('/issue_credential', methods=['GET', 'POST'], strict_slashes=False)
@role_required('admin')
def issue_credential():
    if not app.secret_key:
        logging.error('FLASK_SECRET_KEY not set')
        flash('Server configuration error.', 'danger')
        return render_template('issue_credential.html')
    logging.info(f"Accessing issue_credential, session: {session.get('user_id', 'None')}, role: {session.get('role', 'None')}")
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return render_template('issue_credential.html')
    interns = conn.execute('SELECT user_id, name FROM users WHERE role = ?', ('intern',)).fetchall()
    conn.close()
    interns_list = [{'user_id': intern['user_id'], 'name': intern['name']} for intern in interns]
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        credential_details = request.form.get('credential_details', '').strip()
        logging.info(f'POST request: user_id={user_id}, credential_details={credential_details}')
        if not credential_details:
            flash('Credential details cannot be empty.', 'danger')
            return render_template('issue_credential.html', interns=interns_list)
        try:
            user_id = int(user_id)
            conn = get_db_connection()
            if not conn:
                flash('Database error.', 'danger')
                return render_template('issue_credential.html', interns=interns_list)
            user = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
            conn.close()
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
            conn = get_db_connection()
            if not conn:
                flash('Database error.', 'danger')
                return render_template('issue_credential.html', interns=interns_list)
            conn.execute('INSERT INTO credentials (user_id, credential_hash, signature, issued_date) VALUES (?, ?, ?, ?)',
                         (user_id, credential_hash, signature_hex, datetime.now().strftime('%Y-%m-%d')))
            conn.commit()
            conn.close()
            session['issued_credential'] = {
                'user_id': user_id,
                'credential_details': credential_details,
                'credential_hash': credential_hash,
                'signature': signature_hex,
                'issued_date': datetime.now().strftime('%Y-%m-%d')
            }
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
@app.route('/soft_skills_assessment', methods=['GET', 'POST'], strict_slashes=False)
@role_required('intern')
def soft_skills_assessment():
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('intern_dashboard'))
    resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
    if not resume:
        conn.close()
        flash('Please create your resume first!', 'warning')
        return redirect(url_for('create_resume'))
    
    if request.method == 'POST':
        problem_solving = int(request.form.get('problem_solving', 0))
        teamwork = int(request.form.get('teamwork', 0))
        creativity = int(request.form.get('creativity', 0))
        soft_skills = f"Problem Solving: {problem_solving}/5, Teamwork: {teamwork}/5, Creativity: {creativity}/5"
        conn.execute('UPDATE resume_info SET soft_skills = ? WHERE user_id = ?', (soft_skills, user_id))
        conn.execute('INSERT INTO user_progress (user_id, task_type, task_description, completion_date, points) VALUES (?, ?, ?, ?, ?)',
                     (user_id, 'Soft Skills Assessment', 'Completed soft skills quiz', datetime.now().strftime('%Y-%m-%d'), 50))
        conn.commit()
        conn.close()
        global resume_df, internship_df
        resume_df, internship_df = fetch_data()
        flash('Soft skills assessed successfully!', 'success')
        return redirect(url_for('intern_dashboard'))
    
    conn.close()
    return render_template('soft_skills_assessment.html')

@app.route('/mock_interview', methods=['GET', 'POST'], strict_slashes=False)
@role_required('intern')
def mock_interview():
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('intern_dashboard'))
    resume = conn.execute('SELECT skills, experience FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
    if not resume:
        conn.close()
        flash('Please create your resume first!', 'warning')
        return redirect(url_for('create_resume'))
    
    questions = [
        {'id': 1, 'question': f'Tell me how you applied {resume["skills"]} in a project.', 'category': 'Technical'},
        {'id': 2, 'question': 'Describe a challenge you faced and how you overcame it.', 'category': 'Behavioral'},
        {'id': 3, 'question': 'Why are you interested in this internship?', 'category': 'General'}
    ]
    
    feedback = None
    if request.method == 'POST':
        question_id = int(request.form.get('question_id'))
        response = request.form.get('response')
        # Use Hugging Face NLP for feedback
        nlp_result = feedback_nlp(response)
        feedback = f"AI Feedback: {nlp_result[0]['label']} (confidence: {nlp_result[0]['score']:.2f})"
        # Optionally, store feedback in DB here

    feedback_history = conn.execute('SELECT * FROM interview_feedback WHERE user_id = ?', (user_id,)).fetchall()
    conn.close()
    return render_template('mock_interview.html', questions=questions, feedback_history=feedback_history, feedback=feedback)
@app.route('/ats_insights', methods=['GET', 'POST'], strict_slashes=False)
@role_required('intern')
def ats_insights():
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('intern_dashboard'))
    resume = conn.execute('SELECT * FROM resume_info WHERE user_id = ?', (user_id,)).fetchone()
    if not resume:
        conn.close()
        flash('Please create your resume first!', 'warning')
        return redirect(url_for('create_resume'))
    keyword_score = None
    tips = []
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_text = ' '.join([resume[field] for field in ['skills', 'experience', 'education', 'certifications', 'achievements'] if resume[field]])
        r = Rake()
        r.extract_keywords_from_text(job_description)
        job_keywords = set(r.get_ranked_phrases())
        r.extract_keywords_from_text(resume_text)
        resume_keywords = set(r.get_ranked_phrases())
        match_count = len(job_keywords & resume_keywords)
        keyword_score = int((match_count / len(job_keywords)) * 100) if job_keywords else 0
        tips = [
            f"Add more keywords from the job description: {', '.join(job_keywords - resume_keywords)}"
        ]
    conn.close()
    return render_template('ats_insights.html', keyword_score=keyword_score, tips=tips)
@app.route('/become_mentor', methods=['GET', 'POST'], strict_slashes=False)
@role_required('recruiter')
def become_mentor():
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('recruiter_dashboard'))
    
    if request.method == 'POST':
        industry = request.form.get('industry')
        skills = request.form.get('skills')
        availability = request.form.get('availability')
        existing_mentor = conn.execute('SELECT * FROM mentors WHERE user_id = ?', (user_id,)).fetchone()
        if existing_mentor:
            conn.execute('UPDATE mentors SET industry = ?, skills = ?, availability = ? WHERE user_id = ?',
                         (industry, skills, availability, user_id))
        else:
            conn.execute('INSERT INTO mentors (user_id, industry, skills, availability) VALUES (?, ?, ?, ?)',
                         (user_id, industry, skills, availability))
        conn.commit()
        conn.close()
        flash('Mentor profile created/updated successfully!', 'success')
        return redirect(url_for('recruiter_dashboard'))
    
    mentor = conn.execute('SELECT * FROM mentors WHERE user_id = ?', (user_id,)).fetchone()
    conn.close()
    return render_template('become_mentor.html', mentor=mentor)

@app.route('/mentorship', methods=['GET', 'POST'], strict_slashes=False)
@role_required('intern')
def mentorship():
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('intern_dashboard'))
    
    if request.method == 'POST':
        mentor_id = int(request.form.get('mentor_id'))
        message = request.form.get('message')
        conn.execute('INSERT INTO mentorship_requests (intern_id, mentor_id, message, request_date) VALUES (?, ?, ?, ?)',
                     (user_id, mentor_id, message, datetime.now().strftime('%Y-%m-%d')))
        conn.execute('INSERT INTO user_progress (user_id, task_type, task_description, completion_date, points) VALUES (?, ?, ?, ?, ?)',
                     (user_id, 'Mentorship Request', 'Sent mentorship request', datetime.now().strftime('%Y-%m-%d'), 25))
        conn.commit()
        conn.close()
        flash('Mentorship request sent successfully!', 'success')
        return redirect(url_for('mentorship'))
    
    mentors = conn.execute('SELECT m.*, u.name FROM mentors m JOIN users u ON m.user_id = u.user_id').fetchall()
    requests = conn.execute('SELECT mr.*, u.name FROM mentorship_requests mr JOIN mentors m ON mr.mentor_id = m.mentor_id JOIN users u ON m.user_id = u.user_id WHERE mr.intern_id = ?', (user_id,)).fetchall()
    conn.close()
    return render_template('mentorship.html', mentors=mentors, requests=requests)

@app.route('/mentor_dashboard', strict_slashes=False)
@role_required('recruiter')
def mentor_dashboard():
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('recruiter_dashboard'))
    mentor = conn.execute('SELECT * FROM mentors WHERE user_id = ?', (user_id,)).fetchone()
    if not mentor:
        conn.close()
        flash('Please create a mentor profile first!', 'warning')
        return redirect(url_for('become_mentor'))
    requests = conn.execute('SELECT mr.*, u.name FROM mentorship_requests mr JOIN users u ON mr.intern_id = u.user_id WHERE mr.mentor_id = ?', (mentor['mentor_id'],)).fetchall()
    conn.close()
    return render_template('mentor_dashboard.html', mentor=mentor, requests=requests)

@app.route('/respond_mentorship/<int:request_id>', methods=['POST'], strict_slashes=False)
@role_required('recruiter')
def respond_mentorship(request_id):
    user_id = session['user_id']
    conn = get_db_connection()
    if not conn:
        flash('Database error.', 'danger')
        return redirect(url_for('mentor_dashboard'))
    mentor = conn.execute('SELECT * FROM mentors WHERE user_id = ?', (user_id,)).fetchone()
    if not mentor:
        conn.close()
        flash('Mentor profile not found.', 'danger')
        return redirect(url_for('recruiter_dashboard'))
    status = request.form.get('status')
    conn.execute('UPDATE mentorship_requests SET status = ? WHERE request_id = ? AND mentor_id = ?', (status, request_id, mentor['mentor_id']))
    conn.execute('INSERT INTO user_progress (user_id, task_type, task_description, completion_date, points) VALUES (?, ?, ?, ?, ?)',
                 (user_id, 'Mentorship Response', f'Responded to mentorship request {request_id}', datetime.now().strftime('%Y-%m-%d'), 25))
    conn.commit()
    conn.close()
    flash(f'Mentorship request {status} successfully!', 'success')
    return redirect(url_for('mentor_dashboard'))


@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    error_message = f"Exception: {str(e)}\n{traceback.format_exc()}"
    logging.error(error_message)
    return render_template('500.html', error_message=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 7860)))