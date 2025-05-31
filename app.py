from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from pymongo import MongoClient
import urllib.parse
import os
import logging

# Configure logging to a file and console
log_dir = "/tmp/logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "app.log")),
        logging.StreamHandler()  # Also log to console for Hugging Face Spaces
    ]
)
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Download NLTK dependencies
nltk_data_dir = "/tmp/nltk_data"
nltk.data.path.append(nltk_data_dir)
logger.info("Downloading NLTK data...")
try:
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
    logger.info("NLTK data downloaded successfully")
except Exception as e:
    logger.error(f"Failed to download NLTK data: {str(e)}")
    raise

# MongoDB Connection
logger.info("Connecting to MongoDB...")
try:
    username = os.getenv('MONGO_USERNAME', 'root')
    password = os.getenv('MONGO_PASSWORD', 'Chaitu895@')
    host = os.getenv('MONGO_HOST', 'cluster0.zklixmv.mongodb.net')
    database = os.getenv('MONGO_DATABASE', 'skillsync')

    encoded_username = urllib.parse.quote_plus(username)
    encoded_password = urllib.parse.quote_plus(password)
    MONGO_URI = f"mongodb+srv://{encoded_username}:{encoded_password}@{host}/{database}?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Test the connection
    client.server_info()  # This will raise an exception if the connection fails
    db = client['skillsync']
    logger.info("MongoDB connection successful")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise

# Custom error handler for 500 errors
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal Server Error: {str(error)}")
    return "Internal Server Error: Check the server logs for details.", 500

# Preprocessing function for skills
def preprocess_skills(skills):
    if not isinstance(skills, str) or skills.strip() == '':
        return []
    tokens = word_tokenize(skills.lower())
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    return tokens

# Fetch data from MongoDB
def fetch_data():
    logger.info("Fetching resumes from MongoDB")
    resumes = list(db.resume_info.find())
    resume_df = pd.DataFrame(resumes)
    logger.info(f"Fetched {len(resumes)} resumes")

    logger.info("Fetching internships from MongoDB")
    internships = list(db.internship_info.find())
    internship_df = pd.DataFrame(internships)
    logger.info(f"Fetched {len(internships)} internships")

    resume_df.fillna('', inplace=True)
    internship_df.fillna('', inplace=True)
    resume_df['processed_Skills'] = resume_df['skills'].apply(preprocess_skills)
    internship_df['processed_Required_Skills'] = internship_df['skills_required'].apply(preprocess_skills)

    resume_skills = resume_df['processed_Skills'].explode().dropna().tolist()
    internship_skills = internship_df['processed_Required_Skills'].explode().dropna().tolist()
    all_skills = resume_skills + internship_skills

    global skill_to_index
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

# Global variables for DataFrames (initially None)
resume_df = None
internship_df = None

# Load data on first request
@app.before_first_request
def load_data():
    global resume_df, internship_df
    logger.info("Loading data on first request...")
    try:
        resume_df, internship_df = fetch_data()
        logger.info("Data loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

# Jaccard Similarity for matching
def jaccard_similarity(vec1, vec2):
    set1, set2 = set(vec1), set(vec2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# Routes
@app.route('/')
def index():
    try:
        logger.info("Rendering index.html")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index.html: {str(e)}")
        raise

@app.route('/recruiter_login', methods=['GET', 'POST'])
def recruiter_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        logger.info(f"Recruiter login attempt with email: {email}")
        recruiter = db.recruiter_info.find_one({'email': email, 'password': password})
        if recruiter:
            session['user_id'] = str(recruiter['_id'])
            session['user_type'] = 'recruiter'
            session['user_name'] = recruiter['name']
            flash('Login successful!', 'success')
            logger.info(f"Recruiter login successful: {email}")
            return redirect(url_for('recruiter_dashboard'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
            logger.warning(f"Recruiter login failed: {email}")
    return render_template('recruiter_login.html')

@app.route('/intern_login', methods=['GET', 'POST'])
def intern_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        logger.info(f"Intern login attempt with email: {email}")
        intern = db.intern_info.find_one({'email': email, 'password': password})
        if intern:
            session['user_id'] = str(intern['_id'])
            session['user_type'] = 'intern'
            session['user_name'] = intern['name']
            flash('Login successful!', 'success')
            logger.info(f"Intern login successful: {email}")
            return redirect(url_for('intern_dashboard'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
            logger.warning(f"Intern login failed: {email}")
    return render_template('intern_login.html')

@app.route('/recruiter_signup', methods=['GET', 'POST'])
def recruiter_signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        company = request.form['company']
        logger.info(f"Recruiter signup attempt with email: {email}")
        if db.recruiter_info.find_one({'email': email}):
            flash('Email already exists. Please use a different email.', 'danger')
            logger.warning(f"Recruiter signup failed: Email {email} already exists")
        else:
            db.recruiter_info.insert_one({
                'name': name,
                'email': email,
                'password': password,
                'company': company
            })
            flash('Signup successful! Please login.', 'success')
            logger.info(f"Recruiter signup successful: {email}")
            return redirect(url_for('recruiter_login'))
    return render_template('recruiter_signup.html')

@app.route('/intern_signup', methods=['GET', 'POST'])
def intern_signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        skills = request.form['skills']
        logger.info(f"Intern signup attempt with email: {email}")
        if db.intern_info.find_one({'email': email}):
            flash('Email already exists. Please use a different email.', 'danger')
            logger.warning(f"Intern signup failed: Email {email} already exists")
        else:
            db.intern_info.insert_one({
                'name': name,
                'email': email,
                'password': password,
                'skills': skills
            })
            flash('Signup successful! Please login.', 'success')
            logger.info(f"Intern signup successful: {email}")
            return redirect(url_for('intern_login'))
    return render_template('intern_signup.html')

@app.route('/recruiter_dashboard')
def recruiter_dashboard():
    if 'user_id' not in session or session['user_type'] != 'recruiter':
        flash('Please login as a recruiter.', 'danger')
        return redirect(url_for('recruiter_login'))
    recruiter = db.recruiter_info.find_one({'_id': session['user_id']})
    internships = list(db.internship_info.find({'recruiter_id': session['user_id']}))
    return render_template('recruiter_dashboard.html', recruiter=recruiter, internships=internships)

@app.route('/intern_dashboard')
def intern_dashboard():
    if 'user_id' not in session or session['user_type'] != 'intern':
        flash('Please login as an intern.', 'danger')
        return redirect(url_for('intern_login'))
    intern = db.intern_info.find_one({'_id': session['user_id']})
    internships = list(db.internship_info.find())
    return render_template('intern_dashboard.html', intern=intern, internships=internships)

@app.route('/register_internship', methods=['GET', 'POST'])
def register_internship():
    if 'user_id' not in session or session['user_type'] != 'recruiter':
        flash('Please login as a recruiter.', 'danger')
        return redirect(url_for('recruiter_login'))
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        skills_required = request.form['skills_required']
        db.internship_info.insert_one({
            'title': title,
            'description': description,
            'skills_required': skills_required,
            'recruiter_id': session['user_id']
        })
        flash('Internship registered successfully!', 'success')
        return redirect(url_for('recruiter_dashboard'))
    return render_template('register_internship.html')

@app.route('/upload_resume', methods=['GET', 'POST'])
def upload_resume():
    if 'user_id' not in session or session['user_type'] != 'intern':
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
            filename = f"{session['user_id']}_resume.pdf"
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            skills = request.form['skills']
            db.resume_info.insert_one({
                'intern_id': session['user_id'],
                'resume_path': file_path,
                'skills': skills
            })
            flash('Resume uploaded successfully!', 'success')
            return redirect(url_for('intern_dashboard'))
        else:
            flash('Allowed file types are pdf only', 'danger')
    return render_template('upload_resume.html')

@app.route('/match_resumes/<internship_id>')
def match_resumes(internship_id):
    if 'user_id' not in session or session['user_type'] != 'recruiter':
        flash('Please login as a recruiter.', 'danger')
        return redirect(url_for('recruiter_login'))
    
    global resume_df, internship_df
    if resume_df is None or internship_df is None:
        resume_df, internship_df = fetch_data()

    internship = db.internship_info.find_one({'_id': internship_id})
    if not internship:
        flash('Internship not found.', 'danger')
        return redirect(url_for('recruiter_dashboard'))

    matches = []
    for _, resume in resume_df.iterrows():
        if 'Skill_vector' in resume and 'Required_Skill_vector' in internship_df[internship_df['_id'] == internship_id]:
            similarity = jaccard_similarity(resume['Skill_vector'], internship_df[internship_df['_id'] == internship_id]['Required_Skill_vector'].iloc[0])
            intern = db.intern_info.find_one({'_id': resume['intern_id']})
            if intern:
                matches.append({
                    'intern_name': intern['name'],
                    'email': intern['email'],
                    'similarity': similarity,
                    'resume_path': resume['resume_path']
                })
    
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    return render_template('match_resumes.html', internship=internship, matches=matches)

@app.route('/download_resume/<path:resume_path>')
def download_resume(resume_path):
    if 'user_id' not in session or session['user_type'] != 'recruiter':
        flash('Please login as a recruiter.', 'danger')
        return redirect(url_for('recruiter_login'))
    return send_file(resume_path, as_attachment=True)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))