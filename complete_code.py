# Importing necessary libraries
import mysql.connector
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import tkinter as tk
from tkinter import ttk, messagebox

# Download NLTK dependencies
nltk.download('punkt')
nltk.download('stopwords')

# MySQL Database Connection
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Chaitu895@",  # Change to your actual MySQL password
        database="resume_internship_matching"
    )

# Fetching data from MySQL
def fetch_data():
    conn = connect_db()
    cursor = conn.cursor()

    # Fetch resumes
    cursor.execute("SELECT * FROM resume_info")
    resumes = cursor.fetchall()
    
    # Get column names from MySQL tables
    cursor.execute("SHOW COLUMNS FROM resume_info")
    resume_columns = [column[0] for column in cursor.fetchall()]
    # Fetch internships
    cursor.execute("SELECT * FROM internship_info")
    internships = cursor.fetchall()
    cursor.execute("SHOW COLUMNS FROM internship_info")
    internship_columns = [column[0] for column in cursor.fetchall()]
    resume_df = pd.DataFrame(resumes, columns=resume_columns)
    internship_df = pd.DataFrame(internships, columns=internship_columns)
    cursor.close()
    conn.close()
    print(resume_df.head())
    print(internship_df.head())
    return resume_df, internship_df

# Preprocessing function for skills
def preprocess_skills(skills):
    if not isinstance(skills, str) or skills.strip() == '':
        return []
    tokens = word_tokenize(skills.lower())
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in string.punctuation]
    return tokens

# Load data from database
resume_df, internship_df = fetch_data()

# Preprocessing
resume_df.fillna('', inplace=True)
internship_df.fillna('', inplace=True)

resume_df['processed_Skills'] = resume_df['skills'].apply(preprocess_skills)
print("Processed Resume Skills:")
print(resume_df[['processed_Skills']].head())
internship_df['processed_Required_Skills'] = internship_df['skills_required'].apply(preprocess_skills)
print("\nProcessed Internship Skills:")
print(internship_df[['processed_Required_Skills']].head())
# Creating a set of unique skills
all_Skills = resume_df['processed_Skills'].sum() + internship_df['processed_Required_Skills'].sum()
unique_Skills = set(all_Skills)
Skill_to_index = {skill: idx for idx, skill in enumerate(unique_Skills)}

# Converting skills to numerical vectors
def skills_to_vector(skills):
    vector = [0] * len(Skill_to_index)
    for skill in skills:
        if skill in Skill_to_index:
            vector[Skill_to_index[skill]] += 1
    return vector

resume_df['Skill_vector'] = resume_df['processed_Skills'].apply(skills_to_vector)
internship_df['Required_Skill_vector'] = internship_df['processed_Required_Skills'].apply(skills_to_vector)

# Function to calculate Jaccard similarity
def calculate_similarity(resume_skills, internship_skills):
    set_resume_skills = set(resume_skills)
    set_internship_skills = set(internship_skills)
    intersection = set_resume_skills.intersection(set_internship_skills)
    union = set_resume_skills.union(set_internship_skills)
    return len(intersection) / len(union) if len(union) != 0 else 0

# Function to match internships
def match_internships(resume):
    results = []
    for index, internship in internship_df.iterrows():
        similarity_score = calculate_similarity(resume['processed_Skills'], internship['processed_Required_Skills'])
        if similarity_score > 0:  # Include all matches with a similarity score greater than 0
            results.append({
                'internship_title': internship['role'],
                'company': internship['company_name'],
                'type_of_internship': internship['type_of_internship'],
                'duration': internship['duration'],
                'location': internship['location'],
                'description': internship['description_of_internship'],
                'skills_required': internship['skills_required'],
                'salary': internship['expected_salary'],
                'start_date': internship['start_date'],
                'end_date': internship['end_date'],
                'posted_date': internship['posted_date'],
                'similarity_score': similarity_score
            })
    
    # Sort by highest similarity score and return the top 5 results
    results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)[:5]
    return results

# Function to display matched internships
def show_results(results, resume_name):
    if not results:
        messagebox.showinfo("No Matches", f"No internships matched for {resume_name}.")
        return
    
    results_window = tk.Toplevel(root)
    results_window.title(f"Top 5 Matched Internships for {resume_name}")
    results_window.geometry("800x600")
    results_window.configure(bg='#fafafa')

    tk.Label(results_window, text=f"Top 5 Matched Internships for {resume_name}", font=('Helvetica', 16, 'bold'), bg='#fafafa').pack(pady=10)

    for idx, result in enumerate(results, start=1):
        tk.Label(results_window, text=f"#{idx}: {result['internship_title']} at {result['company']}", font=('Helvetica', 14), bg='#fafafa').pack(pady=5)
        tk.Label(results_window, text=f"Location: {result['location']}", font=('Helvetica', 12), bg='#fafafa').pack(pady=2)
        tk.Label(results_window, text=f"Type: {result['type_of_internship']}", font=('Helvetica', 12), bg='#fafafa').pack(pady=2)
        tk.Label(results_window, text=f"Skills Required: {result['skills_required']}", font=('Helvetica', 12), wraplength=700, bg='#fafafa').pack(pady=2)
        tk.Label(results_window, text=f"Duration: {result['duration']}", font=('Helvetica', 12), bg='#fafafa').pack(pady=2)
        tk.Label(results_window, text=f"Salary: {result['salary']}", font=('Helvetica', 12), bg='#fafafa').pack(pady=2)
        tk.Label(results_window, text=f"Start Date: {result['start_date']}", font=('Helvetica', 12), bg='#fafafa').pack(pady=2)
        tk.Label(results_window, text=f"End Date: {result['end_date']}", font=('Helvetica', 12), bg='#fafafa').pack(pady=2)
        tk.Label(results_window, text=f"Posted Date: {result['posted_date']}", font=('Helvetica', 12), bg='#fafafa').pack(pady=2)
        tk.Label(results_window, text=f"Probability of Getting the Job: {result['similarity_score'] * 100:.2f}%", font=('Helvetica', 12, 'bold'), bg='#fafafa').pack(pady=5)
        tk.Label(results_window, text="-" * 80, font=('Helvetica', 12), bg='#fafafa').pack(pady=5)

    tk.Button(results_window, text="Close", command=results_window.destroy, bg='#00796b', fg='white', font=('Helvetica', 12)).pack(pady=20)

# Function to find and match internships
def find_applicant_and_match_internships():
    applicant_name = entry_name.get().strip()
    if not applicant_name:
        messagebox.showwarning("Input Error", "Please enter a valid applicant name.")
        return

    matching_resume = resume_df[resume_df['name_of_applicant'].str.contains(applicant_name, case=False)]
    if matching_resume.empty:
        messagebox.showinfo("No Results", f"No resume found for applicant: {applicant_name}")
    else:
        resume = matching_resume.iloc[0]
        matched_internships = match_internships(resume)
        show_results(matched_internships, resume['name_of_applicant'])

# Creating the main application window
root = tk.Tk()
root.title("SkillSync - Resume Based Internship Matcher")
root.geometry("800x600")
root.configure(bg='#e0f7fa')

# UI Styling
style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12), padding=10)
style.configure('TLabel', font=('Helvetica', 12), padding=10)
style.configure('TEntry', font=('Helvetica', 12))

# UI Elements
title_label = ttk.Label(root, text="SkillSync - Resume Based Internship Matcher", font=('Helvetica', 24), background='#e0f7fa')
title_label.pack(pady=20)

name_label = ttk.Label(root, text="Enter Applicant Name:", background='#e0f7fa')
name_label.pack(pady=10)

entry_name = ttk.Entry(root, width=40)
entry_name.pack(pady=10)

search_button = ttk.Button(root, text="Find Matching Internships", command=find_applicant_and_match_internships)
search_button.pack(pady=20)

# Run Tkinter main loop
root.mainloop()
