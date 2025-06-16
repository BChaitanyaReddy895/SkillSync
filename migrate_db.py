import sqlite3

def migrate_db():
    conn = sqlite3.connect('/tmp/database.db')
    cursor = conn.cursor()
    
    # Create user_progress table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_progress (
            user_id INTEGER PRIMARY KEY,
            points INTEGER DEFAULT 0,
            badges TEXT DEFAULT '[]'
        )
    ''')
    
    # Create peer_reviews table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS peer_reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_id INTEGER,
            reviewer_id INTEGER,
            feedback TEXT,
            FOREIGN KEY (resume_id) REFERENCES resume_info(resume_id),
            FOREIGN KEY (reviewer_id) REFERENCES users(user_id)
        )
    ''')
    
    # Create internship_ratings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS internship_ratings (
            internship_id INTEGER,
            user_id INTEGER,
            rating INTEGER,
            PRIMARY KEY (internship_id, user_id),
            FOREIGN KEY (internship_id) REFERENCES internship_info(internship_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    ''')
    
    # Add resume_id to resume_info
    cursor.execute('PRAGMA table_info(resume_info)')
    columns = [info[1] for info in cursor.fetchall()]
    if 'resume_id' not in columns:
        cursor.execute('ALTER TABLE resume_info ADD COLUMN resume_id INTEGER PRIMARY KEY AUTOINCREMENT')
    
    # Initialize user_progress for existing users
    cursor.execute('SELECT user_id FROM users WHERE role = ?', ('intern',))
    interns = cursor.fetchall()
    for intern in interns:
        cursor.execute('INSERT OR IGNORE INTO user_progress (user_id, points, badges) VALUES (?, 0, ?)', 
                      (intern['user_id'], '[]'))
    
    conn.commit()
    conn.close()
    print("Database migration completed.")

if __name__ == '__main__':
    migrate_db()