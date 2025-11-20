"""
Advanced NLP/ML Utilities for SkillSync
This module contains all the intelligent ML features for enhanced resume matching,
scoring, prediction, and recommendations.
"""

import os
import numpy as np
import pandas as pd

# Disable TensorFlow logging to avoid Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Try to import sentence_transformers with better error handling
try:
    from sentence_transformers import SentenceTransformer, util
    _SENT_TRANSFORMERS_AVAILABLE = True
except Exception as e:
    # sentence_transformers not available — provide safe fallbacks so static analysis
    # won't fail and runtime code can fallback to simpler heuristics.
    print(f"Warning: sentence_transformers not available: {str(e)}")
    SentenceTransformer = None
    _SENT_TRANSFORMERS_AVAILABLE = False

    class _UtilFallback:
        @staticmethod
        def pytorch_cos_sim(a, b):
            # Basic numpy cosine similarity fallback that provides an .item() method
            try:
                a_np = np.array(a)
                b_np = np.array(b)
                if a_np.ndim == 1:
                    a_np = a_np.reshape(1, -1)
                if b_np.ndim == 1:
                    b_np = b_np.reshape(1, -1)
                num = (a_np * b_np).sum(axis=1)
                denom = np.linalg.norm(a_np, axis=1) * np.linalg.norm(b_np, axis=1)
                denom = np.where(denom == 0, 1e-8, denom)
                sim = num / denom

                class _Sim:
                    def __init__(self, v):
                        self._v = v
                    def item(self):
                        try:
                            return float(self._v[0])
                        except Exception:
                            return float(self._v)

                return _Sim(sim)
            except Exception:
                class _ZeroSim:
                    def item(self): return 0.0
                return _ZeroSim()

    util = _UtilFallback()
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except Exception as e:
    print(f"Warning: xgboost not available: {str(e)}")
    xgb = None
    _XGB_AVAILABLE = False
try:
    import joblib
    _JOBLIB_AVAILABLE = True
except Exception:
    _JOBLIB_AVAILABLE = False
import logging
from collections import Counter
from typing import List, Dict, Tuple
import re
try:
    import textstat
    _TEXTSTAT_AVAILABLE = True
except Exception:
    _TEXTSTAT_AVAILABLE = False

    class _TextstatFallback:
        @staticmethod
        def flesch_reading_ease(text):
            """
            Lightweight fallback for textstat.flesch_reading_ease using a simple heuristic:
            - estimate sentence count by splitting on punctuation,
            - estimate word count via word tokens,
            - estimate syllables by counting vowel groups per word.
            This provides a rough readability score when textstat is unavailable.
            """
            # Basic sentence and word tokenization
            sentences = re.split(r'[.!?]+', text)
            sentences = [s for s in sentences if s.strip()]
            words = re.findall(r'\w+', text)
            word_count = len(words) or 1
            sentence_count = max(1, len(sentences))
            # Estimate syllables as number of vowel groups per word
            syllables = sum(len(re.findall(r'[aeiouy]+', w.lower())) for w in words) or 1
            asl = word_count / sentence_count  # average sentence length
            asw = syllables / word_count       # average syllables per word
            # Flesch reading ease formula approximation
            score = 206.835 - (1.015 * asl) - (84.6 * asw)
            return score

    textstat = _TextstatFallback()

try:
    from fuzzywuzzy import fuzz
    _FUZZYWUZZY_AVAILABLE = True
except Exception as e:
    print(f"Warning: fuzzywuzzy not available: {str(e)}")
    _FUZZYWUZZY_AVAILABLE = False
    class _FuzzFallback:
        @staticmethod
        def ratio(s1, s2):
            # Simple Levenshtein distance fallback
            if s1 == s2:
                return 100
            return 50
    fuzz = _FuzzFallback()

# Configure cache directory for models
MODELS_CACHE = os.getenv('TRANSFORMERS_CACHE', '/tmp/hf_cache')

# Global model instances (lazy loading)
_semantic_model = None
_sentiment_analyzer = None
_ner_model = None

def get_semantic_model():
    """Load or return cached sentence transformer model"""
    global _semantic_model
    if _semantic_model is None:
        try:
            _semantic_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODELS_CACHE)
            logging.info("Semantic model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading semantic model: {str(e)}")
            _semantic_model = None
    return _semantic_model

def get_sentiment_analyzer():
    """Load or return cached sentiment analysis pipeline"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        try:
            # Set TensorFlow to not be required for transformers
            os.environ['TRANSFORMERS_NO_TF'] = '1'
            from transformers import pipeline
            _sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1,  # CPU
                framework='pt'  # Force PyTorch backend
            )
            logging.info("Sentiment analyzer loaded successfully")
        except Exception as e:
            logging.warning(f"Sentiment analyzer not available: {str(e)}")
            _sentiment_analyzer = None
    return _sentiment_analyzer

def get_ner_model():
    """Load or return cached NER model using spaCy-like transformers"""
    global _ner_model
    if _ner_model is None:
        try:
            # Set TensorFlow to not be required for transformers
            os.environ['TRANSFORMERS_NO_TF'] = '1'
            from transformers import pipeline
            _ner_model = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
                device=-1,  # CPU
                framework='pt'  # Force PyTorch backend
            )
            logging.info("NER model loaded successfully")
        except Exception as e:
            logging.warning(f"NER model not available: {str(e)}")
            _ner_model = None
    return _ner_model


# ============================================================================
# 1. SEMANTIC MATCHING ENGINE
# ============================================================================

def semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts using sentence transformers
    Returns similarity score between 0 and 1
    """
    model = get_semantic_model()
    if model is None:
        # Fallback to simple word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)
    
    try:
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
        return max(0.0, min(1.0, similarity))
    except Exception as e:
        logging.error(f"Error in semantic similarity: {str(e)}")
        return 0.0

def enhanced_skill_matching(user_skills: List[str], required_skills: List[str]) -> Dict:
    """
    Advanced skill matching using semantic similarity
    Returns detailed match information
    """
    if not user_skills or not required_skills:
        return {
            'overall_score': 0.0,
            'matched_skills': [],
            'missing_skills': required_skills,
            'semantic_matches': []
        }
    
    user_skills_text = ' '.join(user_skills)
    required_skills_text = ' '.join(required_skills)
    
    # Overall semantic similarity
    overall_score = semantic_similarity(user_skills_text, required_skills_text)
    
    # Individual skill matching
    matched = []
    missing = []
    semantic_matches = []
    
    for req_skill in required_skills:
        best_match_score = 0.0
        best_match_skill = None
        
        for user_skill in user_skills:
            score = semantic_similarity(user_skill, req_skill)
            if score > best_match_score:
                best_match_score = score
                best_match_skill = user_skill
        
        if best_match_score > 0.7:  # Strong match threshold
            matched.append(req_skill)
            if best_match_skill != req_skill:
                semantic_matches.append({
                    'required': req_skill,
                    'user_has': best_match_skill,
                    'score': round(best_match_score, 3)
                })
        else:
            missing.append(req_skill)
    
    return {
        'overall_score': round(overall_score, 3),
        'matched_skills': matched,
        'missing_skills': missing,
        'semantic_matches': semantic_matches,
        'match_percentage': round((len(matched) / len(required_skills)) * 100, 1)
    }


# ============================================================================
# 2. INTELLIGENT SKILL EXTRACTION WITH NER
# ============================================================================

# Comprehensive skill keywords database
TECHNICAL_SKILLS = {
    'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 
                   'go', 'rust', 'typescript', 'scala', 'r', 'matlab', 'perl'],
    'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring', 
           'express', 'fastapi', 'next.js', 'nuxt.js', 'svelte'],
    'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
                'oracle', 'dynamodb', 'firebase'],
    'ml_ai': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras', 'scikit-learn',
             'nlp', 'computer vision', 'neural networks', 'transformers', 'bert', 'gpt'],
    'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'ci/cd',
             'microservices', 'serverless'],
    'tools': ['git', 'github', 'gitlab', 'jira', 'confluence', 'slack', 'vscode', 'intellij']
}

SOFT_SKILLS = ['leadership', 'communication', 'teamwork', 'problem solving', 'critical thinking',
              'time management', 'adaptability', 'creativity', 'collaboration', 'negotiation',
              'public speaking', 'presentation', 'analytical', 'detail-oriented', 'self-motivated']

def extract_skills_intelligent(text: str) -> Dict[str, List[str]]:
    """
    Extract skills using NER and fuzzy matching
    Returns categorized skills
    """
    if not text:
        return {'technical': [], 'soft': [], 'all': []}
    
    text_lower = text.lower()
    technical_skills = []
    soft_skills = []
    
    # Extract using fuzzy matching
    all_technical = [skill for category in TECHNICAL_SKILLS.values() for skill in category]
    
    for skill in all_technical:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            technical_skills.append(skill)
        else:
            # Fuzzy match for variations
            words = text_lower.split()
            for word in words:
                if fuzz.ratio(skill, word) > 85:
                    technical_skills.append(skill)
                    break
    
    for skill in SOFT_SKILLS:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            soft_skills.append(skill)
    
    # Try NER extraction for additional entities
    ner_model = get_ner_model()
    if ner_model:
        try:
            entities = ner_model(text[:512])  # Limit text length
            for entity in entities:
                if entity['entity_group'] in ['ORG', 'MISC']:
                    word = entity['word'].lower().strip()
                    if len(word) > 2 and word not in technical_skills:
                        technical_skills.append(word)
        except Exception as e:
            logging.warning(f"NER extraction warning: {str(e)}")
    
    return {
        'technical': list(set(technical_skills)),
        'soft': list(set(soft_skills)),
        'all': list(set(technical_skills + soft_skills))
    }


# ============================================================================
# 3. AI-POWERED RESUME SCORING
# ============================================================================

def calculate_resume_score(resume_data: Dict, job_description: str = None) -> Dict:
    """
    Comprehensive resume scoring with multiple dimensions
    """
    score_breakdown = {}
    
    # 1. Completeness Score (0-25 points)
    required_fields = ['skills', 'experience', 'education', 'phone_number', 'email']
    filled_fields = sum(1 for field in required_fields if resume_data.get(field))
    completeness_score = (filled_fields / len(required_fields)) * 25
    score_breakdown['completeness'] = round(completeness_score, 1)
    
    # 2. Skills Depth Score (0-25 points)
    skills_text = resume_data.get('skills', '')
    extracted_skills = extract_skills_intelligent(skills_text)
    technical_count = len(extracted_skills['technical'])
    soft_count = len(extracted_skills['soft'])
    skills_depth = min(25, (technical_count * 2 + soft_count) * 1.5)
    score_breakdown['skills_depth'] = round(skills_depth, 1)
    
    # 3. Experience Quality Score (0-25 points)
    experience = resume_data.get('experience', '')
    experience_score = 0
    if experience:
        # Check for quantifiable achievements (numbers, percentages)
        numbers = re.findall(r'\d+', experience)
        experience_score += min(10, len(numbers) * 2)
        # Check for action verbs
        action_verbs = ['developed', 'created', 'managed', 'led', 'implemented', 'designed', 
                       'built', 'optimized', 'increased', 'improved']
        found_verbs = sum(1 for verb in action_verbs if verb in experience.lower())
        experience_score += min(10, found_verbs * 2)
        # Length check
        if len(experience) > 100:
            experience_score += 5
    score_breakdown['experience_quality'] = round(experience_score, 1)
    
    # 4. Job Match Score (0-25 points) - if job description provided
    job_match_score = 0
    if job_description:
        resume_text = ' '.join([str(resume_data.get(field, '')) for field in 
                               ['skills', 'experience', 'education', 'certifications']])
        job_match_score = semantic_similarity(resume_text, job_description) * 25
    else:
        # Default to skills assessment
        job_match_score = min(25, technical_count * 2)
    score_breakdown['job_match'] = round(job_match_score, 1)
    
    # Total Score
    total_score = sum(score_breakdown.values())
    
    # Generate recommendations
    recommendations = []
    if completeness_score < 20:
        missing = [f for f in required_fields if not resume_data.get(f)]
        recommendations.append(f"Complete missing sections: {', '.join(missing)}")
    if skills_depth < 15:
        recommendations.append("Add more technical skills and certifications")
    if experience_score < 15:
        recommendations.append("Use action verbs and quantify achievements (e.g., 'Increased efficiency by 30%')")
    if technical_count < 5:
        recommendations.append("List at least 5-7 technical skills relevant to your field")
    
    return {
        'total_score': round(total_score, 1),
        'grade': get_grade(total_score),
        'breakdown': score_breakdown,
        'recommendations': recommendations,
        'technical_skills_count': technical_count,
        'soft_skills_count': soft_count
    }

def get_grade(score: float) -> str:
    """Convert score to letter grade"""
    if score >= 90:
        return 'A+ (Excellent)'
    elif score >= 80:
        return 'A (Very Good)'
    elif score >= 70:
        return 'B (Good)'
    elif score >= 60:
        return 'C (Fair)'
    else:
        return 'D (Needs Improvement)'


# ============================================================================
# 4. INTERVIEW RESPONSE ANALYSIS
# ============================================================================

def analyze_interview_response(question: str, response: str) -> Dict:
    """
    Analyze interview response using NLP metrics
    """
    if not response or len(response.strip()) < 10:
        return {
            'score': 0,
            'feedback': 'Response too short. Please provide more detail.',
            'metrics': {}
        }
    
    metrics = {}
    
    # 1. Length analysis
    word_count = len(response.split())
    metrics['word_count'] = word_count
    length_score = min(20, (word_count / 10))  # Optimal: 100-200 words
    
    # 2. Readability
    try:
        flesch_score = textstat.flesch_reading_ease(response)
        metrics['readability'] = round(flesch_score, 1)
        readability_score = 15 if 60 <= flesch_score <= 80 else 10
    except:
        readability_score = 10
    
    # 3. Sentiment analysis
    sentiment_analyzer = get_sentiment_analyzer()
    sentiment_score = 0
    if sentiment_analyzer:
        try:
            sentiment = sentiment_analyzer(response[:512])[0]
            metrics['sentiment'] = sentiment['label']
            metrics['confidence'] = round(sentiment['score'], 2)
            # Positive sentiment indicates confidence
            sentiment_score = 15 if sentiment['label'] == 'POSITIVE' else 10
        except:
            sentiment_score = 10
    else:
        sentiment_score = 10
    
    # 4. Structure analysis (STAR method for behavioral questions)
    star_keywords = {
        'situation': ['situation', 'context', 'background', 'scenario'],
        'task': ['task', 'challenge', 'problem', 'goal', 'objective'],
        'action': ['action', 'did', 'implemented', 'developed', 'created', 'solved'],
        'result': ['result', 'outcome', 'achieved', 'improved', 'increased', 'success']
    }
    
    response_lower = response.lower()
    star_found = {key: any(kw in response_lower for kw in keywords) 
                  for key, keywords in star_keywords.items()}
    structure_score = sum(star_found.values()) * 5
    metrics['star_method'] = star_found
    
    # 5. Technical content (check for technical terms)
    technical_terms = extract_skills_intelligent(response)
    technical_score = min(20, len(technical_terms['technical']) * 3)
    metrics['technical_terms_found'] = len(technical_terms['technical'])
    
    # Total score
    total_score = length_score + readability_score + sentiment_score + structure_score + technical_score
    
    # Generate feedback
    feedback = []
    if word_count < 50:
        feedback.append("Provide more detailed responses (aim for 100-150 words)")
    if sum(star_found.values()) < 3:
        feedback.append("Use STAR method: Describe Situation, Task, Action, and Result")
    if technical_score < 10:
        feedback.append("Include relevant technical details and specific examples")
    if not feedback:
        feedback.append("Great response! Clear, detailed, and well-structured.")
    
    return {
        'score': round(min(100, total_score), 1),
        'grade': get_grade(total_score),
        'feedback': ' | '.join(feedback),
        'metrics': metrics
    }


# ============================================================================
# 5. PREDICTIVE ANALYTICS FOR INTERNSHIP SUCCESS
# ============================================================================

class InternshipSuccessPredictor:
    """
    ML model to predict internship application success
    """
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_features(self, user_data: Dict, internship_data: Dict) -> np.ndarray:
        """Extract features for prediction"""
        features = []
        
        # 1. Skills match score
        user_skills = user_data.get('skills', '').lower().split(',')
        required_skills = internship_data.get('skills_required', '').lower().split(',')
        user_skills = [s.strip() for s in user_skills if s.strip()]
        required_skills = [s.strip() for s in required_skills if s.strip()]
        
        match_result = enhanced_skill_matching(user_skills, required_skills)
        features.append(match_result['overall_score'])
        features.append(match_result['match_percentage'] / 100)
        
        # 2. Experience match
        years_required = internship_data.get('years_of_experience', 0)
        user_experience = user_data.get('experience', '')
        # Estimate years from experience text
        years_match = 1.0 if years_required == 0 else 0.5
        features.append(years_match)
        
        # 3. Education level
        education = user_data.get('education', '').lower()
        edu_score = 0.7
        if 'master' in education or 'phd' in education:
            edu_score = 1.0
        elif 'bachelor' in education or 'b.s' in education or 'b.e' in education:
            edu_score = 0.8
        features.append(edu_score)
        
        # 4. Certifications count
        certifications = user_data.get('certifications', '')
        cert_count = len([c for c in certifications.split(',') if c.strip()]) if certifications else 0
        features.append(min(1.0, cert_count / 3))
        
        # 5. Resume completeness
        required_fields = ['skills', 'experience', 'education', 'phone_number', 'email']
        completeness = sum(1 for f in required_fields if user_data.get(f)) / len(required_fields)
        features.append(completeness)
        
        # 6. Location match (same state/city)
        user_location = user_data.get('location', '').lower()
        job_location = internship_data.get('location', '').lower()
        location_match = 1.0 if user_location in job_location or job_location in user_location else 0.5
        features.append(location_match)
        
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data: pd.DataFrame):
        """Train the model with historical data"""
        if len(training_data) < 10:
            logging.warning("Insufficient training data for internship predictor")
            return False
        
        try:
            X = training_data.drop(['success'], axis=1)
            y = training_data['success']
            
            # Use XGBoost if available, otherwise RandomForest
            if _XGB_AVAILABLE and xgb:
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            else:
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
            
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            logging.info("Internship success predictor trained successfully")
            return True
        except Exception as e:
            logging.error(f"Error training predictor: {str(e)}")
            return False
    
    def predict_success_probability(self, user_data: Dict, internship_data: Dict) -> Dict:
        """Predict probability of internship application success"""
        features = self.extract_features(user_data, internship_data)
        
        if self.is_trained and self.model:
            try:
                features_scaled = self.scaler.transform(features)
                probability = self.model.predict_proba(features_scaled)[0][1]
                prediction = self.model.predict(features_scaled)[0]
            except:
                # Fallback to heuristic
                probability = features[0][0] * 0.4 + features[0][1] * 0.3 + features[0][4] * 0.3
                prediction = 1 if probability > 0.5 else 0
        else:
            # Heuristic-based prediction
            probability = features[0][0] * 0.4 + features[0][1] * 0.3 + features[0][4] * 0.3
            prediction = 1 if probability > 0.5 else 0
        
        confidence = "High" if probability > 0.7 or probability < 0.3 else "Medium"
        
        return {
            'success_probability': round(probability * 100, 1),
            'prediction': 'Likely' if prediction == 1 else 'Unlikely',
            'confidence': confidence,
            'recommendation': self._generate_recommendation(probability, features[0])
        }
    
    def _generate_recommendation(self, probability: float, features: np.ndarray) -> str:
        """Generate personalized recommendation"""
        if probability > 0.7:
            return "Strong match! Apply with confidence."
        elif probability > 0.5:
            return "Good match. Consider highlighting relevant projects in your application."
        elif probability > 0.3:
            return "Moderate match. Improve skills alignment or consider skill development."
        else:
            return "Skills gap detected. Focus on building required skills before applying."


# ============================================================================
# 6. PERSONALIZED LEARNING RECOMMENDATIONS
# ============================================================================

def generate_learning_path(user_skills: List[str], target_skills: List[str], 
                          career_goal: str = None) -> Dict:
    """
    Generate personalized learning recommendations
    """
    missing_skills = list(set(target_skills) - set(user_skills))
    
    if not missing_skills:
        return {
            'status': 'complete',
            'message': 'You have all required skills!',
            'recommendations': []
        }
    
    # Categorize missing skills
    categorized = {
        'beginner': [],
        'intermediate': [],
        'advanced': []
    }
    
    for skill in missing_skills:
        skill_lower = skill.lower()
        # Simple heuristic for difficulty
        if any(x in skill_lower for x in ['basic', 'intro', 'fundamental']):
            categorized['beginner'].append(skill)
        elif any(x in skill_lower for x in ['advanced', 'expert', 'architect']):
            categorized['advanced'].append(skill)
        else:
            categorized['intermediate'].append(skill)
    
    # Generate course recommendations
    recommendations = []
    
    for skill in missing_skills[:5]:  # Top 5 priorities
        courses = {
            'skill': skill,
            'resources': [
                {
                    'platform': 'Coursera',
                    'url': f'https://www.coursera.org/search?query={skill.replace(" ", "+")}',
                    'type': 'Online Course'
                },
                {
                    'platform': 'Udemy',
                    'url': f'https://www.udemy.com/courses/search/?q={skill.replace(" ", "+")}',
                    'type': 'Video Tutorial'
                },
                {
                    'platform': 'YouTube',
                    'url': f'https://www.youtube.com/results?search_query={skill.replace(" ", "+")}+tutorial',
                    'type': 'Free Tutorial'
                },
                {
                    'platform': 'Documentation',
                    'url': f'https://www.google.com/search?q={skill.replace(" ", "+")}+official+documentation',
                    'type': 'Official Docs'
                }
            ],
            'estimated_time': '2-4 weeks',
            'priority': 'High' if skill in target_skills[:3] else 'Medium'
        }
        recommendations.append(courses)
    
    return {
        'status': 'learning_path_generated',
        'missing_skills_count': len(missing_skills),
        'categorized_skills': categorized,
        'recommendations': recommendations,
        'estimated_total_time': f'{len(missing_skills) * 3} weeks'
    }


# ============================================================================
# 7. CONTENT QUALITY ANALYSIS
# ============================================================================

def analyze_text_quality(text: str) -> Dict:
    """
    Analyze text quality for resumes, cover letters, etc.
    """
    if not text or len(text.strip()) < 10:
        return {'score': 0, 'issues': ['Text too short']}
    
    issues = []
    score = 100
    
    # 1. Grammar and spelling (basic checks)
    if text != text.strip():
        issues.append("Remove extra whitespace")
        score -= 5
    
    # 2. Readability
    try:
        flesch = textstat.flesch_reading_ease(text)
        if flesch < 30:
            issues.append("Text is too complex. Use simpler language.")
            score -= 10
        elif flesch > 90:
            issues.append("Text may be too simple. Add more detail.")
            score -= 5
    except:
        pass
    
    # 3. Length appropriateness
    word_count = len(text.split())
    if word_count < 50:
        issues.append("Add more content (aim for 100+ words)")
        score -= 15
    
    # 4. Professional tone
    informal_words = ['gonna', 'wanna', 'yeah', 'cool', 'awesome', 'stuff', 'things']
    found_informal = [w for w in informal_words if w in text.lower()]
    if found_informal:
        issues.append(f"Use professional language (avoid: {', '.join(found_informal)})")
        score -= 10
    
    # 5. Action verbs presence (for experience sections)
    action_verbs = ['developed', 'created', 'managed', 'led', 'implemented', 'designed']
    has_action_verbs = any(verb in text.lower() for verb in action_verbs)
    if not has_action_verbs and 'experience' in text.lower():
        issues.append("Use strong action verbs (developed, created, managed, etc.)")
        score -= 10
    
    return {
        'score': max(0, score),
        'grade': get_grade(score),
        'issues': issues if issues else ['Excellent quality!'],
        'word_count': word_count
    }


# Global predictor instance
predictor = InternshipSuccessPredictor()

logging.info("ML utilities module loaded successfully")
