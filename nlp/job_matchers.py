import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from collections import Counter
import string
import logging

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize spaCy model
try:
    nlp = spacy.load('en_core_web_md')  # Medium-sized model with word vectors
except OSError:
    logging.warning("Downloading spaCy model. This may take a while...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])
    nlp = spacy.load('en_core_web_md')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

class ResumeKeywordExtractor:
    """
    Advanced keyword extraction from resumes using NLP techniques
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Add custom stop words relevant to resumes
        self.stop_words.update(['experience', 'skill', 'skills', 'year', 'years', 'month', 'months',
                               'responsible', 'responsibility', 'responsibilities', 'work', 'worked'])
        
    def preprocess_text(self, text):
        """
        Preprocess text by removing punctuation, converting to lowercase,
        tokenizing, removing stop words, and lemmatizing
        """
        # Remove punctuation and convert to lowercase
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(filtered_tokens)
    
    def extract_keywords(self, text, n=20):
        """
        Extract the most important keywords from text using TF-IDF
        """
        preprocessed_text = self.preprocess_text(text)
        
        # Create a document-term matrix with TF-IDF
        vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([preprocessed_text])
        
        # Get feature names and their TF-IDF scores
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # Create a dictionary of feature names and their scores
        word_scores = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names))}
        
        # Sort by score and return top n keywords
        sorted_keywords = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:n]
    
    def extract_named_entities(self, text):
        """
        Extract named entities (organizations, locations, etc.) from text
        """
        doc = nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            if ent.text not in entities[ent.label_]:
                entities[ent.label_].append(ent.text)
        
        return entities

class SkillsExtractor:
    """
    Extract and match skills from resumes and job descriptions
    """
    def __init__(self, skills_db_path=None):
        """
        Initialize with a skills database if provided
        """
        self.skills_patterns = self._load_skills_patterns(skills_db_path)
        
    def _load_skills_patterns(self, skills_db_path):
        """
        Load skills patterns from a database or use default patterns
        """
        # Default technical skills patterns
        default_patterns = {
            'programming_languages': [
                'python', 'java', 'javascript', 'c\+\+', 'c#', 'ruby', 'php', 'swift', 'kotlin',
                'typescript', 'scala', 'perl', 'go', 'rust', 'r programming', 'matlab'
            ],
            'web_technologies': [
                'html', 'css', 'react', 'angular', 'vue', 'node\.?js', 'express\.?js', 'django',
                'flask', 'spring boot', 'asp\.net', 'laravel', 'jquery', 'bootstrap', 'tailwind'
            ],
            'databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'sqlite', 'redis', 'cassandra',
                'dynamodb', 'firebase', 'neo4j', 'elasticsearch'
            ],
            'cloud_platforms': [
                'aws', 'amazon web services', 'azure', 'google cloud', 'gcp', 'heroku', 'digitalocean',
                'kubernetes', 'docker', 'terraform', 'jenkins', 'ci/cd'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'artificial intelligence', 'ai', 'ml', 'nlp',
                'natural language processing', 'computer vision', 'data mining', 'pandas', 'numpy',
                'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'tableau', 'power bi'
            ],
            'soft_skills': [
                'communication', 'teamwork', 'leadership', 'problem solving', 'critical thinking',
                'time management', 'adaptability', 'creativity', 'collaboration', 'presentation'
            ]
        }
        
        # If a skills database is provided, load it
        if skills_db_path:
            try:
                # Implement loading from external source (CSV, JSON, etc.)
                pass
            except Exception as e:
                logging.warning(f"Failed to load skills database: {e}. Using default patterns.")
        
        return default_patterns
    
    def extract_skills(self, text):
        """
        Extract skills from text using pattern matching and NLP
        """
        text = text.lower()
        skills = {}
        
        # Extract skills using pattern matching
        for category, patterns in self.skills_patterns.items():
            skills[category] = []
            for pattern in patterns:
                matches = re.findall(r'\b' + pattern + r'\b', text)
                if matches:
                    skills[category].extend(matches)
            
            # Remove duplicates
            skills[category] = list(set(skills[category]))
        
        # Use spaCy for additional skill extraction
        doc = nlp(text)
        
        # Extract noun phrases as potential skills
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
        
        # Filter noun phrases to find potential skills not caught by patterns
        additional_skills = []
        for phrase in noun_phrases:
            # Check if the phrase contains skill-related words
            if any(skill_word in phrase for skill_word in ['skill', 'proficient', 'experience', 'knowledge']):
                # Extract the actual skill (usually follows the skill-related word)
                for skill_word in ['skill', 'proficient', 'experience', 'knowledge']:
                    if skill_word in phrase:
                        skill_parts = phrase.split(skill_word)
                        if len(skill_parts) > 1 and skill_parts[1].strip():
                            additional_skills.append(skill_parts[1].strip())
        
        skills['additional'] = list(set(additional_skills))
        
        # Flatten the skills dictionary for easier use
        all_skills = []
        for category_skills in skills.values():
            all_skills.extend(category_skills)
        
        return {'categorized': skills, 'all': list(set(all_skills))}
    
    def match_skills(self, resume_text, job_description):
        """
        Match skills between a resume and job description
        Returns matched skills, missing skills, and a match score
        """
        resume_skills = self.extract_skills(resume_text)['all']
        job_skills = self.extract_skills(job_description)['all']
        
        # Find matched and missing skills
        matched_skills = [skill for skill in resume_skills if any(
            self._skill_similarity(skill, job_skill) > 0.8 for job_skill in job_skills
        )]
        
        missing_skills = [skill for skill in job_skills if not any(
            self._skill_similarity(skill, resume_skill) > 0.8 for resume_skill in resume_skills
        )]
        
        # Calculate match score
        if not job_skills:
            match_score = 0
        else:
            match_score = (len(matched_skills) / len(job_skills)) * 100
        
        return {
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'match_score': match_score
        }
    
    def _skill_similarity(self, skill1, skill2):
        """
        Calculate similarity between two skills using spaCy's word vectors
        """
        # Handle multi-word skills
        if ' ' in skill1 or ' ' in skill2:
            doc1 = nlp(skill1)
            doc2 = nlp(skill2)
            return doc1.similarity(doc2)
        else:
            # For single words, use token similarity
            token1 = nlp(skill1)[0]
            token2 = nlp(skill2)[0]
            return token1.similarity(token2)

class ExperienceAnalyzer:
    """
    Analyze work experience from resumes
    """
    def __init__(self):
        # Patterns for experience extraction
        self.experience_patterns = [
            r'(\d+)[\+]?\s+years?\s+(?:of\s+)?experience',
            r'experience\s+(?:of\s+)?(\d+)[\+]?\s+years?',
            r'worked\s+(?:for\s+)?(\d+)[\+]?\s+years?',
            r'(\d+)[\+]?\s+years?\s+(?:in\s+)?(?:the\s+)?(?:industry|field)',
        ]
        
        # Job title patterns
        self.job_title_patterns = [
            r'(senior|lead|principal|staff|junior|associate)\s+([\w\s]+)',
            r'(software|data|web|mobile|cloud|devops|security|network|systems|ui|ux)\s+([\w\s]+)',
            r'(developer|engineer|architect|analyst|scientist|designer|manager|administrator|specialist)'
        ]
    
    def extract_experience_years(self, text):
        """
        Extract years of experience from text
        """
        text = text.lower()
        years = []
        
        for pattern in self.experience_patterns:
            matches = re.findall(pattern, text)
            years.extend([int(year) for year in matches if year.isdigit()])
        
        if years:
            return max(years)  # Return the maximum years mentioned
        
        # If no explicit years found, try to calculate from work history
        return self._calculate_experience_from_history(text)
    
    def _calculate_experience_from_history(self, text):
        """
        Calculate years of experience from work history dates
        """
        # Look for date ranges like "2018-2022" or "Jan 2018 - Dec 2022"
        date_ranges = re.findall(r'(\d{4})\s*[-–—]\s*(\d{4}|\bpresent\b)', text, re.IGNORECASE)
        
        if not date_ranges:
            # Try month-year format
            date_ranges = re.findall(r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})\s*[-–—]\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4}|\bpresent\b)', text, re.IGNORECASE)
        
        total_years = 0
        current_year = 2023  # Use current year as default for "present"
        
        for start, end in date_ranges:
            start_year = int(start)
            end_year = current_year if end.lower() == 'present' else int(end)
            
            if start_year <= end_year:  # Sanity check
                total_years += (end_year - start_year)
        
        return total_years
    
    def extract_job_titles(self, text):
        """
        Extract job titles from text
        """
        text = text.lower()
        titles = []
        
        # Use spaCy for job title extraction
        doc = nlp(text)
        
        # Look for job title patterns
        for pattern in self.job_title_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    titles.append(' '.join(match).strip())
                else:
                    titles.append(match.strip())
        
        # Use noun chunks that follow company names as potential job titles
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                # Look for noun chunks after organization names
                for chunk in doc.noun_chunks:
                    if chunk.start > ent.end and chunk.start - ent.end < 5:  # Within 5 tokens
                        if any(job_word in chunk.text.lower() for job_word in ['engineer', 'developer', 'manager', 'analyst', 'designer', 'architect']):
                            titles.append(chunk.text.strip())
        
        # Remove duplicates and return
        return list(set(titles))
    
    def analyze_experience_relevance(self, resume_text, job_description):
        """
        Analyze how relevant the experience in the resume is to the job description
        """
        # Extract job titles from both resume and job description
        resume_titles = self.extract_job_titles(resume_text)
        job_titles = self.extract_job_titles(job_description)
        
        # Calculate title similarity
        title_similarities = []
        for resume_title in resume_titles:
            for job_title in job_titles:
                resume_doc = nlp(resume_title)
                job_doc = nlp(job_title)
                similarity = resume_doc.similarity(job_doc)
                title_similarities.append(similarity)
        
        # Calculate average title similarity
        avg_title_similarity = sum(title_similarities) / len(title_similarities) if title_similarities else 0
        
        # Extract years of experience
        years_experience = self.extract_experience_years(resume_text)
        
        # Calculate overall experience relevance score (0-100)
        relevance_score = avg_title_similarity * 100
        
        return {
            'years_experience': years_experience,
            'resume_job_titles': resume_titles,
            'job_description_titles': job_titles,
            'title_similarity': avg_title_similarity,
            'experience_relevance_score': relevance_score
        }

class EducationAnalyzer:
    """
    Analyze education information from resumes
    """
    def __init__(self):
        # Degree patterns
        self.degree_patterns = [
            r'\b(ph\.?d\.?|doctor of philosophy)\b',
            r'\b(m\.?s\.?|master of science)\b',
            r'\b(m\.?b\.?a\.?|master of business administration)\b',
            r'\b(b\.?s\.?|bachelor of science)\b',
            r'\b(b\.?a\.?|bachelor of arts)\b',
            r'\b(b\.?tech\.?|bachelor of technology)\b',
            r'\b(m\.?tech\.?|master of technology)\b',
            r'\b(b\.?e\.?|bachelor of engineering)\b',
            r'\b(m\.?e\.?|master of engineering)\b',
            r'\b(b\.?c\.?a\.?|bachelor of computer applications)\b',
            r'\b(m\.?c\.?a\.?|master of computer applications)\b',
            r'\b(associate\'?s? degree)\b',
            r'\b(high school diploma)\b',
            r'\b(certificate)\b'
        ]
        
        # Field of study patterns
        self.field_patterns = [
            r'in\s+([\w\s]+)',
            r'of\s+([\w\s]+)',
            r'(computer science|information technology|data science|artificial intelligence|machine learning|software engineering|electrical engineering|mechanical engineering|civil engineering|business administration|finance|marketing|economics|mathematics|statistics|physics|chemistry|biology)'
        ]
        
        # University patterns
        self.university_patterns = [
            r'(university|college|institute|school) of ([\w\s]+)',
            r'([\w\s]+) (university|college|institute|school)'
        ]
    
    def extract_education(self, text):
        """
        Extract education information from text
        """
        text = text.lower()
        education = []
        
        # Use spaCy for education extraction
        doc = nlp(text)
        
        # Extract degrees
        degrees = []
        for pattern in self.degree_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            degrees.extend(matches)
        
        # Extract fields of study
        fields = []
        for degree in degrees:
            # Look for field patterns after degree mentions
            degree_idx = text.find(degree)
            if degree_idx != -1:
                after_degree = text[degree_idx + len(degree):degree_idx + len(degree) + 100]  # Look 100 chars ahead
                for pattern in self.field_patterns:
                    field_matches = re.findall(pattern, after_degree, re.IGNORECASE)
                    fields.extend(field_matches)
        
        # If no fields found with the above method, try general extraction
        if not fields:
            for pattern in self.field_patterns:
                field_matches = re.findall(pattern, text, re.IGNORECASE)
                fields.extend(field_matches)
        
        # Extract universities/institutions
        universities = []
        for pattern in self.university_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    universities.append(' '.join(match).strip())
                else:
                    universities.append(match.strip())
        
        # Also use spaCy's named entity recognition for organizations
        for ent in doc.ents:
            if ent.label_ == 'ORG' and any(edu_word in ent.text.lower() for edu_word in ['university', 'college', 'institute', 'school']):
                universities.append(ent.text)
        
        # Combine the information
        for i, degree in enumerate(degrees):
            edu_entry = {'degree': degree}
            
            if i < len(fields):
                edu_entry['field'] = fields[i]
            
            if i < len(universities):
                edu_entry['institution'] = universities[i]
            
            education.append(edu_entry)
        
        # If we have more universities than degrees, add them as separate entries
        if len(universities) > len(degrees):
            for uni in universities[len(degrees):]:
                education.append({'institution': uni})
        
        return education
    
    def analyze_education_relevance(self, resume_text, job_description):
        """
        Analyze how relevant the education in the resume is to the job description
        """
        # Extract education from resume
        resume_education = self.extract_education(resume_text)
        
        # Extract required education from job description
        job_education = self.extract_education(job_description)
        
        # Calculate education match score
        education_score = 0
        
        # Check if degrees match
        resume_degrees = [edu.get('degree', '').lower() for edu in resume_education if 'degree' in edu]
        job_degrees = [edu.get('degree', '').lower() for edu in job_education if 'degree' in edu]
        
        # Check if fields match
        resume_fields = [edu.get('field', '').lower() for edu in resume_education if 'field' in edu]
        job_fields = [edu.get('field', '').lower() for edu in job_education if 'field' in edu]
        
        # Calculate degree match
        degree_match = False
        for resume_degree in resume_degrees:
            for job_degree in job_degrees:
                # Check if resume degree is equal or higher than job degree
                if self._is_degree_sufficient(resume_degree, job_degree):
                    degree_match = True
                    break
        
        # Calculate field match
        field_match = False
        for resume_field in resume_fields:
            for job_field in job_fields:
                resume_doc = nlp(resume_field)
                job_doc = nlp(job_field)
                if resume_doc.similarity(job_doc) > 0.7:  # High similarity threshold
                    field_match = True
                    break
        
        # Calculate education score
        if degree_match and field_match:
            education_score = 100
        elif degree_match:
            education_score = 75
        elif field_match:
            education_score = 50
        else:
            education_score = 25
        
        return {
            'resume_education': resume_education,
            'job_required_education': job_education,
            'degree_match': degree_match,
            'field_match': field_match,
            'education_relevance_score': education_score
        }
    
    def _is_degree_sufficient(self, resume_degree, job_degree):
        """
        Check if the resume degree is sufficient for the job degree requirement
        """
        # Define degree hierarchy (higher index = higher degree)
        degree_hierarchy = [
            'high school diploma',
            'certificate',
            'associate',
            'bachelor', 'b.s', 'b.a', 'b.tech', 'b.e', 'b.c.a',
            'master', 'm.s', 'm.b.a', 'm.tech', 'm.e', 'm.c.a',
            'ph.d', 'doctor'
        ]
        
        # Find the level of each degree
        resume_level = -1
        job_level = -1
        
        for i, degree_type in enumerate(degree_hierarchy):
            if degree_type in resume_degree:
                resume_level = i
            if degree_type in job_degree:
                job_level = i
        
        # If we couldn't determine the level, assume it's not sufficient
        if resume_level == -1 or job_level == -1:
            return False
        
        # Resume degree is sufficient if it's at or above the job requirement
        return resume_level >= job_level

class ResumeJobMatcher:
    """
    Main class for matching resumes to job descriptions using advanced NLP
    """
    def __init__(self):
        self.keyword_extractor = ResumeKeywordExtractor()
        self.skills_extractor = SkillsExtractor()
        self.experience_analyzer = ExperienceAnalyzer()
        self.education_analyzer = EducationAnalyzer()
    
    def preprocess_documents(self, resume_text, job_description):
        """
        Preprocess both resume and job description
        """
        # Clean and normalize text
        resume_text = self._clean_text(resume_text)
        job_description = self._clean_text(job_description)
        
        return resume_text, job_description
    
    def _clean_text(self, text):
        """
        Clean and normalize text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.\-\+\@\/]', ' ', text)
        
        return text.strip()
    
    def calculate_document_similarity(self, resume_text, job_description):
        """
        Calculate overall document similarity using TF-IDF and cosine similarity
        """
        # Preprocess documents
        resume_text, job_description = self.preprocess_documents(resume_text, job_description)
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        
        # Create document-term matrix
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return cosine_sim * 100  # Convert to percentage
    
    def match_resume_to_job(self, resume_text, job_description):
        """
        Comprehensive matching of resume to job description
        """
        # Preprocess documents
        resume_text, job_description = self.preprocess_documents(resume_text, job_description)
        
        # Extract keywords
        resume_keywords = self.keyword_extractor.extract_keywords(resume_text)
        job_keywords = self.keyword_extractor.extract_keywords(job_description)
        
        # Match skills
        skills_match = self.skills_extractor.match_skills(resume_text, job_description)
        
        # Analyze experience
        experience_analysis = self.experience_analyzer.analyze_experience_relevance(resume_text, job_description)
        
        # Analyze education
        education_analysis = self.education_analyzer.analyze_education_relevance(resume_text, job_description)
        
        # Calculate document similarity
        document_similarity = self.calculate_document_similarity(resume_text, job_description)
        
        # Calculate overall match score (weighted average)
        overall_score = (
            skills_match['match_score'] * 0.4 +  # Skills are most important
            experience_analysis['experience_relevance_score'] * 0.3 +  # Experience is second
            education_analysis['education_relevance_score'] * 0.2 +  # Education is third
            document_similarity * 0.1  # Overall similarity is least important
        )
        
        # Prepare detailed report
        match_report = {
            'overall_match_score': overall_score,
            'skills_match': skills_match,
            'experience_match': experience_analysis,
            'education_match': education_analysis,
            'document_similarity': document_similarity,
            'resume_keywords': resume_keywords,
            'job_keywords': job_keywords,
            'recommendations': self._generate_recommendations(skills_match, experience_analysis, education_analysis)
        }
        
        return match_report
    
    def _generate_recommendations(self, skills_match, experience_analysis, education_analysis):
        """
        Generate recommendations based on the match analysis
        """
        recommendations = []
        
        # Skills recommendations
        if skills_match['missing_skills']:
            recommendations.append({
                'category': 'Skills',
                'recommendation': f"Consider adding these missing skills to your resume: {', '.join(skills_match['missing_skills'][:5])}"
            })
        
        # Experience recommendations
        if experience_analysis['experience_relevance_score'] < 70:
            recommendations.append({
                'category': 'Experience',
                'recommendation': "Your experience doesn't strongly match the job requirements. Consider highlighting relevant projects or responsibilities that align with the job description."
            })
        
        # Education recommendations
        if education_analysis['education_relevance_score'] < 50:
            recommendations.append({
                'category': 'Education',
                'recommendation': "Your education background may not fully meet the job requirements. Consider highlighting relevant coursework or certifications."
            })
        
        # If no specific recommendations, add a general one
        if not recommendations:
            recommendations.append({
                'category': 'General',
                'recommendation': "Your resume appears to be a good match for this job. Consider customizing your cover letter to highlight your most relevant qualifications."
            })
        
        return recommendations

class ATSOptimizer:
    """
    Optimize resumes for Applicant Tracking Systems
    """
    def __init__(self):
        self.matcher = ResumeJobMatcher()
        self.keyword_extractor = ResumeKeywordExtractor()
    
    def analyze_ats_compatibility(self, resume_text):
        """
        Analyze how well a resume will perform with ATS systems
        """
        # Check for common ATS issues
        issues = []
        
        # Check for tables (ATS often struggle with tables)
        if re.search(r'<table|<tr|<td', resume_text, re.IGNORECASE):
            issues.append({
                'type': 'formatting',
                'issue': 'Tables detected',
                'recommendation': 'Replace tables with bullet points or plain text formatting.'
            })
        
        # Check for images (ATS often can't read text in images)
        if re.search(r'<img|\.jpg|\.png|\.gif', resume_text, re.IGNORECASE):
            issues.append({
                'type': 'formatting',
                'issue': 'Images detected',
                'recommendation': 'Ensure no important information is contained only in images.'
            })
        
        # Check for unusual characters or symbols
        unusual_chars = re.findall(r'[^\x00-\x7F]+', resume_text)
        if unusual_chars:
            issues.append({
                'type': 'formatting',
                'issue': 'Unusual characters detected',
                'recommendation': 'Replace special characters with standard ASCII characters.'
            })
        
        # Check for contact information
        if not re.search(r'[\w\.-]+@[\w\.-]+\.\w+', resume_text):  # Email
            issues.append({
                'type': 'content',
                'issue': 'Email address not detected',
                'recommendation': 'Include a professional email address in your contact information.'
            })
        
        if not re.search(r'\b(?:\+\d{1,3}[-\s]?)?$$?\d{3}$$?[-\s]?\d{3}[-\s]?\d{4}\b', resume_text):  # Phone
            issues.append({
                'type': 'content',
                'issue': 'Phone number not detected',
                'recommendation': 'Include a phone number in your contact information.'
            })
        
        # Check for section headers
        important_sections = ['experience', 'education', 'skills']
        missing_sections = []
        
        for section in important_sections:
            if not re.search(r'\b' + section + r'\b', resume_text, re.IGNORECASE):
                missing_sections.append(section)
        
        if missing_sections:
            issues.append({
                'type': 'structure',
                'issue': f"Missing important sections: {', '.join(missing_sections)}",
                'recommendation': 'Include clearly labeled sections for Experience, Education, and Skills.'
            })
        
        # Check for keyword density
        keywords = self.keyword_extractor.extract_keywords(resume_text, n=10)
        keyword_text = ', '.join([kw[0] for kw in keywords])
        
        # Calculate compatibility score
        compatibility_score = 100 - (len(issues) * 10)  # Deduct 10 points per issue
        compatibility_score = max(0, compatibility_score)  # Ensure score is not negative
        
        return {
            'compatibility_score': compatibility_score,
            'issues': issues,
            'top_keywords': keyword_text
        }
    
    def optimize_for_job(self, resume_text, job_description):
        """
        Provide recommendations to optimize a resume for a specific job
        """
        # Match resume to job
        match_report = self.matcher.match_resume_to_job(resume_text, job_description)
        
        # Analyze ATS compatibility
        ats_analysis = self.analyze_ats_compatibility(resume_text)
        
        # Generate optimization recommendations
        optimizations = []
        
        # Add missing skills
        if match_report['skills_match']['missing_skills']:
            optimizations.append({
                'category': 'Skills',
                'recommendation': f"Add these keywords from the job description: {', '.join(match_report['skills_match']['missing_skills'][:5])}",
                'importance': 'High'
            })
        
        # Improve keyword match
        job_keywords = [kw[0] for kw in match_report['job_keywords']]
        resume_keywords = [kw[0] for kw in match_report['resume_keywords']]
        
        missing_keywords = [kw for kw in job_keywords if kw not in resume_keywords]
        if missing_keywords:
            optimizations.append({
                'category': 'Keywords',
                'recommendation': f"Include these important keywords from the job description: {', '.join(missing_keywords[:5])}",
                'importance': 'High'
            })
        
        # Add ATS compatibility issues
        for issue in ats_analysis['issues']:
            optimizations.append({
                'category': 'ATS Compatibility',
                'recommendation': issue['recommendation'],
                'importance': 'Medium'
            })
        
        # Calculate optimization potential (how much improvement is possible)
        current_match = match_report['overall_match_score']
        optimization_potential = min(100 - current_match, 30)  # Max 30% improvement
        
        return {
            'current_match_score': current_match,
            'ats_compatibility_score': ats_analysis['compatibility_score'],
            'optimization_potential': optimization_potential,
            'optimizations': optimizations,
            'detailed_match_report': match_report
        }

# Function to integrate with the existing resume scoring system
def enhance_resume_score(resume_text, job_description=None):
    """
    Enhance the existing resume score with advanced NLP analysis
    """
    # Initialize analyzers
    keyword_extractor = ResumeKeywordExtractor()
    skills_extractor = SkillsExtractor()
    experience_analyzer = ExperienceAnalyzer()
    education_analyzer = EducationAnalyzer()
    ats_optimizer = ATSOptimizer()
    
    # Extract basic information
    keywords = keyword_extractor.extract_keywords(resume_text)
    skills = skills_extractor.extract_skills(resume_text)
    experience_years = experience_analyzer.extract_experience_years(resume_text)
    education = education_analyzer.extract_education(resume_text)
    ats_compatibility = ats_optimizer.analyze_ats_compatibility(resume_text)
    
    # Calculate enhanced score components
    skill_score = min(len(skills['all']), 20) * 5  # Up to 100 points for skills
    experience_score = min(experience_years, 10) * 10  # Up to 100 points for experience
    education_score = len(education) * 25  # 25 points per education entry, up to 100
    ats_score = ats_compatibility['compatibility_score']  # 0-100 based on ATS compatibility
    
    # If job description is provided, include job matching
    job_match_score = 0
    if job_description:
        matcher = ResumeJobMatcher()
        match_report = matcher.match_resume_to_job(resume_text, job_description)
        job_match_score = match_report['overall_match_score']
    
    # Calculate weighted total score
    if job_description:
        total_score = (
            skill_score * 0.2 +
            experience_score * 0.2 +
            education_score * 0.1 +
            ats_score * 0.2 +
            job_match_score * 0.3
        )
    else:
        total_score = (
            skill_score * 0.3 +
            experience_score * 0.3 +
            education_score * 0.2 +
            ats_score * 0.2
        )
    
    # Prepare detailed report
    report = {
        'total_score': total_score,
        'skill_score': skill_score,
        'experience_score': experience_score,
        'education_score': education_score,
        'ats_compatibility_score': ats_score,
        'extracted_skills': skills,
        'extracted_keywords': [kw[0] for kw in keywords],
        'experience_years': experience_years,
        'education': education,
        'ats_issues': ats_compatibility['issues']
    }
    
    if job_description:
        report['job_match_score'] = job_match_score
    
    return report
