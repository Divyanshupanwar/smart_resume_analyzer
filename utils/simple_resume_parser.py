import os
import io
import re
import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class SimpleResumeParser:
    """
    A simple resume parser that doesn't rely on pyresparser or spaCy
    """
    def __init__(self, resume_path):
        self.resume_path = resume_path
        
    def extract_text_from_pdf(self):
        """Extract text from PDF file"""
        text = ""
        try:
            with open(self.resume_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
        return text
    
    def extract_name(self, text):
        """Extract name from resume text"""
        # Simple heuristic: First line is often the name
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and len(line.split()) <= 5:  # Names are usually short
                return line
        return "Name not found"
    
    def extract_email(self, text):
        """Extract email from resume text"""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, text)
        return emails[0] if emails else "Email not found"
    
    def extract_phone(self, text):
        """Extract phone number from resume text"""
        phone_pattern = r'(?:(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})(?: *x(\d+))?)'
        phones = re.findall(phone_pattern, text)
        if phones:
            # Format the first match
            match = phones[0]
            if match[0]:  # Country code exists
                return f"+{match[0]} {match[1]}-{match[2]}-{match[3]}"
            else:
                return f"{match[1]}-{match[2]}-{match[3]}"
        return "Phone number not found"
    
    def extract_skills(self, text):
        """Extract skills from resume text"""
        # List of common skills
        skills_list = [
            'python', 'java', 'c++', 'c#', 'javascript', 'react', 'angular', 'vue', 'node.js',
            'django', 'flask', 'spring', 'express', 'html', 'css', 'bootstrap', 'jquery',
            'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'nosql', 'firebase',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github',
            'machine learning', 'deep learning', 'artificial intelligence', 'data science',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
            'tableau', 'power bi', 'excel', 'word', 'powerpoint', 'photoshop', 'illustrator',
            'figma', 'sketch', 'adobe xd', 'ui/ux', 'responsive design',
            'agile', 'scrum', 'kanban', 'jira', 'confluence', 'project management',
            'leadership', 'teamwork', 'communication', 'problem solving', 'critical thinking',
            'android', 'ios', 'swift', 'kotlin', 'flutter', 'react native',
            'devops', 'ci/cd', 'testing', 'qa', 'selenium', 'junit', 'pytest'
        ]
        
        # Normalize text
        text_lower = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text_lower)
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
        
        # Extract skills
        found_skills = []
        for skill in skills_list:
            if skill in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def count_pages(self):
        """Count the number of pages in a PDF"""
        try:
            with open(self.resume_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return len(pdf_reader.pages)
        except Exception as e:
            print(f"Error counting pages: {e}")
            return 0
    
    def get_extracted_data(self):
        """Extract data from resume"""
        if not os.path.exists(self.resume_path):
            return None
            
        # Extract text from PDF
        text = self.extract_text_from_pdf()
        
        if not text:
            return None
        
        # Extract information
        name = self.extract_name(text)
        email = self.extract_email(text)
        phone = self.extract_phone(text)
        skills = self.extract_skills(text)
        pages = self.count_pages()
        
        # Create data dictionary
        data = {
            'name': name,
            'email': email,
            'mobile_number': phone,
            'skills': skills,
            'no_of_pages': pages
        }
        
        return data