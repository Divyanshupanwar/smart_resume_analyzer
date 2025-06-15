from logic.sorting import SortingAlgorithms, KnapsackAlgorithm

class ResumeRanker:
    @staticmethod
    def calculate_resume_score(resume_data, resume_text, skills_match_score):
        """
        Calculate comprehensive score for a resume based on multiple factors
        """
        score = 0
        
        # Base score from resume parser
        if 'Objective' in resume_text:
            score += 20
        if 'Declaration' in resume_text:
            score += 20
        if 'Hobbies' in resume_text or 'Interests' in resume_text:
            score += 20
        if 'Achievements' in resume_text:
            score += 20
        if 'Projects' in resume_text:
            score += 20
            
        # Experience level score
        pages = resume_data['no_of_pages']
        if pages == 1:
            score += 10  # Fresher
        elif pages == 2:
            score += 20  # Intermediate
        else:
            score += 30  # Experienced
            
        # Skills match score (calculated externally)
        score += skills_match_score
        
        # Education score (if available)
        if 'education' in resume_data and resume_data['education']:
            score += 15
            
        # Experience score (if available)
        if 'experience' in resume_data and resume_data['experience']:
            # Add 5 points per year of experience, up to 25
            exp_years = min(5, len(resume_data['experience']))
            score += exp_years * 5
            
        return score

    @staticmethod
    def select_best_resumes(resumes, algorithm='merge_sort'):
        """
        Sort and select the best resumes using the specified algorithm
        The algorithm is automatically selected based on the data size if not specified:
        - merge_sort: Stable, good for small datasets
        - quick_sort: Fast, good for medium datasets
        - heap_sort: Consistent, good for large datasets
        """
        # Auto-select algorithm if not specified based on data size
        if algorithm == 'auto':
            if len(resumes) > 100:
                algorithm = 'heap_sort'  # Best for large datasets
            elif len(resumes) > 20:
                algorithm = 'quick_sort'  # Good for medium datasets
            else:
                algorithm = 'merge_sort'  # Stable for small datasets
        
        if algorithm == 'merge_sort':
            return SortingAlgorithms.merge_sort(resumes, key=lambda x: x['total_score'])
        elif algorithm == 'quick_sort':
            return SortingAlgorithms.quick_sort(resumes, key=lambda x: x['total_score'])
        elif algorithm == 'heap_sort':
            return SortingAlgorithms.heap_sort(resumes, key=lambda x: x['total_score'])
        else:
            # Default to merge sort if unknown algorithm is specified
            return SortingAlgorithms.merge_sort(resumes, key=lambda x: x['total_score'])
    
    @staticmethod
    def select_optimal_resumes(resumes, max_candidates=10, diversity_weight=0.3):
        """
        Use knapsack algorithm to select optimal set of resumes based on 
        score and diversity constraints
        """
        if not resumes:
            return []
            
        # Extract values (scores) and weights (diversity measures)
        values = [r['total_score'] for r in resumes]
        
        # Calculate diversity weights based on skills and fields
        # Higher weight means more diverse candidate (helps with team composition)
        weights = []
        skill_sets = [set(r['skills']) for r in resumes]
        fields = [r['predicted_field'] for r in resumes]
        
        for i, resume in enumerate(resumes):
            # Calculate diversity as a function of unique skills compared to others
            skill_diversity = sum(len(skill_sets[i] - skill_sets[j]) 
                                for j in range(len(resumes)) if i != j)
            
            # Field diversity (bonus for underrepresented fields)
            field_count = fields.count(resume['predicted_field'])
            field_diversity = max(1, 10 - field_count)  # Higher for rare fields
            
            # Combined diversity weight (normalize to range 1-10)
            diversity = 1 + (skill_diversity / 10) + field_diversity
            weights.append(int(diversity))
        
        # Use knapsack to select optimal combination
        capacity = max_candidates * 5  # Adjust capacity based on max candidates
        max_value, selected_indices = KnapsackAlgorithm.knapsack_dp(values, weights, capacity)
        
        # Return selected resumes
        return [resumes[i] for i in selected_indices]
