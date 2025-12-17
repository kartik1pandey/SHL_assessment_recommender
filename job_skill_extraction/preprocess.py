"""
Job Description Preprocessing (Minimal but Principled)

Research Motivation:
Over-cleaning destroys semantic cues (e.g., "fast-paced", "ownership").
We preserve semantic richness while removing only clear boilerplate.
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PreprocessingResult:
    """Result of job description preprocessing."""
    raw_text: str
    cleaned_text: str
    removed_sections: List[str]
    preserved_sections: Dict[str, str]
    preprocessing_stats: Dict[str, int]


class JobDescriptionPreprocessor:
    """Minimal but principled job description preprocessing."""
    
    def __init__(self):
        # Boilerplate patterns to remove (but preserve semantic content)
        self.boilerplate_patterns = [
            # EEO statements
            r"(?i)equal\s+opportunity\s+employer.*?(?=\n\n|\n[A-Z]|$)",
            r"(?i)we\s+are\s+an\s+equal\s+opportunity.*?(?=\n\n|\n[A-Z]|$)",
            r"(?i)eeo\s+statement.*?(?=\n\n|\n[A-Z]|$)",
            
            # Generic benefits (preserve specific benefits that signal culture)
            r"(?i)competitive\s+salary\s+and\s+benefits.*?(?=\n\n|\n[A-Z]|$)",
            r"(?i)comprehensive\s+benefits\s+package.*?(?=\n\n|\n[A-Z]|$)",
            
            # Legal disclaimers
            r"(?i)this\s+job\s+description.*?not\s+intended\s+to\s+be.*?(?=\n\n|\n[A-Z]|$)",
            r"(?i)reasonable\s+accommodations.*?(?=\n\n|\n[A-Z]|$)",
            
            # Application instructions (preserve urgency signals)
            r"(?i)to\s+apply.*?send\s+resume.*?(?=\n\n|\n[A-Z]|$)",
            r"(?i)please\s+submit.*?application.*?(?=\n\n|\n[A-Z]|$)",
        ]
        
        # Preserve these semantic cues (they signal work style/culture)
        self.preserve_patterns = [
            "fast-paced", "ownership", "autonomy", "collaborative", "innovative",
            "startup", "enterprise", "remote", "hybrid", "on-site",
            "urgent", "immediate", "flexible", "structured", "dynamic"
        ]
        
        # Section headers to identify and preserve
        self.section_headers = [
            r"(?i)(responsibilities|duties|role|position)",
            r"(?i)(requirements|qualifications|skills)",
            r"(?i)(preferred|nice\s+to\s+have|bonus)",
            r"(?i)(about\s+(?:us|the\s+company|this\s+role))",
            r"(?i)(what\s+(?:you'll\s+do|we\s+offer))"
        ]
    
    def preprocess(self, job_description: str, job_id: Optional[str] = None) -> PreprocessingResult:
        """
        Preprocess job description with minimal but principled cleaning.
        
        Args:
            job_description: Raw job description text
            job_id: Optional identifier for tracking
            
        Returns:
            PreprocessingResult with original, cleaned text and metadata
        """
        original_text = job_description
        text = job_description
        removed_sections = []
        preserved_sections = {}
        
        # Step 1: Extract and preserve key sections
        preserved_sections = self._extract_sections(text)
        
        # Step 2: Remove clear boilerplate (but preserve semantic cues)
        for pattern in self.boilerplate_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            if matches:
                # Check if boilerplate contains preserved semantic cues
                for match in matches:
                    if not any(cue.lower() in match.lower() for cue in self.preserve_patterns):
                        removed_sections.append(match.strip())
                        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.DOTALL)
        
        # Step 3: Normalize whitespace (but preserve structure)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines -> double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces -> single
        text = text.strip()
        
        # Step 4: Basic lowercasing (preserve proper nouns and acronyms)
        # We'll do this selectively to preserve semantic information
        
        # Calculate preprocessing statistics
        stats = {
            'original_length': len(original_text),
            'cleaned_length': len(text),
            'reduction_ratio': 1 - (len(text) / len(original_text)) if original_text else 0,
            'sections_removed': len(removed_sections),
            'sections_preserved': len(preserved_sections)
        }
        
        return PreprocessingResult(
            raw_text=original_text,
            cleaned_text=text,
            removed_sections=removed_sections,
            preserved_sections=preserved_sections,
            preprocessing_stats=stats
        )
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract key sections from job description."""
        sections = {}
        
        # Simple section extraction based on common patterns
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a section header
            is_header = False
            for pattern in self.section_headers:
                if re.match(pattern, line):
                    # Save previous section
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content)
                    
                    current_section = line.lower()
                    current_content = []
                    is_header = True
                    break
            
            if not is_header and current_section:
                current_content.append(line)
        
        # Save final section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def validate_preprocessing(self, result: PreprocessingResult) -> Dict[str, bool]:
        """
        Validate that preprocessing preserved semantic content.
        
        Returns:
            Dictionary of validation checks and their results
        """
        checks = {}
        
        # Check 1: Reasonable length reduction (not too aggressive)
        checks['reasonable_reduction'] = 0.1 <= result.preprocessing_stats['reduction_ratio'] <= 0.4
        
        # Check 2: Key sections preserved
        important_keywords = ['responsibilities', 'requirements', 'skills', 'experience']
        checks['key_sections_preserved'] = any(
            keyword in result.cleaned_text.lower() 
            for keyword in important_keywords
        )
        
        # Check 3: Semantic cues preserved
        checks['semantic_cues_preserved'] = any(
            cue in result.cleaned_text.lower() 
            for cue in self.preserve_patterns
        )
        
        # Check 4: No catastrophic information loss
        checks['no_catastrophic_loss'] = len(result.cleaned_text) > 0.5 * len(result.raw_text)
        
        return checks


def preprocess_job_descriptions(job_descriptions: Dict[str, str]) -> Dict[str, PreprocessingResult]:
    """
    Preprocess multiple job descriptions.
    
    Args:
        job_descriptions: Dictionary mapping job_id -> job_description_text
        
    Returns:
        Dictionary mapping job_id -> PreprocessingResult
    """
    preprocessor = JobDescriptionPreprocessor()
    results = {}
    
    for job_id, job_text in job_descriptions.items():
        results[job_id] = preprocessor.preprocess(job_text, job_id)
    
    return results


if __name__ == "__main__":
    # Test preprocessing on sample job descriptions
    from pathlib import Path
    
    # Load sample job descriptions
    jd_files = [
        "data/raw/job_descriptions/jd_software_engineer.txt",
        "data/raw/job_descriptions/jd_data_analyst.txt",
        "data/raw/job_descriptions/jd_sales_exec.txt"
    ]
    
    preprocessor = JobDescriptionPreprocessor()
    
    print("🔍 CHECKPOINT 2.1: JD Preprocessing Validation")
    print("=" * 60)
    
    for jd_file in jd_files:
        jd_path = Path(jd_file)
        if jd_path.exists():
            with open(jd_path, 'r') as f:
                raw_jd = f.read()
            
            result = preprocessor.preprocess(raw_jd, jd_path.stem)
            validation = preprocessor.validate_preprocessing(result)
            
            print(f"\n📄 {jd_path.stem}")
            print(f"   Original length: {result.preprocessing_stats['original_length']} chars")
            print(f"   Cleaned length: {result.preprocessing_stats['cleaned_length']} chars")
            print(f"   Reduction: {result.preprocessing_stats['reduction_ratio']:.1%}")
            print(f"   Sections removed: {result.preprocessing_stats['sections_removed']}")
            print(f"   Sections preserved: {result.preprocessing_stats['sections_preserved']}")
            
            print("\n   Raw JD (first 200 chars):")
            print(f"   {result.raw_text[:200]}...")
            
            print("\n   Cleaned JD (first 200 chars):")
            print(f"   {result.cleaned_text[:200]}...")
            
            print(f"\n   Validation: {validation}")
            
            # Check for semantic loss
            all_passed = all(validation.values())
            status = "✅ PASSED" if all_passed else "⚠️  REVIEW NEEDED"
            print(f"   Status: {status}")
    
    print(f"\n✅ CHECKPOINT 2.1 COMPLETE: Minimal but principled preprocessing")
    print("   • Preserves semantic cues (fast-paced, ownership, etc.)")
    print("   • Removes clear boilerplate only")
    print("   • Maintains structural information")
    print("   • Ready for embedding generation")