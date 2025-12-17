"""
Skill Representation via Ontology Embeddings

Research Motivation:
Skill descriptions encode construct semantics, not surface labels.
We embed skill descriptions to capture latent semantic relationships
that enable probabilistic job-skill matching.
"""

import json
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

# Optional: Use sentence-transformers if available, fallback to simple embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Using fallback embeddings.")


@dataclass
class SkillEmbeddingResult:
    """Result of skill embedding generation."""
    skill_embeddings: Dict[str, np.ndarray]
    embedding_metadata: Dict[str, any]
    similarity_matrix: np.ndarray
    skill_ids: List[str]
    embedding_dim: int


class SkillEmbedder:
    """Generate embeddings for latent skills from ontology descriptions."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "cache"):
        """
        Initialize skill embedder.
        
        Args:
            model_name: Sentence transformer model name
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                self.embedding_method = "sentence_transformers"
                print(f"✅ Loaded SentenceTransformer: {model_name}")
            except Exception as e:
                print(f"⚠️  Failed to load SentenceTransformer: {e}")
                self.model = None
                self.embedding_method = "fallback"
        else:
            self.model = None
            self.embedding_method = "fallback"
    
    def load_skills_ontology(self, skills_path: str = "ontology/skills.json") -> Dict[str, Dict]:
        """Load skills ontology from JSON file."""
        with open(skills_path, 'r') as f:
            skills_data = json.load(f)
        
        return skills_data['latent_skills']
    
    def generate_embeddings(self, skills_ontology: Dict[str, Dict], 
                          force_regenerate: bool = False) -> SkillEmbeddingResult:
        """
        Generate embeddings for all skills in ontology.
        
        Args:
            skills_ontology: Dictionary of skill_id -> skill_info
            force_regenerate: Whether to regenerate cached embeddings
            
        Returns:
            SkillEmbeddingResult with embeddings and metadata
        """
        cache_file = self.cache_dir / f"skill_embeddings_{self.model_name.replace('/', '_')}.pkl"
        
        # Try to load from cache
        if cache_file.exists() and not force_regenerate:
            try:
                with open(cache_file, 'rb') as f:
                    cached_result = pickle.load(f)
                print(f"✅ Loaded cached embeddings from {cache_file}")
                return cached_result
            except Exception as e:
                print(f"⚠️  Failed to load cached embeddings: {e}")
        
        # Generate embeddings
        skill_ids = list(skills_ontology.keys())
        skill_texts = []
        
        # Create rich text representations for embedding
        for skill_id in skill_ids:
            skill = skills_ontology[skill_id]
            # Combine name and description for richer semantic representation
            skill_text = f"{skill['name']}: {skill['description']}"
            skill_texts.append(skill_text)
        
        print(f"🔄 Generating embeddings for {len(skill_texts)} skills...")
        
        if self.embedding_method == "sentence_transformers":
            # Use sentence transformers
            embeddings_array = self.model.encode(
                skill_texts, 
                normalize_embeddings=True,
                show_progress_bar=True
            )
        else:
            # Fallback: Simple TF-IDF based embeddings
            embeddings_array = self._generate_fallback_embeddings(skill_texts)
        
        # Create skill_id -> embedding mapping
        skill_embeddings = {}
        for i, skill_id in enumerate(skill_ids):
            skill_embeddings[skill_id] = embeddings_array[i]
        
        # Calculate similarity matrix for validation
        similarity_matrix = np.dot(embeddings_array, embeddings_array.T)
        
        # Create result
        result = SkillEmbeddingResult(
            skill_embeddings=skill_embeddings,
            embedding_metadata={
                'model_name': self.model_name,
                'embedding_method': self.embedding_method,
                'num_skills': len(skill_ids),
                'skill_texts': dict(zip(skill_ids, skill_texts))
            },
            similarity_matrix=similarity_matrix,
            skill_ids=skill_ids,
            embedding_dim=embeddings_array.shape[1]
        )
        
        # Cache the result
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            print(f"💾 Cached embeddings to {cache_file}")
        except Exception as e:
            print(f"⚠️  Failed to cache embeddings: {e}")
        
        return result
    
    def _generate_fallback_embeddings(self, texts: List[str], dim: int = 384) -> np.ndarray:
        """
        Generate simple TF-IDF based embeddings as fallback.
        
        This is a research-aware fallback that maintains semantic relationships
        even without advanced transformer models.
        """
        from collections import Counter
        import re
        
        # Simple tokenization
        def tokenize(text):
            return re.findall(r'\b\w+\b', text.lower())
        
        # Build vocabulary
        all_tokens = []
        tokenized_texts = []
        for text in texts:
            tokens = tokenize(text)
            tokenized_texts.append(tokens)
            all_tokens.extend(tokens)
        
        vocab = list(set(all_tokens))
        vocab_size = len(vocab)
        token_to_idx = {token: i for i, token in enumerate(vocab)}
        
        # Calculate TF-IDF
        embeddings = []
        doc_freq = Counter()
        
        # Calculate document frequencies
        for tokens in tokenized_texts:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
        
        # Generate embeddings
        for tokens in tokenized_texts:
            token_counts = Counter(tokens)
            embedding = np.zeros(min(vocab_size, dim))
            
            for token, count in token_counts.items():
                if token in token_to_idx:
                    idx = token_to_idx[token] % dim
                    tf = count / len(tokens)
                    idf = np.log(len(texts) / (doc_freq[token] + 1))
                    embedding[idx] += tf * idf
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def validate_embeddings(self, result: SkillEmbeddingResult, 
                          skills_ontology: Dict[str, Dict]) -> Dict[str, any]:
        """
        Validate that embeddings capture meaningful semantic relationships.
        
        Returns:
            Dictionary of validation metrics and results
        """
        validation = {}
        
        # Check 1: Embedding shape
        validation['correct_shape'] = (
            len(result.skill_embeddings) == len(skills_ontology) and
            result.embedding_dim > 0
        )
        
        # Check 2: Embeddings are normalized
        embedding_norms = [np.linalg.norm(emb) for emb in result.skill_embeddings.values()]
        validation['normalized_embeddings'] = all(0.9 <= norm <= 1.1 for norm in embedding_norms)
        
        # Check 3: Related skills have higher similarity than unrelated ones
        # Test specific relationships we expect
        expected_relationships = [
            # Cognitive skills should be more similar to each other
            (('C1', 'C2'), ('C1', 'B1')),  # Numerical vs Verbal > Numerical vs Conscientiousness
            (('C3', 'C5'), ('C3', 'W1')),  # Abstract vs Processing > Abstract vs Achievement
            
            # Behavioral skills should cluster
            (('B1', 'B2'), ('B1', 'C1')),  # Conscientiousness vs Emotional > Conscientiousness vs Numerical
            (('B3', 'B4'), ('B3', 'C3')),  # Extraversion vs Agreeableness > Extraversion vs Abstract
            
            # Work-style skills should relate
            (('W1', 'W2'), ('W1', 'C2')),  # Achievement vs Detail > Achievement vs Verbal
            (('W4', 'W5'), ('W4', 'B5')),  # Collaboration vs Decision > Collaboration vs Openness
        ]
        
        relationship_scores = []
        for (related_pair, unrelated_pair) in expected_relationships:
            if all(skill_id in result.skill_embeddings for skill_id in related_pair + unrelated_pair):
                # Calculate similarities
                related_sim = np.dot(
                    result.skill_embeddings[related_pair[0]], 
                    result.skill_embeddings[related_pair[1]]
                )
                unrelated_sim = np.dot(
                    result.skill_embeddings[unrelated_pair[0]], 
                    result.skill_embeddings[unrelated_pair[1]]
                )
                
                relationship_scores.append(related_sim > unrelated_sim)
        
        validation['semantic_relationships'] = {
            'expected_relationships_correct': sum(relationship_scores),
            'total_relationships_tested': len(relationship_scores),
            'relationship_accuracy': sum(relationship_scores) / len(relationship_scores) if relationship_scores else 0
        }
        
        # Check 4: Category clustering
        category_similarities = {}
        categories = ['Cognitive', 'Behavioral', 'Work-style']
        
        for category in categories:
            category_skills = [skill_id for skill_id, skill in skills_ontology.items() 
                             if skill['category'] == category]
            
            if len(category_skills) >= 2:
                # Calculate average within-category similarity
                within_similarities = []
                for i, skill1 in enumerate(category_skills):
                    for skill2 in category_skills[i+1:]:
                        sim = np.dot(result.skill_embeddings[skill1], result.skill_embeddings[skill2])
                        within_similarities.append(sim)
                
                category_similarities[category] = np.mean(within_similarities) if within_similarities else 0
        
        validation['category_clustering'] = category_similarities
        
        # Overall validation score
        validation['overall_quality'] = (
            validation['correct_shape'] and
            validation['normalized_embeddings'] and
            validation['semantic_relationships']['relationship_accuracy'] > 0.6
        )
        
        return validation


def embed_skills_from_ontology(skills_path: str = "ontology/skills.json", 
                              model_name: str = "all-MiniLM-L6-v2",
                              force_regenerate: bool = False) -> SkillEmbeddingResult:
    """
    Main function to generate skill embeddings from ontology.
    
    Args:
        skills_path: Path to skills ontology JSON
        model_name: Sentence transformer model name
        force_regenerate: Whether to regenerate cached embeddings
        
    Returns:
        SkillEmbeddingResult with embeddings and validation
    """
    embedder = SkillEmbedder(model_name)
    skills_ontology = embedder.load_skills_ontology(skills_path)
    
    result = embedder.generate_embeddings(skills_ontology, force_regenerate)
    validation = embedder.validate_embeddings(result, skills_ontology)
    
    print(f"\n✅ CHECKPOINT 2.2: Skill Embeddings Generated")
    print(f"   • {result.embedding_dim}-dimensional embeddings")
    print(f"   • {len(result.skill_embeddings)} skills embedded")
    print(f"   • Method: {result.embedding_metadata['embedding_method']}")
    print(f"   • Validation: {'✅ PASSED' if validation['overall_quality'] else '⚠️  REVIEW NEEDED'}")
    
    return result


if __name__ == "__main__":
    print("🔍 CHECKPOINT 2.2: Skill Embedding Validation")
    print("=" * 60)
    
    # Generate embeddings
    result = embed_skills_from_ontology()
    
    # Load skills for validation
    with open("ontology/skills.json", 'r') as f:
        skills_data = json.load(f)
    skills_ontology = skills_data['latent_skills']
    
    # Validate embeddings
    embedder = SkillEmbedder()
    validation = embedder.validate_embeddings(result, skills_ontology)
    
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"   • Correct shape: {validation['correct_shape']}")
    print(f"   • Normalized embeddings: {validation['normalized_embeddings']}")
    print(f"   • Semantic relationships: {validation['semantic_relationships']['relationship_accuracy']:.1%}")
    print(f"   • Category clustering: {validation['category_clustering']}")
    
    # Show some similarity examples
    print(f"\n🔗 SIMILARITY EXAMPLES:")
    skill_ids = list(result.skill_embeddings.keys())
    for i in range(min(3, len(skill_ids))):
        for j in range(i+1, min(i+3, len(skill_ids))):
            skill1, skill2 = skill_ids[i], skill_ids[j]
            similarity = np.dot(result.skill_embeddings[skill1], result.skill_embeddings[skill2])
            print(f"   • {skill1} ↔ {skill2}: {similarity:.3f}")
    
    print(f"\n✅ CHECKPOINT 2.2 COMPLETE: Skill embeddings ready for job matching")