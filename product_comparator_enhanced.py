import spacy
import re
import requests
from rapidfuzz import fuzz
from prettytable import PrettyTable
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os
import json
from typing import Dict, List, Tuple, Optional
import numpy as np

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---
# ENHANCED PRODUCT COMPARATOR
# Combines multiple extraction methods with confidence scoring
# ---

class EnhancedProductComparator:
    def __init__(self, model_path="ner_model_improved"):
        """Initialize the enhanced comparator with all components"""
        self.nlp = self.load_ner_model(model_path)
        self.semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.confidence_threshold = 0.7
        
        # Regex patterns for different fields
        self.patterns = {
            'grade': [
                r'(fe[\s_]?500[d]?|\b43\b|\b53\b|\b415\b|\b550\b)',
                r'(class\s+[iv]+)',
                r'(opc\s+\d+)'
            ],
            'diameter': [
                r'(\d{1,3}\.?\d*)\s?mm',
                r'(\d{1,3}\.?\d*)\s?millimeter'
            ],
            'length': [
                r'(\d{4,5}\.?\d*)\s?mm',
                r'(\d{4,5}\.?\d*)\s?millimeter'
            ],
            'standard': [
                r'is\s?\d{4}',
                r'astm\s+[a-z]+\d+',
                r'bs\s+\d+'
            ]
        }
    
    def load_ner_model(self, model_path: str) -> spacy.language.Language:
        """Load NER model with fallback options"""
        try:
            return spacy.load(model_path)
        except OSError:
            print(f"⚠️  Model not found at {model_path}, trying alternatives...")
            alternatives = ["ner_model", "en_core_web_sm", "en_core_web_md"]
            for alt in alternatives:
                try:
                    return spacy.load(alt)
                except OSError:
                    continue
            print("❌ No NER models found. Using blank model.")
            return spacy.blank("en")
    
    def extract_with_ner(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """Extract entities using NER with confidence scores"""
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            # Use entity length and position as confidence proxy
            confidence = min(0.9, 0.5 + (len(ent.text) / 20) + (ent.start_char / len(text) * 0.3))
            entities[ent.label_].append((ent.text, confidence))
        
        return entities
    
    def extract_with_regex(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """Extract entities using regex patterns"""
        entities = {}
        text_lower = text.lower()
        
        for field, patterns in self.patterns.items():
            entities[field] = []
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    # Higher confidence for longer matches
                    confidence = min(0.8, 0.4 + (len(match.group()) / 10))
                    entities[field].append((match.group(), confidence))
        
        return entities
    
    def extract_with_llm(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """Extract entities using LLM fallback"""
        if not GROQ_API_KEY:
            return {}
        
        prompt = f"""
Extract the following fields from this product description. Return as JSON:
- Material (e.g., TMT, OPC, PC Strand)
- Grade (e.g., Fe500, OPC 43, Class I)
- Diameter (in mm)
- Length (in mm)
- Form (e.g., Loose, Bag, Coil)
- Standard (e.g., IS 1786, ASTM)

Text: {text}

Return only the JSON object, no other text.
"""
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": "llama3-70b-8192",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                },
                timeout=10
            )
            
            result = response.json()["choices"][0]["message"]["content"].strip()
            
            # Try to parse JSON
            try:
                data = json.loads(result)
                entities = {}
                for field, value in data.items():
                    if value and str(value).lower() not in ['unknown', 'none', '']:
                        # Ensure value is a string
                        str_value = str(value)
                        entities[field] = [(str_value, 0.6)]  # Lower confidence for LLM
                return entities
            except json.JSONDecodeError:
                return {}
                
        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return {}
    
    def merge_extractions(self, ner_entities: Dict, regex_entities: Dict, llm_entities: Dict) -> Dict[str, str]:
        """Merge extractions from different methods with confidence scoring"""
        merged = {}
        
        # Combine all extractions
        all_entities = {}
        for method, entities in [('ner', ner_entities), ('regex', regex_entities), ('llm', llm_entities)]:
            for field, values in entities.items():
                if field not in all_entities:
                    all_entities[field] = []
                for value, confidence in values:
                    all_entities[field].append((value, confidence, method))
        
        # Select best extraction for each field
        for field, candidates in all_entities.items():
            if not candidates:
                continue
            
            # Sort by confidence
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_value, best_confidence, best_method = candidates[0]
            
            # Only use if confidence is above threshold
            if best_confidence >= self.confidence_threshold:
                merged[field] = best_value
                print(f"  {field}: {best_value} (confidence: {best_confidence:.2f}, method: {best_method})")
            else:
                print(f"  {field}: Skipped (confidence: {best_confidence:.2f} < {self.confidence_threshold})")
        
        return merged
    
    def semantic_similarity(self, val1: str, val2: str) -> float:
        """Calculate semantic similarity between two values"""
        if not val1 or not val2:
            return 0.0
        
        try:
            emb1 = self.semantic_model.encode(val1, convert_to_tensor=True)
            emb2 = self.semantic_model.encode(val2, convert_to_tensor=True)
            similarity = util.cos_sim(emb1, emb2)
            return similarity.item()
        except Exception:
            return 0.0
    
    def compare_field(self, val1: str, val2: str) -> Tuple[str, str, str, float]:
        """Compare two field values with confidence score"""
        if not val1 and not val2:
            return ("⚪ Not Mentioned", val1, val2, 1.0)
        elif val1 == val2:
            return ("✅ Exact Match", val1, val2, 1.0)
        elif val1 and val2:
            # Try fuzzy matching
            fuzzy_ratio = fuzz.ratio(val1.lower(), val2.lower()) / 100
            if fuzzy_ratio > 0.85:
                return ("✅ Fuzzy Match", val1, val2, fuzzy_ratio)
            
            # Try semantic similarity
            semantic_sim = self.semantic_similarity(val1, val2)
            if semantic_sim > 0.8:
                return ("✅ Semantic Match", val1, val2, semantic_sim)
        
        return ("❌ Mismatch", val1, val2, 0.0)
    
    def compare_products(self, text1: str, text2: str) -> Dict:
        """Compare two product descriptions comprehensively"""
        print(f"\n🔍 Comparing products...")
        print(f"Product 1: {text1[:100]}...")
        print(f"Product 2: {text2[:100]}...")
        
        # Extract entities using all methods
        print("\n📊 Extracting entities from Product 1:")
        ner1 = self.extract_with_ner(text1)
        regex1 = self.extract_with_regex(text1)
        llm1 = self.extract_with_llm(text1)
        entities1 = self.merge_extractions(ner1, regex1, llm1)
        
        print("\n📊 Extracting entities from Product 2:")
        ner2 = self.extract_with_ner(text2)
        regex2 = self.extract_with_regex(text2)
        llm2 = self.extract_with_llm(text2)
        entities2 = self.merge_extractions(ner2, regex2, llm2)
        
        # Compare fields
        print("\n🔄 Comparing fields:")
        results = []
        all_fields = set(list(entities1.keys()) + list(entities2.keys()))
        
        for field in sorted(all_fields):
            val1 = entities1.get(field, "")
            val2 = entities2.get(field, "")
            status, v1, v2, confidence = self.compare_field(val1, val2)
            results.append([field, v1, v2, status, confidence])
        
        # Calculate overall similarity
        total_confidence = sum(r[4] for r in results)
        avg_confidence = total_confidence / len(results) if results else 0
        
        # Create comparison report
        report = {
            "product1": text1,
            "product2": text2,
            "extractions": {
                "product1": entities1,
                "product2": entities2
            },
            "comparison": results,
            "overall_similarity": avg_confidence,
            "matching_fields": sum(1 for r in results if "Match" in r[3]),
            "total_fields": len(results)
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Print a formatted comparison report"""
        print("\n" + "="*80)
        print("📋 ENHANCED PRODUCT COMPARISON REPORT")
        print("="*80)
        
        print(f"\n📊 Overall Similarity: {report['overall_similarity']:.2f}")
        print(f"✅ Matching Fields: {report['matching_fields']}/{report['total_fields']}")
        
        # Create comparison table
        table = PrettyTable()
        table.field_names = ["Field", "Product 1", "Product 2", "Status", "Confidence"]
        table.align = "l"
        
        for field, val1, val2, status, confidence in report["comparison"]:
            table.add_row([
                field.title(),
                val1 or "-",
                val2 or "-", 
                status,
                f"{confidence:.2f}"
            ])
        
        print(table)
        
        # Show extraction details
        print(f"\n📝 Extraction Details:")
        print(f"Product 1: {json.dumps(report['extractions']['product1'], indent=2)}")
        print(f"Product 2: {json.dumps(report['extractions']['product2'], indent=2)}")

def main():
    """Main function for testing the enhanced comparator"""
    comparator = EnhancedProductComparator()
    
    # Test examples
    test_cases = [
        (
            "TMT Fe500D 12mm 12000mm IS 1786 Loose",
            "TMT Fe500D 12mm 12000mm IS 1786 Bulk"
        ),
        (
            "OPC 43 Grade Cement 50kg Bag",
            "OPC 53 Grade Cement 50kg Bag"
        ),
        (
            "PC Strand 12.5mm 16000mm IS 14268 Coil",
            "PC Strand 12.5mm 16000mm IS 14268 Bundle"
        )
    ]
    
    for i, (text1, text2) in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}")
        print(f"{'='*60}")
        
        report = comparator.compare_products(text1, text2)
        comparator.print_report(report)

if __name__ == "__main__":
    main() 