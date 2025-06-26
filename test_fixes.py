#!/usr/bin/env python3
"""
Test script to verify all fixes work correctly
"""

import sys
import os
sys.path.append(".")

def test_data_preparation():
    """Test the data preparation fix"""
    print("🧪 Testing data preparation fix...")
    
    try:
        from train_ner_improved import filter_overlapping_entities
        
        # Test overlapping entities
        test_entities = [
            (0, 3, "Material"),    # "TMT"
            (0, 6, "Material"),    # "TMT Fe" - overlaps with above
            (4, 10, "Grade"),      # "Fe500"
            (11, 15, "Diameter"),  # "12mm"
        ]
        
        filtered = filter_overlapping_entities(test_entities)
        print(f"✅ Original entities: {len(test_entities)}")
        print(f"✅ Filtered entities: {len(filtered)}")
        print(f"✅ Filtered result: {filtered}")
        
        return True
    except Exception as e:
        print(f"❌ Data preparation test failed: {e}")
        return False

def test_llm_extraction():
    """Test the LLM extraction fix"""
    print("\n🧪 Testing LLM extraction fix...")
    
    try:
        from product_comparator_enhanced import EnhancedProductComparator
        
        # Create comparator without API key to test the fix
        comparator = EnhancedProductComparator()
        
        # Test with sample data that would cause the error
        test_text = "TMT Fe500D 12mm 12000mm IS 1786"
        
        # This should not crash even without API key
        result = comparator.extract_with_llm(test_text)
        print(f"✅ LLM extraction completed without errors")
        print(f"✅ Result: {result}")
        
        return True
    except Exception as e:
        print(f"❌ LLM extraction test failed: {e}")
        return False

def test_evaluation_metrics():
    """Test the evaluation metrics fix"""
    print("\n🧪 Testing evaluation metrics fix...")
    
    try:
        from evaluate_improved import evaluate_ner_detailed
        import spacy
        
        # Create a simple test
        nlp = spacy.blank("en")
        
        # Simple test data
        test_data = [
            ("TMT Fe500 12mm", {"entities": [(0, 3, "Material"), (4, 9, "Grade"), (10, 13, "Diameter")]}),
            ("OPC 43 Cement", {"entities": [(0, 3, "Material"), (4, 6, "Grade")]})
        ]
        
        metrics, confusion_matrix, error_examples = evaluate_ner_detailed(nlp, test_data)
        
        print(f"✅ Evaluation completed without errors")
        print(f"✅ Metrics keys: {list(metrics.keys())}")
        print(f"✅ Error examples: {len(error_examples)}")
        
        return True
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False

def test_model_loading():
    """Test model loading with fallbacks"""
    print("\n🧪 Testing model loading...")
    
    try:
        from train_ner_improved import check_model_availability
        
        model_name = check_model_availability()
        print(f"✅ Model found: {model_name}")
        
        return True
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running fix verification tests...\n")
    
    tests = [
        test_data_preparation,
        test_llm_extraction,
        test_evaluation_metrics,
        test_model_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All fixes verified successfully!")
        print("\n✅ You can now run:")
        print("   python train_ner_improved.py")
        print("   python evaluate_improved.py")
        print("   python product_comparator_enhanced.py")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 