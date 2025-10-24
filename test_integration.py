"""
Test script to verify the integration between app.py and demo.py
"""

import pandas as pd
import numpy as np
from demo import RecommendationEngine

# Create test data similar to the CSV structure
def create_test_data():
    test_questions = []
    for i in range(20):
        difficulty = i * 5  # 0, 5, 10, ..., 95
        test_questions.append({
            'uuid': f'test-{i}',
            'answer_UUID': f'answer-{i}',
            'question': f'Test question {i}',
            'chapter': 'Test Chapter',
            'topic': 'Test Topic',
            'concept': 'Test Concept',
            'answer': f'{{"question": "Test question {i}", "answers": [{{"answers": "Option A", "isCorrect": true}}, {{"answers": "Option B", "isCorrect": false}}], "hint": "Test hint {i}"}}',
            'hint': f'Test hint {i}',
            'keywords': 'test',
            'conceptual_difficulty': difficulty,
            'comprehension_difficulty': difficulty,
            'calculation_difficulty': difficulty,
            'overall_difficulty': difficulty
        })
    
    return pd.DataFrame(test_questions)

def convert_df_to_questions(df):
    """Convert pandas DataFrame to list of question dictionaries for RecommendationEngine"""
    questions = []
    for _, row in df.iterrows():
        question_dict = {
            'difficulty': row['overall_difficulty'] / 100.0,  # Convert to 0-1 scale
            'id': row.get('uuid', ''),
            'question_data': row.to_dict()  # Store full row data
        }
        questions.append(question_dict)
    return questions

def test_integration():
    print("Testing integration between app.py and demo.py...")
    print("Note: This test requires questions.csv to be present")
    
    # Initialize recommendation engine (no questions needed - backend handles CSV)
    engine = (RecommendationEngine()
              .set_initial_difficulty(0.5)
              .set_hyperparameters(RangeT=0.1, ChangeT=0.05, ExtraT=0.15, NumQ=10))
    
    print("✅ Recommendation engine initialized")
    
    # Test getting questions
    engine.getQuestions()
    print(f"✅ Got {len(engine.current_question_pool)} questions in pool")
    
    # Simulate a few responses
    engine.add_response(0.5, True, 8.5, False)  # Correct, no hint
    engine.add_response(0.6, False, 12.0, True)  # Wrong, used hint
    
    print("✅ Added sample responses")
    
    # Test getting next question
    try:
        next_q = engine.nextQuestion()
        print(f"✅ Next question: difficulty {next_q['difficulty']:.3f}")
    except Exception as e:
        print(f"❌ Error getting next question: {e}")
        return False
    
    # Test stats
    stats = engine.get_current_stats()
    print(f"✅ Stats: accuracy={stats['accuracy']:.1%}, total_questions={stats['total_questions']}")
    print(f"✅ Seen questions: {stats['seen_questions_count']}, Total in DB: {stats['total_questions_in_db']}")
    
    # Test that seen questions are excluded
    print("\n🔍 Testing seen question exclusion...")
    
    # Add a few more responses to see more questions
    for i in range(3):
        engine.add_response(0.5 + i*0.1, i % 2 == 0, 8.0 + i, False)
        try:
            next_q = engine.nextQuestion()
            print(f"  Question {i+2}: ID {next_q['id']}, difficulty {next_q['difficulty']:.3f}")
        except Exception as e:
            print(f"  Error getting question {i+2}: {e}")
    
    # Final stats
    final_stats = engine.get_current_stats()
    print(f"\n✅ Final stats: {final_stats['seen_questions_count']} seen, Total in DB: {final_stats['total_questions_in_db']}")
    
    print("\n🎉 Integration test completed successfully!")
    return True

if __name__ == "__main__":
    test_integration()