# [Read this doc](https://docs.google.com/document/d/1Or-LYpDyq-Ge9j-xJa4zc8UrS1qkJDGV5eHbQmdSjZg/edit?tab=t.0)

## Run `streamlit run app.py` to test it out
# Adaptive Question System

An intelligent question recommendation system built with Streamlit that adapts to user performance in real-time.

## Features

- **Adaptive Difficulty**: Uses a sophisticated recommendation engine to adjust question difficulty based on user performance
- **Real-time Tracking**: Monitors response time, answer correctness, and hint usage
- **Interactive Interface**: Clean, user-friendly interface with multiple choice questions and optional hints
- **Progress Visualization**: Dynamic graph showing difficulty progression over time
- **Comprehensive Question Database**: Math questions with varying difficulty levels

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run app.py
   ```

3. **Access the Application**
   - Open your browser and navigate to `http://localhost:8501`

## How It Works

### Initial Setup
1. Enter your initial difficulty rating (0-100)
2. Click "Start Questions" to begin

### Question Flow
1. **Question Display**: Each question shows:
   - The math problem
   - Multiple choice options (A, B, C, D)
   - Current difficulty rating in the top-right
   - Optional hint button

2. **Answer Submission**: 
   - Select your answer
   - Optionally view the hint (affects recommendation)
   - Submit your response

3. **Adaptive Feedback**:
   - System records response time
   - Tracks answer correctness
   - Notes hint usage
   - Calculates new difficulty using recommendation engine

4. **Progress Tracking**:
   - Real-time graph updates after each question
   - Shows difficulty progression over time

### Recommendation Engine

The system uses the existing `recommend()` function from `recommender.py` which:
- Takes into account response time, correctness, and hint usage
- Returns a difficulty rating between 0 and 1
- Adapts to user performance patterns

## Learning Modes & Hyperparameters

The system offers three learning modes, each with different hyperparameters that control how the difficulty adjusts:

### 🎓 **Learner Mode** (Gentle & Forgiving)
- **KN = 0.6**: Lower attraction strength - slower difficulty adjustments
- **KT = 0.05**: Minimal time penalty - less punishment for slow responses
- **DELTA_T = 15s**: Longer target time - more time to think through problems
- **Best for**: Beginners, students building confidence, or learning new concepts

### ⚖️ **Normal Mode** (Balanced Progression)  
- **KN = 0.8**: Moderate attraction strength - balanced difficulty changes
- **KT = 0.1**: Standard time penalty - moderate speed expectations
- **DELTA_T = 10s**: Average target time - reasonable thinking time
- **Best for**: Most learners, general practice sessions, steady skill development

### 🏎️ **Racer Mode** (Challenging & Fast-paced)
- **KN = 1.0**: High attraction strength - rapid difficulty adjustments  
- **KT = 0.15**: Strong time penalty - rewards quick thinking
- **DELTA_T = 8s**: Short target time - encourages faster responses
- **Best for**: Advanced users, competitive practice, skill assessment

## Hyperparameter Details

### **KN (Newton's Method Strength)**
- **Range**: 0.0 - 1.0
- **Purpose**: Controls how strongly the system tries to match your performance to the ideal learning curve
- **Higher values**: Faster, more aggressive difficulty adjustments
- **Lower values**: Gentler, more gradual difficulty changes

### **KT (Time Penalty Factor)**
- **Range**: 0.0 - 0.2  
- **Purpose**: Determines how much slow response times affect difficulty
- **Higher values**: Slower responses lead to bigger difficulty drops
- **Lower values**: Response time has less impact on difficulty

### **DELTA_T (Target Response Time)**
- **Range**: 5s - 20s
- **Purpose**: The "ideal" time expected for answering questions
- **Higher values**: More thinking time allowed before penalties
- **Lower values**: Encourages quicker decision making

### **Additional Engine Parameters**
- **RangeT = 0.1**: Question selection range around target difficulty
- **ChangeT = 0.01**: Minimum difficulty change threshold  
- **ExtraT = 0.05**: Buffer for question pool expansion
- **NumQ = 30**: Number of questions to load in each batch

## How Hyperparameters Work Together

The recommendation engine uses an "**attract-and-follow**" mathematical model:

1. **Ideal Performance Curve**: `f(x) = ln(1.7x + 1)`
   - Represents the optimal difficulty progression for learning
   - Your actual performance is compared to this ideal curve

2. **Newton's Method Projection**: 
   - Uses **KN** to control how strongly the system pulls you toward the ideal curve
   - Higher KN = faster convergence to ideal performance

3. **Time-based Adjustments**:
   - **KT** applies penalties when response time exceeds **DELTA_T**  
   - Formula: `time_penalty = KT × max(0, response_time - DELTA_T)`

4. **Final Difficulty Calculation**:
   ```
   new_difficulty = projected_difficulty - time_penalty + hint_penalty
   ```

### **Choosing Your Mode**

| Situation | Recommended Mode | Why |
|-----------|------------------|-----|
| Learning new concepts | **Learner** | Gentle adjustments build confidence |
| Regular practice | **Normal** | Balanced challenge and support |
| Testing your skills | **Racer** | Quick feedback reveals true ability |
| Time pressure training | **Racer** | Builds speed and accuracy together |
| Struggling with material | **Learner** | Reduces frustration, maintains motivation |

## File Structure

```
Rec-v1/
├── app.py              # Main Streamlit application
├── recommender.py      # Existing recommendation engine
├── questions.csv       # Question database
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Question Database Format

The `questions.csv` file contains:
- **question**: The math problem text
- **answer**: JSON containing question details, options, and correct answers
- **hint**: Helpful hint for the question
- **overall_difficulty**: Difficulty rating (0-100)

## Usage Tips

- **Hints**: Using hints will affect the recommendation algorithm
- **Response Time**: Faster responses may lead to increased difficulty
- **Accuracy**: Correct answers generally increase difficulty
- **Progress**: Watch the graph to see how your difficulty adapts over time

## Dependencies

- `streamlit`: Web application framework
- `plotly`: Interactive graphing library
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `scipy`: Scientific computing (used by recommender)
- `matplotlib`: Additional plotting capabilities

## Troubleshooting

1. **Questions not loading**: Ensure `questions.csv` is in the same directory as `app.py`
2. **Import errors**: Make sure all dependencies are installed (`pip install -r requirements.txt`)
3. **Graph not updating**: Clear browser cache and refresh the page

## Customization

To add more questions:
1. Edit `questions.csv` with new entries
2. Follow the existing JSON format for the answer column
3. Include appropriate difficulty ratings

To modify the recommendation algorithm:
1. Edit the `recommend()` function in `recommender.py`
2. The function should return a value between 0 and 1