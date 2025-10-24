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