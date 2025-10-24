import streamlit as st
import pandas as pd
import json
import time
import plotly.graph_objects as go
from demo import RecommendationEngine
from recommender import get_available_modes, get_mode_config
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Adaptive Question System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add custom CSS for smoother transitions
st.markdown("""
<style>
    .stButton > button {
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .element-container {
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stMetric {
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateX(20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .stSuccess, .stError, .stInfo {
        animation: bounceIn 0.5s ease-out;
    }
    
    @keyframes bounceIn {
        0% { transform: scale(0.3); opacity: 0; }
        50% { transform: scale(1.05); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.questions_df = None
    st.session_state.current_question = None
    st.session_state.initial_difficulty = None
    st.session_state.current_difficulty = None
    st.session_state.question_count = 0
    st.session_state.response_times = []
    st.session_state.past_responses = []
    st.session_state.hints_used = []
    st.session_state.difficulty_history = []
    st.session_state.question_start_time = None
    st.session_state.hint_shown = False
    st.session_state.answer_submitted = False
    st.session_state.show_graph = False
    st.session_state.last_selected_answer = None
    st.session_state.last_response_time = 0
    st.session_state.last_hint_used = False
    st.session_state.last_is_correct = False
    st.session_state.recommendation_engine = None
    st.session_state.selected_mode = "Normal"  # Default mode

# Load questions data
@st.cache_data
def load_questions():
    """Load questions from CSV file with robust parsing"""
    try:
        # First, try the standard pandas approach
        df = pd.read_csv('questions.csv', encoding='utf-8')
        
        # Convert difficulty columns to numeric
        numeric_cols = ['conceptual_difficulty', 'comprehension_difficulty', 'calculation_difficulty', 'overall_difficulty']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with missing overall_difficulty
        df = df.dropna(subset=['overall_difficulty'])
        
        return df
        
    except pd.errors.ParserError as e:
        st.warning(f"CSV parsing error: {e}")
        st.info("Attempting to load with error handling...")
        
        try:
            # Try with error_bad_lines=False (for older pandas versions)
            df = pd.read_csv('questions.csv', encoding='utf-8', on_bad_lines='skip')
            
            # Convert difficulty columns to numeric
            numeric_cols = ['conceptual_difficulty', 'comprehension_difficulty', 'calculation_difficulty', 'overall_difficulty']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any rows with missing overall_difficulty
            df = df.dropna(subset=['overall_difficulty'])
            
            st.success(f"Successfully loaded {len(df)} questions with error handling")
            return df
            
        except Exception as e2:
            st.error(f"Failed to load CSV even with error handling: {e2}")
            # Return empty dataframe as fallback
            return pd.DataFrame(columns=['uuid', 'answer_UUID', 'question', 'chapter', 'topic', 'concept', 'answer', 'hint', 'keywords', 'conceptual_difficulty', 'comprehension_difficulty', 'calculation_difficulty', 'overall_difficulty'])
    
    except Exception as e:
        st.error(f"Unexpected error loading questions: {e}")
        return pd.DataFrame()

def parse_answer_json(answer_str):
    """Parse the answer JSON string to extract question details"""
    try:
        answer_data = json.loads(answer_str)
        return answer_data
    except json.JSONDecodeError:
        return None

def initialize_recommendation_engine(initial_difficulty):
    """Initialize the recommendation engine with initial difficulty"""
    print(f"Initializing recommendation engine: RangeT=0.1, ChangeT=0.01, ExtraT=0.05, NumQ=30, InitDiff={initial_difficulty/100}")
    engine = (RecommendationEngine()
              .set_initial_difficulty(initial_difficulty / 100.0)  # Convert to 0-1 scale
              .set_hyperparameters(RangeT=0.1, ChangeT=0.01, ExtraT=0.05, NumQ=30))
    
    return engine

def get_next_question_from_engine(engine):
    """Get the next question using the recommendation engine"""
    try:
        if engine is None:
            st.error("Recommendation engine not initialized")
            return None
            
        if len(engine.question_responses) == 0:
            # For first question, just get questions and pick one close to initial difficulty
            engine.getQuestions()
            if engine.current_question_pool:
                # Find question closest to current difficulty
                closest_q = min(engine.current_question_pool, 
                              key=lambda q: abs(q['difficulty'] - engine.current_difficulty))
                # Mark as seen
                if closest_q and 'id' in closest_q:
                    engine.mark_question_as_seen(closest_q['id'])
                return closest_q
            else:
                # No questions available in pool
                return None
        else:
            # Use the recommendation engine for subsequent questions
            return engine.nextQuestion()
    except Exception as e:
        st.error(f"Error getting next question: {e}")
        # No fallback available since backend manages questions
        return None

def display_question_interface(question_data, difficulty_rating):
    """Display the question interface with options and hint button"""
    
    # Create layout with difficulty rating in top right
    col1, col2 = st.columns([4, 1])
    
    with col2:
        st.metric("Difficulty", f"{int(difficulty_rating * 100)}/100")
    
    with col1:
        # Display question
        st.markdown("### Question")
        st.markdown(question_data['question'])
        
        # Parse answer data
        answer_data = parse_answer_json(question_data['answer'])
        if not answer_data:
            st.error("Could not parse question data")
            return None, None, None
        
        # Display answer options
        options = answer_data.get('answers', [])
        if not options:
            st.error("No answer options found")
            return None, None, None
        
        # Create radio buttons for answers
        answer_choices = []
        for i, option in enumerate(options):
            answer_text = option['answers']
            answer_choices.append(f"{chr(65+i)}. {answer_text}")
        
        selected_answer = st.radio("Select your answer:", answer_choices, key=f"answer_{st.session_state.question_count}")
        
        # Hint section
        col_hint1, col_hint2 = st.columns([1, 3])
        
        with col_hint1:
            if st.button("💡 Show Hint", key=f"hint_{st.session_state.question_count}"):
                st.session_state.hint_shown = True
        
        with col_hint2:
            if st.session_state.hint_shown:
                st.info(f"**Hint:** {question_data['hint']}")
        
        # Submit and Next buttons in the same row
        col_submit, col_next, col_spacer = st.columns([2, 2, 2])
        
        with col_submit:
            submitted = st.button("Submit Answer", type="primary", key=f"submit_{st.session_state.question_count}")
        
        # Show next button only after answer is submitted
        next_clicked = False
        with col_next:
            if st.session_state.answer_submitted:
                next_clicked = st.button("Next Question ➡️", type="secondary", key=f"next_{st.session_state.question_count}")
        
        return selected_answer, submitted, next_clicked

def process_answer(selected_answer, correct_answers, question_start_time):
    """Process the submitted answer and return results"""
    
    # Calculate response time
    response_time = time.time() - question_start_time
    
    # Extract answer index
    answer_index = ord(selected_answer[0]) - ord('A')
    
    # Check if answer is correct
    is_correct = correct_answers[answer_index]['isCorrect']
    
    # Record hint usage
    hint_used = st.session_state.hint_shown
    
    return {
        'is_correct': is_correct,
        'response_time': response_time,
        'hint_used': hint_used,
        'answer_index': answer_index
    }

def update_difficulty_with_recommendation(result, question_difficulty):
    """Use the recommendation engine to update difficulty and add response"""
    
    # Add response to the recommendation engine
    st.session_state.recommendation_engine.add_response(
        question_difficulty / 100.0,  # Convert to 0-1 scale
        result['is_correct'],
        result['response_time'],
        result['hint_used']
    )
    
    # Update session state for backward compatibility
    st.session_state.response_times.append(result['response_time'])
    st.session_state.past_responses.append((question_difficulty / 100.0, result['is_correct']))
    st.session_state.hints_used.append(result['hint_used'])
    
    # Update current difficulty
    new_difficulty = st.session_state.recommendation_engine.current_difficulty
    st.session_state.current_difficulty = new_difficulty
    st.session_state.difficulty_history.append(new_difficulty * 100)  # Convert to 0-100 scale for display
    
    return new_difficulty

def create_difficulty_graph():
    """Create and display the difficulty progression graph"""
    if len(st.session_state.difficulty_history) > 0:
        fig = go.Figure()
        
        question_numbers = list(range(1, len(st.session_state.difficulty_history) + 1))
        
        fig.add_trace(go.Scatter(
            x=question_numbers,
            y=st.session_state.difficulty_history,
            mode='lines+markers',
            name='Difficulty Progression',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Difficulty Rating Progression",
            xaxis_title="Question Number",
            yaxis_title="Difficulty Rating (0-100)",
            yaxis=dict(range=[0, 100]),
            height=400,
            showlegend=False
        )
        
        return fig
    return None

# Main application
def main():
    st.title("🎯 Adaptive Question System")
    st.markdown("---")
    
    # Mode selection
    st.header("🎯 Choose Your Learning Mode")
    available_modes = get_available_modes()
    mode_descriptions = {
        "Learner": "🎓 Gentle learning with forgiving difficulty adjustments",
        "Normal": "⚖️ Balanced progression for most learners", 
        "Racer": "🏎️ Challenging mode with rapid difficulty increases"
    }

    selected_mode = st.selectbox(
        "Select your learning style:",
        options=available_modes,
        index=available_modes.index(st.session_state.selected_mode),
        format_func=lambda x: f"{x} - {mode_descriptions.get(x, '')}"
    )

    # Update mode if changed
    if selected_mode != st.session_state.selected_mode:
        st.session_state.selected_mode = selected_mode
        # Update the recommendation engine mode when it's initialized
        if st.session_state.recommendation_engine is not None:
            st.session_state.recommendation_engine.set_mode(selected_mode)
        st.rerun()

    st.markdown("---")
    
    # Load questions data
    if st.session_state.questions_df is None:
        with st.spinner("Loading questions..."):
            st.session_state.questions_df = load_questions()
            
        # Check if we successfully loaded questions
        if st.session_state.questions_df.empty:
            st.error("❌ Failed to load questions from CSV file. Please check the file format and try again.")
            st.stop()
        else:
            st.success(f"✅ Successfully loaded {len(st.session_state.questions_df)} questions!")
    
    # Phase 1: Get initial difficulty
    if st.session_state.initial_difficulty is None:
        st.markdown("### Welcome to the Adaptive Question System")
        st.markdown("Please enter your initial difficulty rating to get started.")
        
        initial_difficulty = st.slider(
            "Initial Difficulty Rating (0-100)",
            min_value=0,
            max_value=100,
            value=50,
            help="Select your preferred starting difficulty level"
        )
        
        if st.button("Start Questions", type="primary"):
            st.session_state.initial_difficulty = initial_difficulty
            st.session_state.current_difficulty = initial_difficulty / 100.0  # Convert to 0-1 scale
            st.session_state.difficulty_history.append(initial_difficulty)
            
            # Initialize the recommendation engine
            st.session_state.recommendation_engine = initialize_recommendation_engine(initial_difficulty)
            # Set the selected mode
            st.session_state.recommendation_engine.set_mode(st.session_state.selected_mode)
            
            st.rerun()
    
    # Phase 2: Question flow
    else:
        # Create two columns: main content and graph
        if st.session_state.show_graph and len(st.session_state.difficulty_history) > 1:
            col_main, col_graph = st.columns([2, 1])
        else:
            col_main = st.container()
            col_graph = None
        
        with col_main:
            # Check if we need to load a new question
            if not st.session_state.answer_submitted:
                # Load new question if needed
                if st.session_state.current_question is None:
                    # Check if recommendation engine is initialized
                    if st.session_state.recommendation_engine is None:
                        st.error("Recommendation engine not initialized. Please restart the session.")
                        st.stop()
                        
                    with st.spinner("Loading next question..."):
                        # Get next question using recommendation engine
                        next_question = get_next_question_from_engine(st.session_state.recommendation_engine)
                        
                        if next_question and 'question_data' in next_question:
                            # Convert back to DataFrame row format for compatibility
                            question_series = pd.Series(next_question['question_data'])
                            st.session_state.current_question = question_series
                        else:
                            st.error("Unable to load next question")
                            st.stop()
                            
                        st.session_state.question_start_time = time.time()
                        st.session_state.hint_shown = False
                        st.session_state.question_count += 1
                
                # Display question interface
                selected_answer, submitted, next_clicked = display_question_interface(
                    st.session_state.current_question,
                    st.session_state.current_difficulty
                )
                
                # Process submission
                if submitted and selected_answer:
                    # Parse answer data
                    answer_data = parse_answer_json(st.session_state.current_question['answer'])
                    
                    # Process the answer
                    result = process_answer(
                        selected_answer,
                        answer_data['answers'],
                        st.session_state.question_start_time
                    )
                    
                    # Store answer details for display
                    st.session_state.last_selected_answer = selected_answer
                    st.session_state.last_response_time = result['response_time']
                    st.session_state.last_hint_used = result['hint_used']
                    st.session_state.last_is_correct = result['is_correct']
                    
                    # Update difficulty using recommendation engine
                    new_difficulty = update_difficulty_with_recommendation(
                        result, 
                        st.session_state.current_question['overall_difficulty']
                    )
                    
                    # Mark as submitted
                    st.session_state.answer_submitted = True
                    st.session_state.show_graph = True
                    
                    # Show results inline
                    st.rerun()
            
            else:
                # Show results after submission with smooth transition
                # Re-display the question for context
                st.markdown("### Question")
                st.markdown(st.session_state.current_question['question'])
                
                # Show the answer that was selected
                answer_data = parse_answer_json(st.session_state.current_question['answer'])
                options = answer_data.get('answers', [])
                
                # Display all options with the selected one highlighted
                st.markdown("**Your Answer:**")
                for i, option in enumerate(options):
                    answer_text = option['answers']
                    is_selected = st.session_state.last_selected_answer and i == (ord(st.session_state.last_selected_answer[0]) - ord('A'))
                    is_correct = option['isCorrect']
                    
                    if is_selected:
                        if is_correct:
                            st.success(f"✅ {chr(65+i)}. {answer_text} (Your answer - Correct!)")
                        else:
                            st.error(f"❌ {chr(65+i)}. {answer_text} (Your answer - Incorrect)")
                    elif is_correct and not (st.session_state.last_selected_answer and any(opt['isCorrect'] for j, opt in enumerate(options) if j == (ord(st.session_state.last_selected_answer[0]) - ord('A')))):
                        st.info(f"✅ {chr(65+i)}. {answer_text} (Correct answer)")
                
                # Show performance metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Response Time", f"{st.session_state.last_response_time:.2f}s")
                
                with col2:
                    st.metric("Hint Used", "Yes" if st.session_state.last_hint_used else "No")
                
                with col3:
                    st.metric("Current Score", f"{sum(1 for _, correct in st.session_state.past_responses if correct)}/{len(st.session_state.past_responses)}")
                
                with col4:
                    st.metric("Next Difficulty", f"{int(st.session_state.current_difficulty * 100)}/100")
                
                # Show recommendation engine stats
                if st.session_state.recommendation_engine:
                    stats = st.session_state.recommendation_engine.get_current_stats()
                    with st.expander("🔬 Recommendation Engine Stats", expanded=False):
                        col_stats1, col_stats2, col_stats3 = st.columns(3)
                        with col_stats1:
                            st.metric("Engine Accuracy", f"{stats['accuracy']:.1%}")
                            st.metric("Avg Response Time", f"{stats['avg_response_time']:.2f}s")
                        with col_stats2:
                            st.metric("Total Questions", stats['total_questions'])
                            st.metric("Questions in Pool", stats['questions_in_pool'])
                        with col_stats3:
                            st.metric("Seen Questions", stats['seen_questions_count'])
                            st.metric("Total in DB", stats.get('total_questions_in_db', 'Unknown'))
                        
                        # Add mode information
                        st.markdown("---")
                        mode_info = f"**Current Mode:** {st.session_state.selected_mode}"
                        if 'mode_params' in stats and stats['mode_params']:
                            mode_params = stats['mode_params']
                            mode_info += f" (KN: {mode_params.get('KN', 'N/A')}, KT: {mode_params.get('KT', 'N/A')}, ΔT: {mode_params.get('DELTA_T', 'N/A')})"
                        st.markdown(mode_info)
                
                # Inline next question button
                col_next1, col_next2, col_next3 = st.columns([2, 2, 2])
                with col_next2:
                    if st.button("Next Question ➡️", type="primary", key="next_main"):
                        # Show loading animation
                        with st.spinner("🔄 Preparing your next question..."):
                            time.sleep(0.5)  # Brief pause for smooth transition
                        
                        # Reset for next question with smooth transition
                        st.session_state.current_question = None
                        st.session_state.answer_submitted = False
                        st.session_state.hint_shown = False
                        st.rerun()
        
        # Display graph in sidebar or column
        if col_graph is not None:
            with col_graph:
                st.markdown("### Progress")
                fig = create_difficulty_graph()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        elif st.session_state.show_graph:
            # Show graph below main content if not in columns
            st.markdown("---")
            st.markdown("### Difficulty Progression")
            fig = create_difficulty_graph()
            if fig:
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()