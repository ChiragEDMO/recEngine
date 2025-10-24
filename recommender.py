import time
import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt

N = 5
KN = 0.8   
KT = 0.1   
DELTA_T = 0.1
TARGETTIME = 10
A_GATE = 10
R_0 = 0.2

# Mode configurations for different user types
MODE_CONFIGS = {
    "Learner": {
        "KN": 0.6,           # Lower adaptation rate - more forgiving
        "KT": 0.05,          # Lower tangent following - less aggressive
        "DELTA_T": 0.05,     # Smaller steps - gentler difficulty changes
        "TARGETTIME": 15,    # More time allowed - less pressure
        "A_GATE": 8,         # Lower gate threshold - more patient
        "R_0": 0.25,         # Larger error tolerance - more forgiving
        "description": "Gentle learning mode with forgiving difficulty adjustments"
    },
    "Normal": {
        "KN": 0.8,           # Current default values
        "KT": 0.1,
        "DELTA_T": 0.1,
        "TARGETTIME": 10,
        "A_GATE": 10,
        "R_0": 0.2,
        "description": "Balanced mode for typical learning pace"
    },
    "Racer": {
        "KN": 1.0,           # Higher adaptation rate - more responsive
        "KT": 0.15,          # Higher tangent following - more aggressive
        "DELTA_T": 0.15,     # Larger steps - faster difficulty changes
        "TARGETTIME": 7,     # Less time allowed - more pressure
        "A_GATE": 12,        # Higher gate threshold - less patient
        "R_0": 0.15,         # Smaller error tolerance - more challenging
        "description": "Fast-paced mode for quick learners and challenges"
    }
}

def get_mode_config(mode: str) -> dict:
    """
    Get the configuration parameters for a specific mode.
    
    Args:
        mode: One of "Learner", "Normal", or "Racer"
        
    Returns:
        Dictionary containing mode-specific parameters
    """
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Unknown mode: {mode}. Available modes: {list(MODE_CONFIGS.keys())}")
    
    return MODE_CONFIGS[mode].copy()

def get_available_modes() -> list:
    """
    Get list of available modes.
    
    Returns:
        List of available mode names
    """
    return list(MODE_CONFIGS.keys())

# Using the function f(x) = ln(1.7x + 1) for 0 < x <= 1
def f(x):
    """Ideal performance curve: ln(1.7x + 1)."""
    return np.log(1.7 * np.clip(x, 0, 1) + 1.0)

def f_prime(x):
    """First derivative of the ideal curve."""
    return 1.7 / (1.7 * np.clip(x, 0, 1) + 1.0)

def f_double_prime(x):
    """Second derivative of the ideal curve."""
    return -2.89 / ((1.7 * np.clip(x, 0, 1) + 1.0)**2)

# y= tan(x/1.3)
# def f(x):
#     """y(x) = tan(x / 1.3), clipped to [0,1]."""
#     y = np.tan(np.clip(x, 0, 1) / 1.3)
#     return np.clip(y, 0, 1)

# def f_prime(x):
#     x_clip = np.clip(x, 0, 1)
#     return (1/1.3) * (1 / np.cos(x_clip / 1.3))**2

# def f_double_prime(x):
#     x_clip = np.clip(x, 0, 1)
#     sec2 = (1 / np.cos(x_clip / 1.3))**2
#     tan_val = np.tan(x_clip / 1.3)
#     return (2 / (1.3**2)) * sec2 * tan_val

def recommend(response_times, current_difficulty, hints_used, past_responses, mode_params=None) -> float:
    """
    Recommend a new difficulty score based on past responses, response times, and hint usage.
    This function implements an "attract-and-follow" model.
    
    Args:
        response_times: List of response times
        current_difficulty: Current difficulty level
        hints_used: List of hint usage flags
        past_responses: List of (difficulty, correctness) tuples
        mode_params: Optional dict with mode-specific parameters (KN, KT, DELTA_T, etc.)
    """
    # Use mode-specific parameters if provided, otherwise use defaults
    if mode_params:
        k_n = mode_params.get('KN', KN)
        k_t = mode_params.get('KT', KT)
        delta_t = mode_params.get('DELTA_T', DELTA_T)
        target_time = mode_params.get('TARGETTIME', TARGETTIME)
        a_gate = mode_params.get('A_GATE', A_GATE)
        r0 = mode_params.get('R_0', R_0)
    else:
        k_n = KN 
        k_t = KT
        delta_t = DELTA_T
        target_time = TARGETTIME
        a_gate = A_GATE
        r0 = R_0

    x0 = current_difficulty
    correct = past_responses[-1][1]
    hint_used = hints_used[-1]
    time_taken = response_times[-1]

    time_penalty = max(0, (time_taken - target_time) / target_time)

    if correct and not hint_used:
        y0 = 1.0 - 0.5 * time_penalty
    elif correct and hint_used:
        y0 = 0.4 
    else: 
        y0 = 0.0
    
    p = np.array([x0, y0])
    def g(x, p_point):
        return (x - p_point[0]) + (f(x) - p_point[1]) * f_prime(x)
    
    def g_prime(x, p_point):
        return 1 + f_prime(x)**2 + (f(x) - p_point[1]) * f_double_prime(x)

    try:
        x_star = newton(g, x0, fprime=g_prime, args=(p,), tol=1e-5, maxiter=50)
    except (RuntimeError, OverflowError):
        x_star = x0

    c = np.array([x_star, f(x_star)])

    error_vec = p - c
    error_norm = np.linalg.norm(error_vec)

    tangent_vec = np.array([1, f_prime(x_star)])
    tangent_hat = tangent_vec / np.linalg.norm(tangent_vec)

    sigma = 1.0 / (1.0 + np.exp(a_gate * (error_norm - r0)))

    v = -k_n * error_vec + k_t * sigma * tangent_hat

    new_difficulty = x0 + delta_t * v[0]

    new_difficulty = np.clip(new_difficulty, 0.0, 1.0)
    return new_difficulty

def get_response_with_timing():
    """
    Get response from user and measure actual response time
    Handle hints (response = 2) automatically
    """
    start_time = time.time()
    hint_used = False
    
    while True:
        try:
            response = int(input("Response (0=Wrong, 1=Correct, 2=Need Hint): "))
            
            if response == 2:
                hint_used = True
                print("💡 HINT GIVEN")
                continue
            elif response in [0, 1, -1]:
                end_time = time.time()
                response_time = end_time - start_time
                response_time = np.random.rand()
                print(f"Response Time: {response_time:.2f} seconds")
                return response, response_time, hint_used
            else:
                print("Invalid input. Please enter -1, 0, 1, or 2.")
        except ValueError:
            print("Invalid input. Please enter a number (-1, 0, 1, or 2).")

def plot_performance(performance_points):
    """
    Generates a plot of the user's performance against the ideal curve.
    """
    if not performance_points:
        print("No performance data to plot.")
        return

    x_curve = np.linspace(0, 1, 200)
    y_curve = f(x_curve)
    plt.figure(figsize=(10, 6))
    plt.plot(x_curve, y_curve, 'g-', label='Ideal Performance Curve: ln(1.7x+1)')

    difficulties = [p[0] for p in performance_points]
    scores = [p[1] for p in performance_points]
    
    plt.plot(difficulties, scores, 'bo-', label='User\'s Path', markersize=8, markerfacecolor='lightblue')
    
    plt.plot(difficulties[0], scores[0], 'yo', markersize=10, label='Start')
    plt.plot(difficulties[-1], scores[-1], 'ro', markersize=10, label='End')

    plt.title('User Performance vs. Ideal Curve')
    plt.xlabel('Difficulty (x)')
    plt.ylabel('Performance Score (y)')
    plt.xlim(0, 1)
    plt.ylim(-0.1, 1.5)
    plt.grid(True)
    plt.legend()
    plt.show(block=True)

def plot_difficulty_trend(difficulties):
    """
    Generates a plot of the difficulty level over the course of the questions.
    """
    if not difficulties:
        print("No difficulty data to plot.")
        return
    
    question_numbers = range(1, len(difficulties) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(question_numbers, difficulties, 'r-o', label='Difficulty')
    
    plt.title('Difficulty Trend')
    plt.xlabel('Question Number')
    plt.ylabel('Difficulty')
    plt.xlim(left=0, right=len(difficulties) + 1)
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()
    plt.show(block=True)

def backend(difficulty_ranges, excluded_question_ids=None):
    """
    Backend function to fetch questions from CSV based on difficulty ranges.
    
    Args:
        difficulty_ranges: List of (min_diff, max_diff, count) tuples
        excluded_question_ids: Set of question IDs to exclude (already seen questions)
    
    Returns:
        List of selected question dictionaries
    """
    import pandas as pd
    
    if excluded_question_ids is None:
        excluded_question_ids = set()
    
    print("#"*50)
    print("BACKEND FUNCTION CALLED")
    print("#"*50)
    
    try:
        # Load questions from CSV file
        df = pd.read_csv('questions.csv', encoding='utf-8', on_bad_lines='skip')
        
        # Convert difficulty to numeric and filter valid rows
        df['overall_difficulty'] = pd.to_numeric(df['overall_difficulty'], errors='coerce')
        df = df.dropna(subset=['overall_difficulty'])
        
        print(f"Loaded {len(df)} questions from CSV")
        print(f"Excluding {len(excluded_question_ids)} seen questions")
        
        # Convert to list of dictionaries with normalized difficulty (0-1 scale)
        all_questions = []
        for _, row in df.iterrows():
            question_dict = {
                'difficulty': row['overall_difficulty'] / 100.0,  # Convert to 0-1 scale
                'id': row.get('uuid', ''),
                'question_data': row.to_dict()  # Store full row data
            }
            # Skip if question is in excluded list
            if question_dict['id'] not in excluded_question_ids:
                all_questions.append(question_dict)
        
        print(f"Available unseen questions: {len(all_questions)}")
        
    except Exception as e:
        print(f"Error loading questions from CSV: {e}")
        return []
    
    selected_questions = []
    
    for i, (min_diff, max_diff, count) in enumerate(difficulty_ranges):
        print(f"Range {i+1}: [{min_diff:.3f}, {max_diff:.3f}] - requesting {count} questions")
        
        # Filter questions within the difficulty range
        questions_in_range = [
            q for q in all_questions 
            if min_diff <= q.get('difficulty', 0) <= max_diff
        ]
        
        print(f"  Found {len(questions_in_range)} questions in range")
        
        # Randomly select up to 'count' questions from this range
        if questions_in_range:
            selected_count = min(count, len(questions_in_range))
            selected = np.random.choice(questions_in_range, size=selected_count, replace=False).tolist()
            selected_questions.extend(selected)
            print(f"  Selected {selected_count} questions")
        else:
            print(f"  No questions available in this range")
    
    print(f"Total selected: {len(selected_questions)} questions")
    print("#"*50)
    
    return selected_questions

def main():
    recommendedDifficulty = float(input("Initial Difficulty Score (0-1): "))
    pastResponses = []
    responseTime = []
    hintUsed = []
    performance_points = [] # To store (difficulty, performance_score) tuples for plotting

    print("Answer with -1 to stop.")
    while True:
        print(f"\nCurrent Difficulty: {recommendedDifficulty:.3f}")
        answer, time_taken, hint_used_flag = get_response_with_timing()
        if answer == -1:
            break
        
        # Calculate performance score y0 for the current point
        target_time = TARGETTIME
        time_penalty = max(0, (time_taken - target_time) / target_time)
        if answer == 1 and not hint_used_flag:
            y0 = 1.0 - 0.5 * time_penalty
        elif answer == 1 and hint_used_flag:
            y0 = 0.4
        else:
            y0 = 0.0
        
        performance_points.append((recommendedDifficulty, y0))

        # Update history
        answer = bool(answer)
        pastResponses.append((recommendedDifficulty, answer))
        responseTime.append(time_taken)
        hintUsed.append(hint_used_flag)
        
        # Get next difficulty
        recommendedDifficulty = recommend(responseTime, recommendedDifficulty, hintUsed, pastResponses)

    print("\nFinal Summary:")
    if pastResponses:
        total_correct = sum(1 for _, correct in pastResponses if correct)
        total_hints = sum(1 for hint in hintUsed if hint)
        avg_time = sum(responseTime) / len(responseTime)
        print(f"Total questions: {len(pastResponses)}")
        print(f"Correct answers: {total_correct}/{len(pastResponses)} ({total_correct/len(pastResponses)*100:.1f}%)")
        print(f"Hints used: {total_hints}")
        print(f"Average response time: {avg_time:.2f} seconds")
        print(f"Final difficulty: {recommendedDifficulty:.3f}")

        # Generate and show the plots
        plot_performance(performance_points)
        
        difficulties_over_time = [p[0] for p in performance_points]
        plot_difficulty_trend(difficulties_over_time)
    else:
        print("No questions were answered.")