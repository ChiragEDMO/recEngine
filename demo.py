import numpy as np
from recommender import recommend, backend, get_mode_config, get_available_modes
from typing import List, Dict, Tuple, Optional


class RecommendationEngine:
    """
    A recommendation engine that handles question selection and recommendation 
    based on difficulty levels using the attract-and-follow model.
    """
    
    def __init__(self):
        # Core attributes
        self.initial_difficulty: Optional[float] = None
        self.question_responses: List[Tuple[float, bool]] = []
        self.current_question_pool: List[Dict] = []
        self.response_times: List[float] = []
        self.hints_used: List[bool] = []
        self.current_difficulty: Optional[float] = None
        self.seen_question_ids: set = set()  # Track seen questions
        self.total_questions_count: int = 0  # Track total available questions from backend
        self.mode: str = "Normal"  # Default mode
        self.mode_params: dict = {}  # Mode-specific parameters for recommend function
        
        # Hyperparameters
        self.RangeT: float = 0.1  # Range around current difficulty for core questions
        self.ChangeT: float = 0.01  # Threshold for difficulty change
        self.ExtraT: float = 0.05  # Extra range for outer bands
        self.NumQ: int = 30  # Total number of questions to load
        
    def set_initial_difficulty(self, difficulty: float) -> 'RecommendationEngine':
        """
        Builder function to set the initial difficulty level.
        
        Args:
            difficulty: Initial difficulty level (0.0 to 1.0)
            
        Returns:
            Self for method chaining
        """
        if not 0.0 <= difficulty <= 1.0:
            raise ValueError("Difficulty must be between 0.0 and 1.0")
        
        self.initial_difficulty = difficulty
        self.current_difficulty = difficulty
        return self
    
    def set_response_history(self, responses: List[Tuple[float, bool]], 
                           response_times: List[float], 
                           hints_used: List[bool]) -> 'RecommendationEngine':
        """
        Builder function to set the response history.
        
        Args:
            responses: List of (difficulty, is_correct) tuples
            response_times: List of response times in seconds
            hints_used: List of boolean values indicating hint usage
            
        Returns:
            Self for method chaining
        """
        if not (len(responses) == len(response_times) == len(hints_used)):
            raise ValueError("All history lists must have the same length")
        
        self.question_responses = responses.copy()
        self.response_times = response_times.copy()
        self.hints_used = hints_used.copy()
        
        # Update current difficulty based on latest response
        if responses:
            self.current_difficulty = responses[-1][0]
        
        return self
    
    def set_mode(self, mode: str) -> 'RecommendationEngine':
        """
        Builder function to set the learning mode.
        
        Args:
            mode: One of "Learner", "Normal", or "Racer"
            
        Returns:
            Self for method chaining
        """
        available_modes = get_available_modes()
        if mode not in available_modes:
            raise ValueError(f"Invalid mode: {mode}. Available modes: {available_modes}")
        
        self.mode = mode
        self.mode_params = get_mode_config(mode)
        return self

    def set_hyperparameters(self, RangeT: Optional[float] = None,
                          ChangeT: Optional[float] = None,
                          ExtraT: Optional[float] = None, 
                          NumQ: Optional[int] = None) -> 'RecommendationEngine':
        """
        Builder function to set hyperparameters.
        
        Args:
            RangeT: Range around current difficulty for core questions
            ChangeT: Threshold for difficulty change
            ExtraT: Extra range for outer bands
            NumQ: Total number of questions to load
            
        Returns:
            Self for method chaining
        """
        if RangeT is not None:
            if not 0.0 <= RangeT <= 0.5:
                raise ValueError("RangeT must be between 0.0 and 0.5")
            self.RangeT = RangeT
        
        if ChangeT is not None:
            if not 0.0 <= ChangeT <= 0.5:
                raise ValueError("ChangeT must be between 0.0 and 0.5")
            self.ChangeT = ChangeT
        
        if ExtraT is not None:
            if not 0.0 <= ExtraT <= 0.5:
                raise ValueError("ExtraT must be between 0.0 and 0.5")
            self.ExtraT = ExtraT
        
        if NumQ is not None:
            if not isinstance(NumQ, int) or NumQ <= 0:
                raise ValueError("NumQ must be a positive integer")
            self.NumQ = NumQ
        
        return self
    
    def getQuestions(self) -> List[Dict]:
        """
        Fetches a new batch of questions based on current difficulty.
        
        This function calls the backend() function from recommender.py with 
        difficulty ranges and excluded question IDs.
        
        Returns:
            List of selected questions
        """
        if self.current_difficulty is None:
            raise ValueError("Current difficulty not set. Call set_initial_difficulty() first.")
        
        current_diff = self.current_difficulty
        
        # Calculate question counts for each band
        core_count = int(self.NumQ * 0.8)  # 80% from core range
        outer_count_each = int(self.NumQ * 0.1)  # 10% from each outer band
        
        # Define difficulty ranges: (min_diff, max_diff, count)
        difficulty_ranges = [
            # Core range: [currentDiff - RangeT, currentDiff + RangeT]
            (
                max(0.0, current_diff - self.RangeT),
                min(1.0, current_diff + self.RangeT),
                core_count
            ),
            # Lower outer band: [currentDiff - RangeT - ExtraT, currentDiff - RangeT)
            (
                max(0.0, current_diff - self.RangeT - self.ExtraT),
                max(0.0, current_diff - self.RangeT),
                outer_count_each
            ),
            # Upper outer band: (currentDiff + RangeT, currentDiff + RangeT + ExtraT]
            (
                min(1.0, current_diff + self.RangeT),
                min(1.0, current_diff + self.RangeT + self.ExtraT),
                outer_count_each
            )
        ]
        
        print(f"Frontend requesting questions, excluding {len(self.seen_question_ids)} seen questions")
        
        # Call backend function with difficulty ranges and excluded question IDs
        self.current_question_pool = backend(difficulty_ranges, self.seen_question_ids)
        
        return self.current_question_pool.copy()
    
    def nextQuestion(self) -> Dict:
        """
        Determines the next question to recommend based on the recommendation engine.
        
        This function calls the recommend() function from recommender.py and applies
        additional checks before finalizing the recommendation.
        
        Returns:
            The recommended question dictionary
            
        Raises:
            ValueError: If insufficient data for recommendation
        """
        if not self.question_responses or not self.response_times or not self.hints_used:
            raise ValueError("No response history available. Cannot make recommendation.")
        
        if not self.current_question_pool:
            # If no questions in pool, fetch some first
            self.getQuestions()
        
        if not self.current_question_pool:
            raise ValueError("No questions available in current pool")
        
        # Get recommended difficulty using the recommendation engine
        recommended_difficulty = recommend(
            self.response_times, 
            self.current_difficulty, 
            self.hints_used, 
            self.question_responses,
            self.mode_params  # Pass mode-specific parameters
        )
        
        
        # Find the closest available question to the recommended difficulty
        closest_question = min(
            self.current_question_pool,
            key=lambda q: abs(q['difficulty'] - recommended_difficulty)
        )
        
        closest_difficulty = closest_question['difficulty']
        
        # Check conditions for refreshing question pool
        should_refresh = False
        
        # ExtraT Range Check: Is the recommended difficulty in ExtraT range?
        core_min = max(0.0, self.current_difficulty - self.RangeT)
        core_max = min(1.0, self.current_difficulty + self.RangeT)
        
        if not (core_min <= recommended_difficulty <= core_max):
            should_refresh = True
        
        # ChangeT Threshold Check: Is the difference too large?
        if abs(recommended_difficulty - closest_difficulty) > self.ChangeT:
            should_refresh = True
        
        # If either condition is true, refresh the question pool
        if should_refresh:
            print(f"🔄 Refreshing question pool (ExtraT range: {not (core_min <= recommended_difficulty <= core_max)}, ChangeT threshold: {abs(recommended_difficulty - closest_difficulty) > self.ChangeT})")
            
            # Update current difficulty to the recommended one
            self.current_difficulty = recommended_difficulty
            # Refresh question pool
            self.getQuestions()
            
            # Find the closest question again from the new pool
            if self.current_question_pool:
                closest_question = min(
                    self.current_question_pool,
                    key=lambda q: abs(q['difficulty'] - recommended_difficulty)
                )
                new_closest_difficulty = closest_question['difficulty']

        # Mark the selected question as seen and remove from current pool
        if closest_question and 'id' in closest_question:
            question_id = closest_question['id']
            self.seen_question_ids.add(question_id)
            
            # Final logging summary
            final_difficulty = closest_question['difficulty']
            print(f"Recommended Difficulty: {recommended_difficulty:.4f}")
            print(f"Selected Question Difficulty: {final_difficulty:.4f}")
            print(f"Final Difference: {abs(recommended_difficulty - final_difficulty):.4f}")
            print(f"Questions Left in Pool: {len(self.current_question_pool) - 1}\n")
            
            # Remove from current pool to avoid selecting it again immediately
            self.current_question_pool = [q for q in self.current_question_pool if q.get('id') != question_id]
        
        return closest_question
    
    def mark_question_as_seen(self, question_id: str) -> None:
        """
        Mark a question as seen to exclude it from future recommendations.
        
        Args:
            question_id: ID of the question to mark as seen
        """
        self.seen_question_ids.add(question_id)
    
    def add_response(self, question_difficulty: float, is_correct: bool, 
                    response_time: float, hint_used: bool = False) -> None:
        """
        Add a new response to the history.
        
        Args:
            question_difficulty: Difficulty of the answered question
            is_correct: Whether the answer was correct
            response_time: Time taken to answer in seconds
            hint_used: Whether a hint was used
        """
        self.question_responses.append((question_difficulty, is_correct))
        self.response_times.append(response_time)
        self.hints_used.append(hint_used)
        self.current_difficulty = question_difficulty
    
    def get_current_stats(self) -> Dict:
        """
        Get current statistics about the recommendation engine state.
        
        Returns:
            Dictionary containing current statistics
        """
        if not self.question_responses:
            return {
                'current_difficulty': self.current_difficulty,
                'total_questions': 0,
                'accuracy': 0.0,
                'avg_response_time': 0.0,
                'hints_used_count': 0
            }
        
        total_correct = sum(1 for _, correct in self.question_responses if correct)
        accuracy = total_correct / len(self.question_responses) if self.question_responses else 0.0
        avg_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
        hints_count = sum(1 for hint in self.hints_used if hint)
        
        return {
            'current_difficulty': self.current_difficulty,
            'total_questions': len(self.question_responses),
            'accuracy': accuracy,
            'avg_response_time': avg_time,
            'hints_used_count': hints_count,
            'questions_in_pool': len(self.current_question_pool),
            'seen_questions_count': len(self.seen_question_ids),
            'total_questions_in_db': self.total_questions_count,  # Updated by backend
            'mode': self.mode,
            'mode_params': self.mode_params
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize and configure the recommendation engine
    # No need to set questions - backend handles CSV access directly
    engine = (RecommendationEngine()
              .set_initial_difficulty(0.5)
              .set_hyperparameters(RangeT=0.1, ChangeT=0.05, ExtraT=0.15, NumQ=10))
    
    print("Engine initialized, getting first batch of questions...")
    
    # Get questions from backend via CSV
    try:
        questions = engine.getQuestions()
        print(f"Fetched {len(questions)} questions from backend")
        
        if questions:
            # Add some sample response history
            engine.add_response(0.5, True, 8.5, False)  # Correct answer, no hint
            engine.add_response(0.6, False, 12.0, True)  # Wrong answer, used hint
            
            print("Current stats:", engine.get_current_stats())
            
            # Get next recommendation
            next_q = engine.nextQuestion()
            print(f"Next recommended question: ID {next_q['id']}, difficulty {next_q['difficulty']}")
        else:
            print("No questions available from backend")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure questions.csv exists in the current directory")