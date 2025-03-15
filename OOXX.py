import numpy as np

class Agent():
    def __init__(self, OOXX_Index: int, epsilon: float, alpha: float):
        """
        Initialize an agent for Tic-Tac-Toe.

        Parameters:
        - OOXX_Index (int): Identifier for the player's mark (e.g., 1 for 'O' or 2 for 'X').
        - epsilon (float): Probability of choosing a random move (exploration rate in ε-greedy strategy).
        - alpha (float): Learning rate used for updating the value estimates.

        The agent maintains:
        - A value table: a 9-dimensional numpy array (each dimension size 3) representing the estimated 
          value for every possible board state. Each board position can be in one of three states:
          empty (0), 'O' (1), or 'X' (2).
        - stored_outcome: a record of the previous board state (as a 1D array of length 9), used for updating the value.
        """
        self.index = OOXX_Index
        self.epsilon = epsilon 
        self.alpha = alpha
        self.value = np.zeros((3,3,3,3,3,3,3,3,3))
        self.stored_outcome = np.zeros(9).astype(np.int8)
        
    def reset(self):
        self.stored_outcome = np.zeros(9).astype(np.int8)
        
    def move(self, state):
        """
        Select and execute a move on the board using an ε-greedy strategy.

        Parameters:
        - state (numpy array): Current board state as a 1D array of length 9.

        Returns:
        - outcome (numpy array): The new board state after the move.

        Process:
        1. Create a copy of the current state.
        2. Identify all empty positions on the board.
        3. With probability 'epsilon', choose a random available move (exploration).
        4. Otherwise, simulate each available move, evaluate its value, and choose the move with the highest value (exploitation).
        5. After executing the move, update the value estimate of the previous board state using the temporal difference (TD) error:
           - TD Error = value(new state) - value(previous state)
           - Update: previous state's value += alpha * TD Error.
        6. Store the new state for the next update.
        """
         # Copy the current state to avoid modifying the original board directly
        outcome = state.copy()
        # Identify indices corresponding to empty board positions (state == 0)
        available = np.where(outcome == 0)[0]
        
        # Decide whether to explore (random move) or exploit (best known move)
        if np.random.binomial(1, self.epsilon):
            # Exploration: choose a random available position and mark it
            outcome[np.random.choice(available)] = self.index
        else:
            # Exploitation: evaluate the outcome of each potential move
            temp_value = np.zeros(len(available))
            for i in range(len(available)):
                temp_outcome = outcome.copy()           # Simulate the board state
                temp_outcome[available[i]] = self.index  # Apply the move
                # Retrieve the estimated value of the simulated state from the value table
                temp_value[i] = self.value[tuple(temp_outcome)]
            # Choose the move that maximizes the estimated value
            best_move = np.argmax(temp_value)
            outcome[available[best_move]] = self.index  # Execute the best move
        
        # Update the value function for the previous state:
        # Calculate the temporal difference (TD) error: difference between new state's value and previous state's value
        error = self.value[tuple(outcome)] - self.value[tuple(self.stored_outcome)]
        # Update the value estimate for the previous state using the learning rate alpha
        self.value[tuple(self.stored_outcome)] += self.alpha * error 
        # Save the new state as the latest observed state for future updates
        self.stored_outcome = outcome.copy()
        
        return outcome 
    
def Judge(outcome, OOXX_Index): 
    """
    Determine the game result for a given player based on the current board state.

    Parameters:
    - outcome (numpy array): Current board state as a 1D array of length 9.
    - OOXX_Index (int): Identifier for the player's mark to check (e.g., 1 for 'O' or 2 for 'X').

    Returns:
    - winner (int): 
        * Returns OOXX_Index if the player meets a winning condition (three in a row).
        * Returns 3 if the board is full with no winner (draw).
        * Returns 0 if the game is still in progress.
    
    Winning conditions are checked for:
    - All rows (positions 0-2, 3-5, 6-8)
    - All columns (positions 0,3,6; 1,4,7; 2,5,8)
    - Both diagonals (positions 0,4,8 and 2,4,6)
    """
    # Create an array representing a winning line for the player (three identical marks)
    triple = np.repeat(OOXX_Index, 3)
    winner = 0  # Default: game still in progress
    
    # Check for a draw: if there are no empty positions left
    if 0 not in outcome:
        winner = 3  # Game is a draw
        
    # Check each row for a win
    if (outcome[0:3] == triple).all() or (outcome[3:6] == triple).all() or (outcome[6:9] == triple).all():
        winner = OOXX_Index
        
    # Check each column for a win
    if (outcome[0:7:3] == triple).all() or (outcome[1:8:3] == triple).all() or (outcome[2:9:3] == triple).all():
        winner = OOXX_Index
        
    # Check both diagonals for a win
    if (outcome[0:9:4] == triple).all() or (outcome[2:7:2] == triple).all():
        winner = OOXX_Index
        
    return winner