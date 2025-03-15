from OOXX import Agent, Judge
import numpy as np

# Create two Tic-Tac-Toe agents
Agent1 = Agent(OOXX_Index=1, epsilon=0.1, alpha=0.1)
Agent2 = Agent(OOXX_Index=2, epsilon=0.1, alpha=0.1)

# Set the total number of training games
Trial = 30000  # Train for 30,000 games
Winner = np.zeros(Trial)  # Array to record the outcome of each game

for i in range(Trial):
    # After 20,000 games, disable exploration by setting epsilon to 0
    if i == 20000:
        Agent1.epsilon = 0
        Agent2.epsilon = 0
    
    # Reset each agent's internal state for the new game
    Agent1.reset()
    Agent2.reset()
    
    winner = 0  # Initialize game outcome: 0 means the game is still in progress
    # Initialize the game board as an array of 9 zeros (empty positions)
    State = np.zeros(9, dtype=np.int8)
    
    # Agent1 always takes the first move; the board state is defined from Agent1's perspective
    while winner == 0:
        # Agent1 makes a move and updates its value function based on the current state
        Outcome = Agent1.move(State)
        # Check if Agent1 has achieved a winning condition after its move
        winner = Judge(Outcome, 1)
        
        if winner == 1:  # If Agent1 wins:
            # Set the value of the winning board state to 1 for Agent1
            Agent1.value[tuple(Outcome)] = 1
            # Set the value of the previous state (from Agent1's perspective) to -1 for Agent2
            Agent2.value[tuple(State)] = -1
        
        elif winner == 0:  # If the game is still undecided:
            # Agent2 makes a move from the board state after Agent1's move and updates its value function
            State = Agent2.move(Outcome)
            # Check if Agent2 has won the game after its move
            winner = Judge(State, 2)
            if winner == 2:  # If Agent2 wins:
                # Set the value of the winning board state to 1 for Agent2
                Agent2.value[tuple(State)] = 1
                # Set the value of the previous state (from Agent1's perspective) to -1 for Agent1
                Agent1.value[tuple(Outcome)] = -1
    
    # Record the final outcome of the game:
    #   1 -> Agent1 wins, 2 -> Agent2 wins, 3 -> Draw
    Winner[i] = winner

import matplotlib.pyplot as plt

# Define parameters to calculate win rates over intervals of games
step = 250      # Recompute win rates every 250 games
duration = 500  # Calculate win rate over a moving window of 500 games

def Rate(Winner):
    """
    Compute the win and draw rates over intervals of the training games.

    Parameters:
        Winner (numpy array): Array containing the outcome of each game, where:
            - 1 indicates a win for Agent1,
            - 2 indicates a win for Agent2,
            - 3 indicates a draw.

    Returns:
        Rate1 (numpy array): Win rate of Agent1 for each window.
        Rate2 (numpy array): Win rate of Agent2 for each window.
        Rate3 (numpy array): Draw rate for each window.
    """
    num_windows = int((Trial - duration) / step) + 1
    Rate1 = np.zeros(num_windows)  # Win rate for Agent1
    Rate2 = np.zeros(num_windows)  # Win rate for Agent2
    Rate3 = np.zeros(num_windows)  # Draw rate

    # Calculate win/draw rates for each window of games
    for i in range(num_windows):
        window = Winner[step * i: duration + step * i]
        Rate1[i] = np.sum(window == 1) / duration  # Fraction of games Agent1 wins
        Rate2[i] = np.sum(window == 2) / duration  # Fraction of games Agent2 wins
        Rate3[i] = np.sum(window == 3) / duration  # Fraction of games that end in a draw

    return Rate1, Rate2, Rate3

Rate1,Rate2,Rate3=Rate(Winner)

fig,ax=plt.subplots(figsize=(16,9))
plt.plot(Rate1,linewidth=4,marker='.',markersize=20,color="#0071B7",label="Agent1")
plt.plot(Rate2,linewidth=4,marker='.',markersize=20,color="#DB2C2C",label="Agent2")
plt.plot(Rate3,linewidth=4,marker='.',markersize=20,color="#FAB70D",label="Draw")
plt.xticks(np.arange(0,121,40),np.arange(0,31+1,10),fontsize=30)
plt.yticks(np.arange(0,1.1,0.2),np.round(np.arange(0,1.1,0.2),2),fontsize=30)
plt.xlabel("Trials(x1k)",fontsize=30)
plt.ylabel("Winning Rate",fontsize=30)
plt.legend(loc="best",fontsize=25)
plt.tick_params(width=4,length=10)
ax.spines[:].set_linewidth(4)
plt.show()