import tkinter as tk
from tkinter import messagebox
import json
import os
import random
import math

# --- Configuration ---
BOARD_SIZE = 15
WIN_LENGTH = 5
CELL_SIZE = 40
HUMAN_PLAYER = 1  # X
AI_PLAYER = 2     # O
EMPTY_CELL = 0

# Initial weights for the evaluation function. These will be tuned.
INITIAL_WEIGHTS = {
    'own_five': 10000000,  # Winning state
    'opp_five': -100000000, # Opponent winning state (block urgently)
    'own_open_four': 50000,
    'opp_open_four': -1000000, # Must block
    'own_half_open_four': 5000,
    'opp_half_open_four': -10000,
    'own_open_three': 1000,
    'opp_open_three': -8000, # Important to block
    'own_half_open_three': 200,
    'opp_half_open_three': -500,
    'own_open_two': 50,
    'opp_open_two': -100,
    'own_half_open_two': 10,
    'opp_half_open_two': -20,
    'own_piece_near_center': 5, # Slight preference for center
    'opp_piece_near_center': -5
}
WEIGHTS_FILE = 'gomoku_weights.json'
MAX_DEPTH = 2 # Minimax search depth. Increase for stronger AI, but slower moves.

# --- Gomoku Board Logic ---
class GomokuBoard:
    def __init__(self, size=BOARD_SIZE):
        self.size = size
        self.board = [[EMPTY_CELL for _ in range(size)] for _ in range(size)]
        self.current_winner = None
        self.moves_history = [] # For potential rollback or advanced learning

    def make_move(self, row, col, player):
        if 0 <= row < self.size and 0 <= col < self.size and self.board[row][col] == EMPTY_CELL:
            self.board[row][col] = player
            self.moves_history.append(((row, col), player))
            if self.check_win_at(row, col, player):
                self.current_winner = player
            return True
        return False

    def undo_last_move(self):
        if not self.moves_history:
            return False
        (row, col), player = self.moves_history.pop()
        self.board[row][col] = EMPTY_CELL
        self.current_winner = None # Reset winner status as it might change
        return True

    def get_empty_cells(self):
        empty = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == EMPTY_CELL:
                    empty.append((r, c))
        return empty
    
    def is_full(self):
        return not any(EMPTY_CELL in row for row in self.board)

    def check_win_at(self, r, c, player):
        # Check all 4 directions from the last placed stone
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)] # Horizontal, Vertical, Diag_down_right, Diag_up_right
        for dr, dc in directions:
            count = 1
            # Check in one direction
            for i in range(1, WIN_LENGTH):
                nr, nc = r + dr * i, c + dc * i
                if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr][nc] == player:
                    count += 1
                else:
                    break
            # Check in opposite direction
            for i in range(1, WIN_LENGTH):
                nr, nc = r - dr * i, c - dc * i
                if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr][nc] == player:
                    count += 1
                else:
                    break
            if count >= WIN_LENGTH:
                return True
        return False

    def get_board_state_copy(self):
        return [row[:] for row in self.board]

    def print_board(self): # For debugging
        for row in self.board:
            print(" ".join(map(str, row)))
        print("-" * self.size * 2)

# --- Evaluation Function and RL-based Tuner ---
class Evaluation:
    def __init__(self, weights_file=WEIGHTS_FILE):
        self.weights_file = weights_file
        self.weights = self._load_weights()

    def _load_weights(self):
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Error decoding {self.weights_file}. Using initial weights.")
                return INITIAL_WEIGHTS.copy() # Return a copy
        return INITIAL_WEIGHTS.copy() # Return a copy

    def save_weights(self):
        try:
            with open(self.weights_file, 'w') as f:
                json.dump(self.weights, f, indent=4)
        except IOError:
            print(f"Error: Could not save weights to {self.weights_file}")

    def _count_patterns_in_line(self, line, player):
        """ Counts patterns for 'player' in a given 'line' of stones. """
        patterns = {
            'five': 0, 'open_four': 0, 'half_open_four': 0,
            'open_three': 0, 'half_open_three': 0,
            'open_two': 0, 'half_open_two': 0
        }
        opponent = AI_PLAYER if player == HUMAN_PLAYER else HUMAN_PLAYER
        n = len(line)

        # Fives
        for i in range(n - WIN_LENGTH + 1):
            if all(line[j] == player for j in range(i, i + WIN_LENGTH)):
                patterns['five'] += 1
                return patterns # If five is found, it's the most important

        # Fours
        for i in range(n - 3): # Check for X X X X
            # Open Four: _XXXX_
            if i > 0 and i + 4 < n and line[i-1] == EMPTY_CELL and line[i+4] == EMPTY_CELL and \
               all(line[j] == player for j in range(i, i + 4)):
                patterns['open_four'] += 1
            # Half Open Four: OXXXX_ or _XXXXO or X_XXX or XX_XX or XXX_X
            # Simplified: Check for exactly 4 player stones in a sequence of 5, with one empty.
            # Or 4 player stones at ends of a 6-long sequence with two empties.
            # This part can be very complex to get all cases. Let's simplify.
            # Count sequences of 4 player stones.
            if all(line[j] == player for j in range(i, i + 4)):
                # Check left boundary for half-open
                left_open = (i > 0 and line[i-1] == EMPTY_CELL)
                left_blocked = (i == 0 or line[i-1] == opponent)
                # Check right boundary for half-open
                right_open = (i + 4 < n and line[i+4] == EMPTY_CELL)
                right_blocked = (i + 4 >= n or line[i+4] == opponent)

                if left_open and right_open: # Already caught by open_four if it's _XXXX_
                    pass # patterns['open_four'] +=1 (This logic is tricky to avoid double counts)
                elif (left_open and right_blocked) or (right_open and left_blocked):
                    patterns['half_open_four'] += 1
        
        # Threes (similar logic for open and half-open)
        for i in range(n - 2): # X X X
            if all(line[j] == player for j in range(i, i + 3)):
                left_open = (i > 0 and line[i-1] == EMPTY_CELL)
                right_open = (i + 3 < n and line[i+3] == EMPTY_CELL)
                
                # Check for one more empty cell for true open three: _XXX_
                # And not part of an open four like _XXX_X
                is_true_open_three = False
                if left_open and right_open:
                    # Check if it's _XXX_ and not _XXXP_ or _PXXX_
                    # Ensure the cells beyond the empty ones are not player stones (to avoid counting parts of fours)
                    # _ E P P P E _
                    #   ^       ^
                    # i-1 i   i+2 i+3
                    if not (i > 1 and line[i-2] == player) and \
                       not (i + 4 < n and line[i+4] == player):
                        patterns['open_three'] += 1
                        is_true_open_three = True

                if not is_true_open_three: # If not a true open three, check for half-open
                    # OXXX_ or _XXXO
                    left_blocked = (i == 0 or line[i-1] == opponent)
                    right_blocked = (i + 3 >= n or line[i+3] == opponent)
                    if (left_open and right_blocked) or (right_open and left_blocked):
                        patterns['half_open_three'] += 1
        
        # Twos (similar logic)
        for i in range(n - 1): # X X
            if all(line[j] == player for j in range(i, i + 2)):
                left_open = (i > 0 and line[i-1] == EMPTY_CELL)
                right_open = (i + 2 < n and line[i+2] == EMPTY_CELL)
                
                is_true_open_two = False
                if left_open and right_open:
                     # _XX_ , not part of _XXP_ or _PXX_
                    if not (i > 1 and line[i-2] == player) and \
                       not (i + 3 < n and line[i+3] == player):
                        patterns['open_two'] += 1
                        is_true_open_two = True

                if not is_true_open_two:
                    left_blocked = (i == 0 or line[i-1] == opponent)
                    right_blocked = (i + 2 >= n or line[i+2] == opponent)
                    if (left_open and right_blocked) or (right_open and left_blocked):
                        patterns['half_open_two'] += 1
        return patterns


    def _extract_features(self, board_state, player_perspective):
        """
        Extracts features from the board_state from the perspective of player_perspective.
        Returns a dictionary like {'own_open_threes': count, 'opp_open_threes': count, ...}
        """
        features = {key: 0 for key in self.weights.keys()} # Initialize all feature counts to 0
        opponent_perspective = AI_PLAYER if player_perspective == HUMAN_PLAYER else HUMAN_PLAYER
        board_size = len(board_state)

        lines_to_check = []
        # Horizontal lines
        for r in range(board_size):
            lines_to_check.append(board_state[r])
        # Vertical lines
        for c in range(board_size):
            lines_to_check.append([board_state[r][c] for r in range(board_size)])
        # Diagonal lines (top-left to bottom-right)
        for k in range(-(board_size - WIN_LENGTH), board_size - WIN_LENGTH + 1):
            lines_to_check.append([board_state[r][r - k] for r in range(board_size) if 0 <= r - k < board_size])
        # Diagonal lines (top-right to bottom-left)
        for k in range(WIN_LENGTH - 1, 2 * board_size - WIN_LENGTH):
            lines_to_check.append([board_state[r][k - r] for r in range(board_size) if 0 <= k - r < board_size])

        for line in lines_to_check:
            if len(line) < WIN_LENGTH: continue

            own_patterns = self._count_patterns_in_line(line, player_perspective)
            opp_patterns = self._count_patterns_in_line(line, opponent_perspective)

            for pattern_type, count in own_patterns.items():
                features[f'own_{pattern_type}'] = features.get(f'own_{pattern_type}', 0) + count
            for pattern_type, count in opp_patterns.items():
                features[f'opp_{pattern_type}'] = features.get(f'opp_{pattern_type}', 0) + count
        
        # Center control (simple version)
        center_r, center_c = board_size // 2, board_size // 2
        for r_offset in range(-2, 3): # Check a 5x5 area around center
            for c_offset in range(-2, 3):
                r, c = center_r + r_offset, center_c + c_offset
                if 0 <= r < board_size and 0 <= c < board_size:
                    if board_state[r][c] == player_perspective:
                        features['own_piece_near_center'] = features.get('own_piece_near_center', 0) + 1
                    elif board_state[r][c] == opponent_perspective:
                        features['opp_piece_near_center'] = features.get('opp_piece_near_center', 0) + 1
        return features


    def evaluate_board(self, gomoku_board, player_to_maximize):
        """ Evaluates the board state for player_to_maximize. """
        # Check for immediate win/loss first
        if gomoku_board.current_winner == player_to_maximize:
            return self.weights['own_five'] 
        opponent = AI_PLAYER if player_to_maximize == HUMAN_PLAYER else HUMAN_PLAYER
        if gomoku_board.current_winner == opponent:
            return self.weights['opp_five'] 
        if gomoku_board.is_full():
            return 0 # Draw

        board_state = gomoku_board.board # Use the actual board object's state
        features = self._extract_features(board_state, player_to_maximize)
        
        score = 0
        for feature_name, count in features.items():
            score += self.weights.get(feature_name, 0) * count
        
        return score

    def update_weights(self, final_board_obj, ai_player_id, outcome, learning_rate=0.01):
        """
        Updates weights based on the game outcome.
        outcome: 1 if ai_player_id wins, -1 if ai_player_id loses, 0 for draw.
        """
        if outcome == 0: # No learning on draw for this simple version
            return

        final_board_state = final_board_obj.board
        features = self._extract_features(final_board_state, ai_player_id)
        
        # Target score: High positive for win, high negative for loss.
        # This is a simplified approach. A more robust method might use the difference
        # between predicted score and actual outcome (e.g. Temporal Difference learning).

        for feature_name, count in features.items():
            if count == 0:
                continue

            current_weight = self.weights.get(feature_name, 0)
            adjustment = 0

            # If AI won, reinforce features that were present.
            # If AI lost, penalize features that were present (for AI) or make opponent's features seem more dangerous.
            if outcome == 1: # AI won
                if "own_" in feature_name: # AI's positive pattern
                    adjustment = learning_rate * count * abs(INITIAL_WEIGHTS.get(feature_name, 1)) # Scale by initial magnitude
                elif "opp_" in feature_name: # Opponent's pattern (AI successfully dealt with it or it wasn't enough for opp)
                                            # Make it more negative (AI should value blocking/countering it more)
                    adjustment = -learning_rate * count * abs(INITIAL_WEIGHTS.get(feature_name, 1))
            
            elif outcome == -1: # AI lost
                if "own_" in feature_name: # AI had this pattern but lost
                    adjustment = -learning_rate * count * abs(INITIAL_WEIGHTS.get(feature_name, 1))
                elif "opp_" in feature_name: # Opponent had this pattern and AI lost
                                            # Make it significantly more negative
                    adjustment = -learning_rate * count * abs(INITIAL_WEIGHTS.get(feature_name, 1)) * 2 # Stronger penalty

            self.weights[feature_name] = current_weight + adjustment
            
            # Prevent weights from drifting too far or changing sign unexpectedly for critical features
            if "five" in feature_name: # Win/loss conditions should remain dominant
                if "own_five" in feature_name and self.weights[feature_name] < 0: self.weights[feature_name] = INITIAL_WEIGHTS['own_five']
                if "opp_five" in feature_name and self.weights[feature_name] > 0: self.weights[feature_name] = INITIAL_WEIGHTS['opp_five']

        print(f"Weights updated after game. Outcome for AI: {outcome}")
        self.save_weights()


# --- AI Player (Minimax) ---
class AIPlayer:
    def __init__(self, player_id, evaluator, depth=MAX_DEPTH):
        self.player_id = player_id
        self.evaluator = evaluator
        self.depth = depth

    def minimax(self, board_obj, depth, alpha, beta, maximizing_player):
        if depth == 0 or board_obj.current_winner is not None or board_obj.is_full():
            return self.evaluator.evaluate_board(board_obj, self.player_id), None

        possible_moves = self._get_ordered_moves(board_obj, maximizing_player)
        
        best_move = random.choice(possible_moves) if possible_moves else None

        if maximizing_player:
            max_eval = -float('inf')
            for r, c in possible_moves:
                board_obj.make_move(r, c, self.player_id)
                current_eval, _ = self.minimax(board_obj, depth - 1, alpha, beta, False)
                board_obj.undo_last_move()
                if current_eval > max_eval:
                    max_eval = current_eval
                    best_move = (r, c)
                alpha = max(alpha, current_eval)
                if beta <= alpha:
                    break # Beta cut-off
            return max_eval, best_move
        else: # Minimizing player (opponent)
            min_eval = float('inf')
            opponent_id = HUMAN_PLAYER if self.player_id == AI_PLAYER else AI_PLAYER
            for r, c in possible_moves:
                board_obj.make_move(r, c, opponent_id)
                current_eval, _ = self.minimax(board_obj, depth - 1, alpha, beta, True)
                board_obj.undo_last_move()
                if current_eval < min_eval:
                    min_eval = current_eval
                    best_move = (r, c) # Not really used by minimizer's return, but good for structure
                beta = min(beta, current_eval)
                if beta <= alpha:
                    break # Alpha cut-off
            return min_eval, best_move

    def _get_ordered_moves(self, board_obj, for_player):
        """
        Get possible moves, ordered by a quick heuristic.
        Prioritize moves that create threats or block opponent's threats.
        """
        empty_cells = board_obj.get_empty_cells()
        if not empty_cells:
            return []

        opponent = HUMAN_PLAYER if for_player == AI_PLAYER else AI_PLAYER
        
        # Simple heuristic: check moves that are adjacent to existing stones
        candidate_moves = set()
        for r_idx, row in enumerate(board_obj.board):
            for c_idx, cell_val in enumerate(row):
                if cell_val != EMPTY_CELL:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r_idx + dr, c_idx + dc
                            if 0 <= nr < board_obj.size and 0 <= nc < board_obj.size and \
                               board_obj.board[nr][nc] == EMPTY_CELL:
                                candidate_moves.add((nr, nc))
        
        if not candidate_moves: # If board is empty or no adjacent cells
            # Play near center if board is sparse
            if len(board_obj.moves_history) < 3:
                center_r, center_c = board_obj.size // 2, board_obj.size // 2
                if board_obj.board[center_r][center_c] == EMPTY_CELL:
                    return [(center_r, center_c)]
                else: # try around center
                    potential_center_moves = []
                    for dr in [-1,0,1]:
                        for dc in [-1,0,1]:
                            if dr==0 and dc==0: continue
                            nr, nc = center_r+dr, center_c+dc
                            if 0 <= nr < board_obj.size and 0 <= nc < board_obj.size and \
                               board_obj.board[nr][nc] == EMPTY_CELL:
                                potential_center_moves.append((nr,nc))
                    if potential_center_moves: return random.sample(potential_center_moves, len(potential_center_moves))


            return random.sample(empty_cells, len(empty_cells)) # Random order if no other heuristic

        # Further refinement: Score candidate moves quickly
        # (This can be a simplified version of the main evaluation)
        # For now, just use the candidate set or random if it's empty
        
        # Check for immediate win/block moves (highest priority)
        priority_moves = []
        for r, c in candidate_moves:
            # Check if this move wins for 'for_player'
            board_obj.make_move(r, c, for_player)
            if board_obj.check_win_at(r,c,for_player):
                priority_moves.append(((r,c), float('inf'))) # Winning move
            board_obj.undo_last_move()

            # Check if this move blocks opponent's win
            board_obj.make_move(r,c,opponent)
            if board_obj.check_win_at(r,c,opponent):
                 priority_moves.append(((r,c), float('inf')-1)) # Blocking move
            board_obj.undo_last_move()
        
        if priority_moves:
            priority_moves.sort(key=lambda x: x[1], reverse=True)
            return [move for move, score in priority_moves]

        return random.sample(list(candidate_moves), len(candidate_moves))


    def find_best_move(self, board_obj):
        _, move = self.minimax(board_obj, self.depth, -float('inf'), float('inf'), True)
        if move is None: # Should not happen if there are empty cells
            empty_cells = board_obj.get_empty_cells()
            if empty_cells:
                return random.choice(empty_cells)
        return move

# --- Game GUI (Tkinter) ---
class GomokuGUI:
    def __init__(self, master):
        self.master = master
        master.title("Gomoku with Minimax & RL-Tuned Eval")

        self.board_logic = GomokuBoard(BOARD_SIZE)
        self.evaluator = Evaluation(WEIGHTS_FILE)
        self.ai_player = AIPlayer(AI_PLAYER, self.evaluator, depth=MAX_DEPTH)
        
        self.current_player = HUMAN_PLAYER
        self.game_over = False

        self.canvas = tk.Canvas(master, width=BOARD_SIZE * CELL_SIZE, height=BOARD_SIZE * CELL_SIZE, bg='burlywood')
        self.canvas.pack(pady=20)
        self.canvas.bind("<Button-1>", self.handle_click)

        self.status_label = tk.Label(master, text="Your turn (X)", font=('Arial', 16))
        self.status_label.pack()

        self.reset_button = tk.Button(master, text="Reset Game", command=self.reset_game, font=('Arial', 14))
        self.reset_button.pack(pady=10)
        
        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        # Draw grid lines
        for i in range(BOARD_SIZE):
            self.canvas.create_line(i * CELL_SIZE, 0, i * CELL_SIZE, BOARD_SIZE * CELL_SIZE)
            self.canvas.create_line(0, i * CELL_SIZE, BOARD_SIZE * CELL_SIZE, i * CELL_SIZE)
        
        # Draw stones
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                player = self.board_logic.board[r][c]
                if player != EMPTY_CELL:
                    x0, y0 = c * CELL_SIZE + CELL_SIZE // 10, r * CELL_SIZE + CELL_SIZE // 10
                    x1, y1 = (c + 1) * CELL_SIZE - CELL_SIZE // 10, (r + 1) * CELL_SIZE - CELL_SIZE // 10
                    color = "black" if player == HUMAN_PLAYER else "white" # Human (X) is black, AI (O) is white
                    self.canvas.create_oval(x0, y0, x1, y1, fill=color, outline="gray")

    def handle_click(self, event):
        if self.game_over or self.current_player != HUMAN_PLAYER:
            return

        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE

        if self.board_logic.make_move(row, col, HUMAN_PLAYER):
            self.draw_board()
            if self.check_game_status():
                return
            
            self.current_player = AI_PLAYER
            self.status_label.config(text="AI's turn (O)...")
            self.master.after(500, self.ai_move) # Give a slight delay for UI update

    def ai_move(self):
        if self.game_over:
            return

        # print("AI is thinking...")
        # self.board_logic.print_board() # Debug
        ai_r, ai_c = self.ai_player.find_best_move(self.board_logic)
        
        if ai_r is not None and ai_c is not None:
            self.board_logic.make_move(ai_r, ai_c, AI_PLAYER)
            self.draw_board()
            if self.check_game_status():
                return
        else: # Should not happen if game not over
            print("AI couldn't find a move.")
            if self.board_logic.is_full() and not self.board_logic.current_winner: # Check for draw if AI passes
                 self.end_game("It's a Draw!")
                 self.evaluator.update_weights(self.board_logic, AI_PLAYER, 0) # Outcome 0 for draw
                 return


        self.current_player = HUMAN_PLAYER
        self.status_label.config(text="Your turn (X)")

    def check_game_status(self):
        winner = self.board_logic.current_winner
        if winner:
            outcome_for_ai = 0
            if winner == HUMAN_PLAYER:
                self.end_game("You (X) win!")
                outcome_for_ai = -1 # AI lost
            else: # AI wins
                self.end_game("AI (O) wins!")
                outcome_for_ai = 1 # AI won
            self.evaluator.update_weights(self.board_logic, AI_PLAYER, outcome_for_ai)
            return True
        elif self.board_logic.is_full():
            self.end_game("It's a Draw!")
            self.evaluator.update_weights(self.board_logic, AI_PLAYER, 0) # Outcome 0 for draw
            return True
        return False

    def end_game(self, message):
        self.game_over = True
        self.status_label.config(text=message)
        messagebox.showinfo("Game Over", message)

    def reset_game(self):
        self.board_logic = GomokuBoard(BOARD_SIZE)
        # self.evaluator = Evaluation(WEIGHTS_FILE) # Re-load weights, or keep tuned ones? Keep tuned.
        # self.ai_player = AIPlayer(AI_PLAYER, self.evaluator, depth=MAX_DEPTH) # AI keeps its learned weights
        
        self.current_player = HUMAN_PLAYER
        self.game_over = False
        self.status_label.config(text="Your turn (X)")
        self.draw_board()

# --- Main ---
if __name__ == "__main__":
    root = tk.Tk()
    gui = GomokuGUI(root)
    root.mainloop()
