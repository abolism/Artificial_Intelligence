from Board import BoardUtility
import random


class Player:
    def __init__(self, player_piece):
        self.piece = player_piece

    def play(self, board):
        return 0


class RandomPlayer(Player):
    def play(self, board):
        return random.choice(BoardUtility.get_valid_locations(board))


class HumanPlayer(Player):
    def play(self, board):
        move = int(input("input the next column index 0 to 8:"))
        return move


class MiniMaxPlayer(Player):
    def __init__(self, player_piece, depth=5):
        super().__init__(player_piece)
        self.depth = depth

    def play(self, board):
        """
        Inputs : 
           board : 7*9 numpy array. 0 for empty cell, 1 and 2 for cells containig a piece.
        return the next move(columns to play in) of the player based on minimax algorithm.
        """
        # Todo: implement minimax algorithm with alpha beta pruning
        # Your code here
        # return 0
        return self.minimax(board, self.depth, self.piece, -float('inf'), float('inf'))[1]


        # move = ...
        # return move

    def minimax(self, board, depth, player, alpha, beta):
        if depth == 0 or BoardUtility.is_terminal_state(board):
            return BoardUtility.score_position(board, self.piece), None

        valid_locations = BoardUtility.get_valid_locations(board)
        random.shuffle(valid_locations)
        best_score = -float('inf') if player == self.piece else float('inf')
        best_column = random.choice(valid_locations)

        for column in valid_locations:
            row = BoardUtility.get_next_open_row(board, column)
            b_copy = board.copy()
            BoardUtility.make_move(b_copy, column, player)
            score = self.minimax(b_copy, depth - 1, 3 - player, alpha, beta)[0]

            if player == self.piece:
                if score > best_score:
                    best_score = score
                    best_column = column
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            else:
                if score < best_score:
                    best_score = score
                    best_column = column
                beta = min(beta, score)
                if alpha >= beta:
                    break
        return best_score, best_column

    
class MiniMaxProbPlayer(Player):
    def __init__(self, player_piece, depth=5, prob_stochastic=0.1):
        super().__init__(player_piece)
        self.depth = depth
        self.prob_stochastic = prob_stochastic

    def play(self, board):
        """
        Inputs : 
           board : 7*9 numpy array. 0 for empty cell, 1 and 2 for cells containig a piece.
        same as above but each time you are playing as max choose a random move instead of the best move
        with probability self.prob_stochastic.
        """


        # Todo: implement minimax algorithm with alpha beta pruning
        # Your code here
        # return 0
        return self.minimax(board, self.depth, self.piece, -float('inf'), float('inf'))[1]

        # move = ...
        # return move

    def minimax(self, board, depth, player, alpha, beta):
        if depth == 0 or BoardUtility.is_terminal_state(board):
            return BoardUtility.score_position(board, self.piece), None

        valid_locations = BoardUtility.get_valid_locations(board)
        random.shuffle(valid_locations)
        best_score = -float('inf') if player == self.piece else float('inf')
        best_column = random.choice(valid_locations)

        for column in valid_locations:
            row = BoardUtility.get_next_open_row(board, column)
            b_copy = board.copy()
            BoardUtility.make_move(b_copy, column, player)
            score = self.minimax(b_copy, depth - 1, 3 - player, alpha, beta)[0]

            if player == self.piece:
                prob = random.random()
                if prob < self.prob_stochastic:
                    best_score = score
                    best_column = column
                elif score > best_score:
                    best_score = score
                    prob = random.random()
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            else:
                if score < best_score:
                    best_score = score
                    best_column = column
                beta = min(beta, score)
                if alpha >= beta:
                    break
        return best_score, best_column
