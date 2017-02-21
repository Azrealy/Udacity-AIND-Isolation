"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import sample_players

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

## Heuristic 1 ###

def my_moves_vs_opponent_moves(game, player, factor=0):
    """
    Returns a score for a game state and player based on his moves and
    opponent moves
    :param game:
    :param player:
    :param factor: weight of opponent moves
    :return:
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(my_moves - factor*opponent_moves)

## Heuristic 2 ###

def common_moves(game, player, factor=1):
    """
    Returns a score for a game and player based on the intersection of player
    legal moves and opponent legal moves.
    :param game:
    :param player:
    :param factor: weight of the common moves
    :return:
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))

    common = set(my_moves).intersection(set(opponent_moves))

    return float(len(my_moves)-len(opponent_moves)+factor*len(common))

## Heuristic 3 Testing ###

def center_opening(game, player, factor=2):
    """
    Return a score for game and player based on the stage of the game, my moves,
    oponent moves and moves targeting the center of the board.
    If the game is in early state, gives weight to center moves
    :param game:
    :param player:
    :param factor: move quantity threshold for center opening
    :return:
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    if (game.move_count < factor):
        location = game.get_player_location(player)
        opponent_moves = game.get_legal_moves(game.get_opponent(player))

        if (location[0] >= 2 and location[0] < game.height - 2) and\
            (location[1] >= 2 and location[1] < game.width - 2):
            return float(100 - len(opponent_moves))
        else:
            my_moves = game.get_legal_moves(player)

            return float(len(my_moves) - len(opponent_moves))
    else:
        my_moves = game.get_legal_moves(player)
        opponent_moves = game.get_legal_moves(game.get_opponent(player))

        return float(len(my_moves) - len(opponent_moves))

## Heuristic Mixed ###

def final_heuristic(game, player, startfactor=2, factor=3):
    """
        Return a score for game and player based on the stage of the game, my moves,
        oponent moves (with weight) and moves targeting the center of the board.
        :param game:
        :param player:
        :param startfactor: move quantity threshold for center opening
        :param factor: opponent moves weight
        :return:
        """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    if (game.move_count < factor):
        location = game.get_player_location(player)
        opponent_moves = game.get_legal_moves(game.get_opponent(player))

        if (location[0] >= 2 and location[0] < game.height - 2) and\
            (location[1] >= 2 and location[1] < game.width - 2):
            return float(100 - len(opponent_moves))
        else:
            my_moves = game.get_legal_moves(player)

            return float(len(my_moves) + factor*len(opponent_moves))
    else:
        my_moves = game.get_legal_moves(player)
        opponent_moves = game.get_legal_moves(game.get_opponent(player))

        return float(len(my_moves) + factor*len(opponent_moves))


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    return final_heuristic(game,player,5,-3)

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        method = getattr(self, self.method)
        best_move = (-1,-1)

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            if self.iterative:
                depth = 1
                selected = [(float("-inf"),(-1,-1))]

                while (True):
                    move = method(game, depth)
                    selected.append(move)
                    depth = depth+1


            else:
                _,best_move = method(game,self.search_depth)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            if self.iterative:
                best_move = max(selected)[1]
                # Return the best move from the last completed search iteration
            return best_move


        if self.iterative:
            best_move = max(selected)[1]
        # Return the best move from the last completed search iteration
        return best_move

        # raise NotImplementedError

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # keeps dfs visited path
        path = []
        score, path = self.minimax_max_value(game, depth, path)

        # return first node of the path
        return score, path[0]


    def minimax_max_value(self, game, depth, path, maximizing_player=True):
        # Auxiliary for max nodes
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth == 0:
            # Returns score
            return self.score(game, self), path

        candidate = (float("-inf"), [(-1, -1)])

        #Get possible moves
        possible_moves = game.get_legal_moves(game.active_player)

        for move in possible_moves:
            # Tries each move
            forecast_game = game.forecast_move(move)

            # Gets move score and path (recursion)
            move_score = self.minimax_min_value(forecast_game, depth - 1, path,not maximizing_player)[0]

            # If it better keeps
            if move_score > candidate[0]:
                candidate = (move_score, path + [move])

        return candidate

    def minimax_min_value(self, game, depth, path,maximizing_player=True):
        # Auxiliary for min nodes
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth == 0:
            return self.score(game, self), path

        candidate = (float("inf"), [(-1, -1)])
        # Get possible moves
        possible_moves = game.get_legal_moves(game.active_player)

        for move in possible_moves:
            # Tries each move
            forecast_game = game.forecast_move(move)

            # Gets move score and path (recursion)
            move_score = self.minimax_max_value(forecast_game, depth - 1, path, not maximizing_player)[0]

            # If it better keeps
            if move_score < candidate[0]:
                candidate = (move_score, move)

        return candidate

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        path = []
        # keeps dfs visited path
        value, path = self.alphabeta_max_value(game,depth, path, alpha, beta, maximizing_player)

        if len(path) == 0:
            return float("-inf"),(-1,-1)

        # returns score and first visit of path
        return value, path[0]


    def alphabeta_max_value(self, game, depth, path, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth == 0:
            return self.score(game, self), path

        candidate = (float("-inf"), [(-1,-1)])
        possible_moves = game.get_legal_moves(game.active_player)

        for move in possible_moves:
            # Tries each move
            forecast_game = game.forecast_move(move)

            move_score = self.alphabeta_min_value(forecast_game, depth-1, path, alpha, beta, not maximizing_player)[0]

            # If it better keeps
            if move_score > candidate[0]:
                candidate = (move_score, path+[move])

            #pruning
            if move_score >= beta:
                return move_score, path+[move]

            alpha = max(alpha,move_score)

        return candidate

    def alphabeta_min_value(self, game, depth, path, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth == 0:
            return self.score(game, self), path

        candidate = (float("inf"),[(-1,-1)])
        possible_moves = game.get_legal_moves(game.active_player)

        for move in possible_moves:
            # Tries each move
            forecast_game = game.forecast_move(move)

            move_score = self.alphabeta_max_value(forecast_game, depth - 1, path, alpha, beta, not maximizing_player)[0]

            # If it better keeps
            if move_score < candidate[0]:
                candidate = (move_score,move)

            # pruning
            if candidate[0] <= alpha:
                return move_score,path+[move]

            beta = min(beta, move_score)

        return candidate





