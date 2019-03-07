"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

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
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    """relative_mobility (own mobility vs opponent's (normalized [-1.0 ... 1.0]))"""
    my_mobility = len(my_moves)
    opp_mobility = len(opp_moves)
    relative_mobility = ((my_mobility - opp_mobility) /max(my_mobility, opp_mobility))

    """opponent_block_ability (ability to block opponent on next move)"""
    opponent_block_ability = 0
    if game.active_player == player:
        opponent_block_ability = (1 if len(set(my_moves) & set(opp_moves)) != 0 
                                  else 0)
    
    """ corner_domination (Less number of corner moves relative to the opponent 
    is better.)"""
    game_state_factor=1
    """Being in a corner in late game (less than 25% of board empty) is bad"""
    if len(game.get_blank_spaces()) < 0.25*game.width * game.height:
        game_state_factor = 4
    
    """ Four corners """
    corners = [(0, 0),(0, (game.width - 1)),((game.height - 1), 0),
               ((game.height - 1),(game.width - 1))]
    
    my_in_corner = [move for move in my_moves if move in corners]
    opp_in_corner = [move for move in opp_moves if move in corners]
    corner_domination = game_state_factor * (len(opp_in_corner)-len(my_in_corner))
    				 
    """ Score is calculated using a weighted method for each parameter."""
    return float(50 * relative_mobility + 40 * opponent_block_ability + 10*corner_domination)


def custom_score_2(game, player):
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
    if game.is_winner(player):
        return float('inf')
    if game.is_loser(player):
        return float('-inf')

    # get legal moves for both players
    player_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    # check if there is one stealable move
    stealmove_bonus = 0.
    if len(set(player_moves).intersection(opp_moves)) == 1:
      if game.active_player == player:
          stealmove_bonus += 1.
      else:
          stealmove_bonus -= 1.

    return len(player_moves) - len(opp_moves) + stealmove_bonus


def custom_score_3(game, player):
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

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(own_moves - 3 * opp_moves)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

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

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def __min_value(self, game, depth):
        self.__check_time()
        if self.__is_terminal(game, depth):
            return self.score(game, self)
        min_val = float("inf")
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            forecast = game.forecast_move(move)
            min_val = min(min_val, self.__max_value(forecast, depth - 1))
        return min_val

    def __max_value(self, game, depth):

        """
        Method as described in the lectures to define max value (analogous to min value method above)
        """
        self.__check_time()
        if self.__is_terminal(game, depth):
            return self.score(game, self)
        max_val = float("-inf")
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            forecast = game.forecast_move(move)
            max_val = max(max_val, self.__min_value(forecast, depth - 1))
        return max_val

    def __is_terminal(self, game, depth):
        """Helper method to check if we've reached endgame
        """
        self.__check_time()
        if len(game.get_legal_moves()) != 0 and depth > 0:
            return False
        return True

    def __check_time(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def minimax(self, game, depth):
        """Implementation of depth-limited minimax search algorithm.
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        self.__check_time()
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)
        vals = [(self.__min_value(game.forecast_move(m), depth - 1), m) for m in legal_moves]
        _, move = max(vals)
        return move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

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

        #This is pretty much the same as what is implemented for minimax, but updating depth
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)


        depth = 1
        while 1:
            try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
                best_move = self.alphabeta(game, depth)
                depth += 1

            except SearchTimeout:
                break  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move





    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

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

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # Set max value to the smallest possible value
        best_value = float("-inf")
        best_move = ()
      
        # Iterate over all possible candidate moves
        for cand_move in game.get_legal_moves():
            # Obtain copy of game.
            cand_game = game.forecast_move(cand_move)
            cand_value = self.min_value(cand_game, depth-1, alpha, beta)
            # Update best_move and max_value if cand_value has max value
            if cand_value > best_value:
                best_move, best_value = cand_move, cand_value
            # Best move found.
            if best_value >= beta:
                break
            # Update lower bound for pruning
            alpha = max(alpha, best_value)
        return best_move

    def max_value(self, game, depth, alpha, beta):
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return self.score(game, self)

        value = float("-inf")

        for move in game.get_legal_moves():
            value = max(value, self.min_value(game.forecast_move(move), depth-1, alpha, beta))
            # Update lower bound
            alpha = max(alpha, value)
            if value >= beta:
                return value # Found candidate upper value
        return value

    def min_value(self, game, depth, alpha, beta):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return self.score(game, self)

        value = float("inf")

        for move in game.get_legal_moves():
            value = min(value, self.max_value(game.forecast_move(move), depth-1, alpha, beta))
            if value <= alpha:
                return value # Found candidate lower value
            # Update upper bound
            beta = min(beta, value)
        return value

