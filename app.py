"""
Tic-Tac-Toe Game with AI Agents using OpenAI Agents SDK and Gradio

This application features two GPT-4o-mini agents competing in Tic-Tac-Toe,
each using optimal strategy to win or force a draw. The starting agent (X)
makes a random first move to add variety and unpredictability to games.
"""

import gradio as gr
from agents import Agent, Runner
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import json
import os
import asyncio
import threading
import time
import random


class Player(Enum):
    """Enumeration for game players."""
    X = "X"
    O = "O"
    EMPTY = " "


@dataclass
class GameState:
    """Represents the current state of the Tic-Tac-Toe game."""
    board: List[List[Player]]
    current_player: Player
    move_count: int
    game_over: bool
    winner: Optional[Player]
    move_log: List[Tuple[int, int, Player]]


class TicTacToeGame:
    """Manages the Tic-Tac-Toe game logic and state."""
    
    def __init__(self):
        """Initialize a new game."""
        self.reset_game()
    
    def reset_game(self) -> None:
        """Reset the game to initial state."""
        self.state = GameState(
            board=[[Player.EMPTY for _ in range(3)] for _ in range(3)],
            current_player=Player.X,
            move_count=0,
            game_over=False,
            winner=None,
            move_log=[]
        )
    
    def make_move(self, row: int, col: int, player: Player) -> bool:
        """
        Make a move on the board.
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            player: The player making the move
            
        Returns:
            True if move was valid and made, False otherwise
        """
        if (self.state.game_over or 
            row < 0 or row > 2 or col < 0 or col > 2 or
            self.state.board[row][col] != Player.EMPTY):
            return False
        
        self.state.board[row][col] = player
        self.state.move_count += 1
        self.state.move_log.append((row, col, player))
        
        # Check for win or draw
        if self._check_winner():
            self.state.game_over = True
            self.state.winner = player
        elif self.state.move_count == 9:
            self.state.game_over = True
            self.state.winner = None  # Draw
        else:
            # Switch to the other player
            self.state.current_player = Player.O if player == Player.X else Player.X
        
        return True
    
    def _check_winner(self) -> bool:
        """Check if the last player to move has won."""
        board = self.state.board
        # Get the last player who moved
        if not self.state.move_log:
            return False
        player = self.state.move_log[-1][2]
        
        # Check rows
        for row in board:
            if all(cell == player for cell in row):
                return True
        
        # Check columns
        for col in range(3):
            if all(board[row][col] == player for row in range(3)):
                return True
        
        # Check diagonals
        if (board[0][0] == player and board[1][1] == player and board[2][2] == player):
            return True
        if (board[0][2] == player and board[1][1] == player and board[2][0] == player):
            return True
        
        return False
    
    def get_available_moves(self) -> List[Tuple[int, int]]:
        """Get list of available moves."""
        moves = []
        for row in range(3):
            for col in range(3):
                if self.state.board[row][col] == Player.EMPTY:
                    moves.append((row, col))
        return moves
    
    def board_to_string(self) -> str:
        """Convert board to string representation."""
        lines = []
        for row in self.state.board:
            line = " | ".join(cell.value for cell in row)
            lines.append(line)
        return "\n".join(lines)


class TicTacToeAgent:
    """Tic-Tac-Toe Agent using OpenAI Agents SDK."""
    
    def __init__(self, player: Player):
        """
        Initialize the agent.
        
        Args:
            player: The player this agent represents (X or O)
        """
        self.player = player
        self.agent = Agent(
            model="gpt-4o-mini",
            name=f"TicTacToe-{player.value}",
            instructions=f"""You are a Tic-Tac-Toe player playing as {player.value}. 
            Follow this strategy in order of priority:
            
            PRIORITY 1 - WIN IMMEDIATELY: If you can win in this move, do it!
            PRIORITY 2 - BLOCK OPPONENT: If opponent can win next turn, block them!
            
            CRITICAL RULES:
            - ALWAYS check if you can win first
            - ALWAYS check if opponent can win next turn and block them
            - Look for 2-in-a-row patterns (horizontal, vertical, diagonal)
            - If opponent has 2 in a row, you MUST block the third position
            - For other moves, be creative and strategic!
            
            RESPONSE FORMAT: Respond with ONLY coordinates like (1,1) or (0,0)
            
            Examples:
            - If you can win: Take the winning position
            - If opponent can win: Block their winning position  
            - Otherwise: Choose any strategic position you think is best"""
        )
    
    def get_move(self, game: TicTacToeGame) -> Tuple[int, int]:
        """
        Get the best move for the current game state using OpenAI Agents SDK.
        
        Args:
            game: Current game state
            
        Returns:
            Tuple of (row, col) for the best move
        """
        available_moves = game.get_available_moves()
        if not available_moves:
            return (0, 0)  # Fallback
        
        # Check if this is the first move of the game (move count = 0)
        if game.state.move_count == 0:
            # For the first move, use random position for more interesting games
            return self._get_random_first_move(available_moves)
        
        # Create prompt for the agent
        prompt = self._create_prompt(game, available_moves)
        
        try:
            # Run the agent in a new event loop to avoid asyncio issues
            def run_agent():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(Runner.run(self.agent, prompt))
                    return result
                finally:
                    loop.close()
            
            result = run_agent()
            move_text = result.final_output.strip()
            return self._parse_move(move_text, available_moves)
            
        except Exception as e:
            print(f"Error getting move from agent {self.player.value}: {e}")
            return available_moves[0]  # Fallback to first available move
    
    def _get_random_first_move(self, available_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Get a random first move to add variety to games.
        
        Args:
            available_moves: List of available moves
            
        Returns:
            Random move from available moves
        """
        if not available_moves:
            return (0, 0)
        
        # Prefer center and corners for first move, but add randomness
        center = (1, 1)
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        
        # 40% chance for center, 40% chance for corners, 20% chance for any position
        rand = random.random()
        
        if rand < 0.4 and center in available_moves:
            return center
        elif rand < 0.8:
            # Try corners
            available_corners = [corner for corner in corners if corner in available_moves]
            if available_corners:
                return random.choice(available_corners)
        
        # Fallback to any available move
        return random.choice(available_moves)
    
    def _create_prompt(self, game: TicTacToeGame, available_moves: List[Tuple[int, int]]) -> str:
        """Create prompt for the agent."""
        board_str = game.board_to_string()
        moves_str = ", ".join([f"({row},{col})" for row, col in available_moves])
        
        # Analyze the board for strategic information
        opponent = "O" if self.player == Player.X else "X"
        board = game.state.board
        
        # Check for immediate win opportunities
        win_opportunities = []
        block_opportunities = []
        
        # Check rows, columns, and diagonals for 2-in-a-row patterns
        for row in range(3):
            for col in range(3):
                if board[row][col] == Player.EMPTY:
                    # Check if this move would complete a win for current player
                    if self._would_complete_line(board, row, col, self.player):
                        win_opportunities.append(f"({row},{col})")
                    # Check if this move would block opponent
                    elif self._would_complete_line(board, row, col, Player.X if self.player == Player.O else Player.O):
                        block_opportunities.append(f"({row},{col})")
        
        # Check if this is the first move
        is_first_move = game.state.move_count == 0
        move_info = "FIRST MOVE - Choose any position!" if is_first_move else f"Move #{game.state.move_count + 1}"
        
        return f"""Current Tic-Tac-Toe board:
{board_str}

You are playing as {self.player.value}. Your opponent is {opponent}.
{move_info}
Available moves: {moves_str}

STRATEGIC ANALYSIS:
- Your symbol: {self.player.value}
- Opponent symbol: {opponent}
- Win opportunities: {', '.join(win_opportunities) if win_opportunities else 'None'}
- Block opportunities: {', '.join(block_opportunities) if block_opportunities else 'None'}

FOLLOW THE PRIORITY ORDER:
1. WIN: {win_opportunities[0] if win_opportunities else 'No immediate win'}
2. BLOCK: {block_opportunities[0] if block_opportunities else 'No blocking needed'}
3. STRATEGIC: Choose the best position for your strategy

Respond with ONLY the coordinates of your move, e.g., (1,1)"""
    
    def _would_complete_line(self, board: List[List[Player]], row: int, col: int, player: Player) -> bool:
        """Check if placing a piece at (row, col) would complete a line for the given player."""
        # Temporarily place the piece
        original = board[row][col]
        board[row][col] = player
        
        # Check if this creates a winning line
        is_win = self._check_line_completion(board, row, col, player)
        
        # Restore original state
        board[row][col] = original
        
        return is_win
    
    def _check_line_completion(self, board: List[List[Player]], row: int, col: int, player: Player) -> bool:
        """Check if the piece at (row, col) completes a line for the given player."""
        # Check row
        if all(board[row][c] == player for c in range(3)):
            return True
        
        # Check column
        if all(board[r][col] == player for r in range(3)):
            return True
        
        # Check main diagonal
        if row == col and all(board[i][i] == player for i in range(3)):
            return True
        
        # Check anti-diagonal
        if row + col == 2 and all(board[i][2-i] == player for i in range(3)):
            return True
        
        return False
    
    def _parse_move(self, move_text: str, available_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Parse the agent's move response."""
        try:
            # Extract coordinates from response
            import re
            match = re.search(r'\((\d+),(\d+)\)', move_text)
            if match:
                row, col = int(match.group(1)), int(match.group(2))
                if (row, col) in available_moves:
                    return (row, col)
        except:
            pass
        
        # Fallback to first available move
        return available_moves[0]


class TicTacToeApp:
    """Main application class managing the Gradio interface and game flow."""
    
    def __init__(self):
        """Initialize the application."""
        self.game = TicTacToeGame()
        self.agent_x = None
        self.agent_o = None
        self.game_history = []
    
    def setup_agents(self) -> bool:
        """
        Setup OpenAI agents using the Agents SDK.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            self.agent_x = TicTacToeAgent(Player.X)
            self.agent_o = TicTacToeAgent(Player.O)
            return True
        except Exception as e:
            print(f"Error setting up agents: {e}")
            return False
    
    def play_automatic_game(self, api_key: str):
        """
        Play a complete automatic game with real-time updates.
        
        Args:
            api_key: OpenAI API key (used to set environment variable)
            
        Yields:
            Tuple of (board_display, move_log, result_message) after each move
        """
        # Set the API key for the Agents SDK
        if api_key.strip():
            os.environ["OPENAI_API_KEY"] = api_key.strip()
        
        # Setup agents if not already done
        if not self.agent_x or not self.agent_o:
            if not self.setup_agents():
                yield "Error: Failed to setup agents", "", "Failed to setup agents"
                return
        
        # Reset game if needed
        if self.game.state.game_over:
            self.game.reset_game()
        
        move_log = []
        
        # Play the complete game automatically with real-time updates
        while not self.game.state.game_over:
            current_player = self.game.state.current_player
            current_agent = self.agent_x if current_player == Player.X else self.agent_o
            
            # Get move from current agent
            row, col = current_agent.get_move(self.game)
            
            # Make the move
            if self.game.make_move(row, col, current_player):
                move_log.append(f"Move {self.game.state.move_count}: {current_player.value} plays at ({row},{col})")
                
                # Yield the current state after each move
                board_display = self._format_board_display()
                move_log_text = "\n".join(move_log)
                result_message = self._get_result_message()
                
                yield board_display, move_log_text, result_message
                
                # Small delay to make the progression visible
                time.sleep(2)
            else:
                move_log.append(f"Error: Invalid move attempted by {current_player.value}")
                break
        
        # Final state after game completion
        board_display = self._format_board_display()
        move_log_text = "\n".join(move_log)
        result_message = self._get_result_message()
        
        yield board_display, move_log_text, result_message
    
    def play_automatic_game_with_progress(self, api_key: str, progress_callback=None):
        """
        Play a complete automatic game with progress updates.
        
        Args:
            api_key: OpenAI API key (used to set environment variable)
            progress_callback: Function to call with progress updates
            
        Returns:
            Final game result
        """
        # Set the API key for the Agents SDK
        if api_key.strip():
            os.environ["OPENAI_API_KEY"] = api_key.strip()
        
        # Setup agents if not already done
        if not self.agent_x or not self.agent_o:
            if not self.setup_agents():
                return "Error: Failed to setup agents", "", "Failed to setup agents"
        
        # Reset game if needed
        if self.game.state.game_over:
            self.game.reset_game()
        
        move_log = []
        
        # Play the complete game automatically
        while not self.game.state.game_over:
            current_player = self.game.state.current_player
            current_agent = self.agent_x if current_player == Player.X else self.agent_o
            
            # Get move from current agent
            row, col = current_agent.get_move(self.game)
            
            # Make the move
            if self.game.make_move(row, col, current_player):
                move_log.append(f"Move {self.game.state.move_count}: {current_player.value} plays at ({row},{col})")
                
                # Call progress callback if provided
                if progress_callback:
                    board_display = self._format_board_display()
                    move_log_text = "\n".join(move_log)
                    status = f"Move {self.game.state.move_count}: {current_player.value} plays at ({row},{col})"
                    progress_callback(board_display, move_log_text, status)
                    
                    # Small delay to make the progression visible
                    time.sleep(1)
            else:
                move_log.append(f"Error: Invalid move attempted by {current_player.value}")
                break
        
        # Create final displays
        board_display = self._format_board_display()
        move_log_text = "\n".join(move_log)
        result_message = self._get_result_message()
        
        return board_display, move_log_text, result_message
    
    def _format_move_log(self) -> str:
        """Format the move log for display."""
        if not self.game.state.move_log:
            return "No moves yet"
        
        log_entries = []
        for i, (row, col, player) in enumerate(self.game.state.move_log, 1):
            log_entries.append(f"Move {i}: {player.value} plays at ({row},{col})")
        
        return "\n".join(log_entries)
    
    def _get_result_message(self) -> str:
        """Get the current result message."""
        if self.game.state.game_over:
            if self.game.state.winner:
                return f"🎉 {self.game.state.winner.value} wins!"
            else:
                return "🤝 It's a draw!"
        else:
            current_player = self.game.state.current_player
            return f"Next move: {current_player.value}"
    
    def _format_board_display(self) -> str:
        """Format the board for display."""
        board = self.game.state.board
        lines = []
        for i, row in enumerate(board):
            line = " | ".join(cell.value for cell in row)
            lines.append(f"Row {i}: {line}")
        return "\n".join(lines)
    
    def new_game(self) -> Tuple[str, str, str]:
        """Start a new game."""
        self.game.reset_game()
        return "New game started! Enter your API key and click 'Play Game' to watch the AI battle", "", "Ready to play!"


def create_interface() -> gr.Interface:
    """Create the Gradio interface."""
    app = TicTacToeApp()
    
    def play_game_wrapper(api_key: str):
        """Wrapper function for playing the complete automatic game with real-time updates."""
        if not api_key.strip():
            yield "Please enter your OpenAI API key", "", "API key required"
            return
        
        # Play the game with real-time updates
        for board_display, move_log, result_message in app.play_automatic_game(api_key):
            yield board_display, move_log, result_message
    
    def new_game_wrapper() -> Tuple[str, str, str]:
        """Wrapper function for starting a new game."""
        return app.new_game()
    
    with gr.Blocks(title="AI Tic-Tac-Toe Battle", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 AI Tic-Tac-Toe Battle")
        gr.Markdown("Watch two GPT-4o-mini agents compete in Tic-Tac-Toe! Click 'Play Game' to see the match unfold move by move.")
        
        with gr.Row():
            with gr.Column():
                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="Enter your OpenAI API key",
                    type="password"
                )
                
                play_game_button = gr.Button("🎮 Play Game", variant="primary", size="lg")
                new_game_button = gr.Button("🔄 New Game", variant="secondary")
        
        with gr.Row():
            with gr.Column():
                board_display = gr.Textbox(
                    label="Game Board",
                    lines=5,
                    interactive=False
                )
            
            with gr.Column():
                move_log = gr.Textbox(
                    label="Move Log",
                    lines=10,
                    interactive=False
                )
        
        result_display = gr.Textbox(
            label="Game Status",
            lines=2,
            interactive=False
        )
        
        # Event handlers
        play_game_button.click(
            fn=play_game_wrapper,
            inputs=[api_key_input],
            outputs=[board_display, move_log, result_display]
        )
        
        new_game_button.click(
            fn=new_game_wrapper,
            inputs=[],
            outputs=[board_display, move_log, result_display]
        )
    
    return interface


def main():
    """Main function to run the application."""
    # Check for API key in environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    interface = create_interface()
    
    print("🚀 Starting AI Tic-Tac-Toe Battle...")
    print("📝 Make sure to set your OPENAI_API_KEY environment variable or enter it in the interface")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
