# AI Tic-Tac-Toe Battle

A Python application using Gradio and OpenAI Agents SDK where two GPT-4o-mini agents compete in Tic-Tac-Toe with creative strategies and random first moves for variety.

## 🚀 Live Demo

Try the application online: **[AI Tic-Tac-Toe Battle on Hugging Face Spaces](https://huggingface.co/spaces/devsu/agentic-tic-tac-toe)**

*Note: You'll need to provide your OpenAI API key to run the demo.*

## Features

- 🤖 Two GPT-4o-mini agents competing in Tic-Tac-Toe
- 🎮 Interactive Gradio web interface with real-time updates
- 📊 Move-by-move board display and logging
- 🏆 Automatic winner detection and result announcement
- 🔄 New Game functionality to restart matches
- 🎲 Random first move for game variety
- 🧠 Creative strategy implementation (win/block + creative positioning)

## Requirements

- Python 3.8+
- OpenAI API key
- Required packages: `gradio`, `openai-agents`

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Open your browser and navigate to `http://localhost:7860`

4. Enter your OpenAI API key in the interface (if not set as environment variable)

5. Click "Play Game" to watch the AI agents compete move by move in real-time!

## How It Works

- **Agent X** and **Agent O** are both powered by GPT-4o-mini using the **OpenAI Agents SDK**
- Each agent is created using the `Agent` class with creative strategy instructions
- The `Runner.run()` method orchestrates the agent decision-making process with async support
- **Random First Move**: Agent X starts with a random position (center 40%, corners 40%, any 20%)
- **Creative Strategy**: Agents focus on winning and blocking, but use creative positioning
- **Real-time Updates**: The interface updates move by move as the game progresses
- **Strategic Analysis**: Each move includes win/block opportunity detection
- The game continues until there's a winner or a draw
- All moves are logged and displayed in real-time with 2-second delays between moves

## Game Rules

- Standard 3x3 Tic-Tac-Toe rules
- X always goes first with a **random starting position**
- Agents alternate turns automatically with real-time updates
- Winner is determined by three in a row (horizontal, vertical, or diagonal)
- Game ends in a draw if all 9 squares are filled with no winner

## Strategy Features

- **Priority 1**: Win immediately if possible
- **Priority 2**: Block opponent from winning
- **Priority 3**: Creative strategic positioning
- **Random First Move**: Adds variety and unpredictability to games
- **Real-time Analysis**: Each move shows win/block opportunities

Enjoy watching the creative AI battle! 🎯🎲