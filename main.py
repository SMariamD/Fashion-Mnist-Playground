# Main Sudoku Solver Application
# This is the main entry point for the Sudoku solver

import sys
from sudoku_logic import solve_sudoku, load_puzzle, print_board
from sudoku_gui import SudokuGUI

def main():
    """
    Main function to run the Sudoku solver
    """
    print("Sudoku Solver")
    print("1. Solve puzzle from file")
    print("2. Interactive GUI mode")
    print("3. Exit")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        # Solve puzzle from file
        filename = input("Enter puzzle filename: ")
        try:
            board = load_puzzle(filename)
            print("Original puzzle:")
            print_board(board)
            
            if solve_sudoku(board):
                print("\nSolved puzzle:")
                print_board(board)
            else:
                print("No solution found!")
        except FileNotFoundError:
            print(f"File {filename} not found!")
    
    elif choice == "2":
        # Interactive GUI mode
        gui = SudokuGUI()
        gui.run()
    
    elif choice == "3":
        print("Goodbye!")
        sys.exit(0)
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
