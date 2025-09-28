# Sudoku Logic Module
# This module contains the core logic for solving Sudoku puzzles

def is_valid(board, row, col, num):
    """
    Check if placing 'num' at position (row, col) is valid
    """
    pass

def solve_sudoku(board):
    """
    Solve the Sudoku puzzle using backtracking algorithm
    """
    pass

def print_board(board):
    """
    Print the Sudoku board in a readable format with visual separators.
    This function displays the 9x9 Sudoku grid with horizontal and vertical lines
    to separate the 3x3 boxes, making it easy to read.
    """
    # Loop through each row of the 9x9 Sudoku board
    for i in range(9):
        # Add horizontal separator line after every 3 rows (but not before the first row)
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - -")
        
        # Loop through each column in the current row
        for j in range(9):
            # Add vertical separator after every 3 columns (but not before the first column)
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            
            # Print the number at position (i,j)
            if j == 8:
                # If this is the last column in the row, print the number and move to next line
                print(board[i][j])
            else:
                # If this is not the last column, print the number followed by a space
                print(str(board[i][j]) + " ", end="")

def load_puzzle(filename):
    """
    Load a Sudoku puzzle from a text file
    """
    pass

def save_solution(board, filename):
    """
    Save the solved puzzle to a text file
    """
    pass
