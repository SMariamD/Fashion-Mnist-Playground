# Sudoku Logic Module
# This module contains the core logic for solving Sudoku puzzles

def is_valid(board, pos, num):
    """
    Check if placing 'num' at position 'pos' is valid according to Sudoku rules.
    
    Args:
        board: 9x9 Sudoku board (list of lists)
        pos: Position tuple (row, col) where we want to place the number
        num: Number (1-9) we want to place
    
    Returns:
        bool: True if the number can be placed, False otherwise
    """
    # Check if the position is already occupied (not 0)
    if board[pos[0]][pos[1]] != 0:
        return False  # Invalid: position is already occupied
    
    # Check if the number already exists in the same row
    # Loop through all columns in the current row
    for i in range(9):
        # If we find the same number in the row and it's not the current position
        if board[pos[0]][i] == num and pos[1] != i:
            return False  # Invalid: number already exists in this row
    
    # Check if the number already exists in the same column
    # Loop through all rows in the current column
    for i in range(9):
        # If we find the same number in the column and it's not the current position
        if board[i][pos[1]] == num and pos[0] != i:
            return False  # Invalid: number already exists in this column
    
    # Check if the number already exists in the same 3x3 box
    # Calculate which 3x3 box the position belongs to
    box_x = pos[1] // 3  # Box column (0, 1, or 2)
    box_y = pos[0] // 3  # Box row (0, 1, or 2)
    
    # Loop through all positions in the 3x3 box
    for i in range(box_y * 3, box_y * 3 + 3):      # Rows in the box
        for j in range(box_x * 3, box_x * 3 + 3):  # Columns in the box
            # If we find the same number in the box and it's not the current position
            if board[i][j] == num and (i, j) != pos:
                return False  # Invalid: number already exists in this 3x3 box
    
    # If we reach here, the number is valid to place at this position
    return True

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
