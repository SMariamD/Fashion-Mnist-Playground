# Test script for is_valid function
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sudoku_logic import is_valid, print_board

# Create a test board with some numbers already placed
test_board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

print("Testing is_valid function:")
print("=" * 50)
print("Test board:")
print_board(test_board)
print()

# Test cases
test_cases = [
    # (position, number, expected_result, description)
    ((0, 2), 4, True, "Valid: 4 can be placed at (0,2) - no conflicts"),
    ((0, 2), 5, False, "Invalid: 5 already exists in row 0"),
    ((0, 2), 6, False, "Invalid: 6 already exists in column 2"),
    ((0, 2), 1, True, "Valid: 1 can be placed at (0,2) - no conflicts in 3x3 box"),
    ((1, 1), 2, True, "Valid: 2 can be placed at (1,1) - no conflicts"),
    ((1, 1), 1, False, "Invalid: 1 already exists in column 1"),
    ((1, 1), 6, False, "Invalid: 6 already exists in row 1"),
    ((4, 4), 5, True, "Valid: 5 can be placed at (4,4) - no conflicts"),
    ((4, 4), 6, False, "Invalid: 6 already exists in row 4"),
    ((4, 4), 3, False, "Invalid: 3 already exists in column 4"),
    ((4, 4), 8, False, "Invalid: 8 already exists in middle 3x3 box"),
    ((8, 8), 1, False, "Invalid: 1 cannot be placed at (8,8) - position already has 9"),
    ((8, 8), 9, False, "Invalid: 9 already exists at position (8,8)"),
    ((8, 8), 5, False, "Invalid: 5 already exists in column 8"),
    ((8, 8), 7, False, "Invalid: 7 already exists in bottom-right 3x3 box"),
]

print("Running test cases:")
print("-" * 50)

passed = 0
failed = 0

for pos, num, expected, description in test_cases:
    result = is_valid(test_board, pos, num)
    status = "✓ PASS" if result == expected else "✗ FAIL"
    
    if result == expected:
        passed += 1
    else:
        failed += 1
    
    print(f"{status} | {description}")
    print(f"      Position: {pos}, Number: {num}, Expected: {expected}, Got: {result}")
    print()

print("=" * 50)
print(f"Test Results: {passed} passed, {failed} failed")
print(f"Success Rate: {(passed/(passed+failed)*100):.1f}%")
