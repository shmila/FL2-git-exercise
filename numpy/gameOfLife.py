import time
from pprint import pprint

from pylab import *


def array_to_list(arr):
    if isinstance(arr, type(array([]))):
        return array_to_list(arr.tolist())
    elif isinstance(arr, type([])):
        return [array_to_list(a) for a in arr]
    else:
        return arr


def pretty_array(grid):
    rows = grid.shape[0]
    cols = grid.shape[1]
    to_print = array_to_list(grid)
    for i in range(rows):
        for j in range(cols):
            if (grid[i][j] == 0.0):
                to_print[i][j] = ' '
            else:
                to_print[i][j] = 'X'
    pprint(to_print)


def count_alive_neighbors(grid, row, col):
    rows = grid.shape[0]
    cols = grid.shape[1]
    count = 0
    for i in range(row-1, row+2):
        for j in range(col - 1, col + 2):
            if (i == row and j == col) or i < 0 or j < 0 or i >= rows or j >= cols:
                continue
            else:
                count += grid[i][j]
    return count


def process_entry(grid, i, j):
    num_of_alive_neighbors = count_alive_neighbors(grid, i, j)
    if grid[i][j] == 1 and (num_of_alive_neighbors == 2 or num_of_alive_neighbors == 3):
        return 1
    elif grid[i][j] == 0 and num_of_alive_neighbors == 3:
        return 1
    else:
        return 0


def game_of_life(grid):
    rows = grid.shape[0]
    cols = grid.shape[1]
    new_grid = np.zeros_like(grid)
    for i in range(rows):
        for j in range(cols):
            new_grid[i][j] = process_entry(grid, i, j)
    return new_grid


def init_grid(rows, cols):
    grid = np.zeros([rows, cols])
    for i in range(rows):
        for j in range(cols):
            grid[i][j] = int(np.random.randint(2, size=1)[0])
    return grid


def print_grid(new_grid):
    rows = new_grid.shape[0]
    cols = new_grid.shape[1]
    pretty_array(new_grid)


def game_of_life_aux(grid):
    print "grid is: "
    plt.plot(grid)
    while True:
        new_grid = game_of_life(grid)
        grid = new_grid
        print "grid is: "
        print_grid(new_grid)
        time.sleep(1)


def main():
    rows = raw_input('please insert the rows dimension: ')
    while(int(rows) <= 0):
        print "illegal input!"
        rows = raw_input('please insert the rows dimension: ')
    cols = raw_input('please insert the cols dimension: ')
    while int(cols) <= 0:
        print "illegal input!"
        cols = raw_input('please insert the cols dimension: ')
    grid = init_grid(int(rows), int(cols))
    game_of_life_aux(grid)

if __name__ == "__main__":
    main()
