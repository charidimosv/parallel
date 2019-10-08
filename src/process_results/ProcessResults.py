import os
import re
import pandas as pd

path = os.getcwd() + '\\results'

cols = ['implementation', 'steps', 'x', 'y', 'tasks', 'nodes', 'ppn', 'threads']
allCols = ['implementation', 'steps', 'x', 'y', 'tasks', 'nodes', 'ppn', 'threads', 'runtime']
lst = []

for r, d, f in os.walk(path):
    for filename in f:
        implementation = 0
        tasks = 0
        nodes = 0
        ppn = 0
        threads = 0
        x = 0
        y = 0
        steps = 0

        search = re.search(r"^\b[a-zA-Z]+", filename)
        if search is not None:
            implementation = search.group()

        search = re.search(r"Ts(\d+)_", filename)
        if search is not None:
            tasks = int(search.group(1))

        search = re.search(r"N(\d+)_", filename)
        if search is not None:
            nodes = int(search.group(1))

        search = re.search(r"P(\d+)_", filename)
        if search is not None:
            ppn = int(search.group(1))

        search = re.search(r"T(\d+)_", filename)
        if search is not None:
            threads = int(search.group(1))

        search = re.search(r"X(\d+)_", filename)
        if search is not None:
            x = int(search.group(1))

        search = re.search(r"Y(\d+)_", filename)
        if search is not None:
            y = int(search.group(1))

        search = re.search(r"S_(\d+)", filename)
        if search is not None:
            steps = int(search.group(1))

        if implementation != 0 and tasks != 0 and nodes != 0 and ppn != 0 and threads != 0 and x != 0 and y != 0 and steps != 0:
            file = open(path + '\\' + filename, "r")
            for line in file:
                search = re.search(r"(\d*\.*\d*) sec", line)
                if search is not None:
                    runtime = float(search.group(1))
                    lst.append([implementation, steps, x, y, tasks, nodes, ppn, threads, runtime])
                    break

df = pd.DataFrame(lst, columns=allCols)
df = df.pivot_table('runtime', cols, aggfunc='mean').reset_index()
df.sort_values(cols, inplace=True)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)
