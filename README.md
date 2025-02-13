# League of Ireland Fantasy Solver

A modifed version of the FPL Solver found here: https://github.com/sertalpbilal/FPL-Optimization-Tools/tree/main

# Instructions
- `git clone https://github.com/ChrisMusson/LOI_Solver.git LOI & cd LOI`
- edit settings in `settings.json`, most importantly the path to your `highs.exe` executable
- replace the file `xpts.csv` with an updated xpts CSV
- `python solve.py`


# Options

Mostly the same as the FPL solver. To clarify the differences:

- `locked`: can specify a player ID to ban them for the entire horizon, or a `[player_id, gw]` pair to ban them for a specific gameweek, e.g. `[683665, [225100, 1]]` to ban ID 683665 for the entire horizon and ID 225100 for GW1
- `banned`: same as locked

# Sensitivity Analysis

The aim of the sensitivity analysis is to randomise the xpts for all players based on their xmins and then solve with these new xpts. This is repeated many times and then you can find the transfers that come out as optimal most often. To do this,
-  ensure that the `results/` folder is empty of any solve solutions
- run `python simulations.py --no 50 --parallel 6` where 50 is the total number of simulations you want to run, and 6 is the number of solves to run in parallel. If not supplying a value for --parallel, it will use a value based on the number of CPU processes
- when this has finished, run `python sensitivity.py` and answer the questions, or run e.g. `python sensitivity.py --all_gws n --gw 5 --wildcard n. You can run `python sensitivity.py --help` for help with the optional command line arguments.