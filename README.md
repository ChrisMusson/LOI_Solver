# League of Ireland Fantasy Solver

A modifed version of the FPL Solver found here: https://github.com/sertalpbilal/FPL-Optimization-Tools/tree/main

# Instructions
- `git clone https://github.com/ChrisMusson/LOI_Solver.git LOI & cd LOI`
- `pip install -r requirements.txt`
- replace the file `xpts.csv` with an updated xpts CSV
- edit settings in `settings.json`, most importantly the path to your `highs.exe` executable. You will then also need to give the solver a way of finding your current squad and the gameweek to solve from. You can do this in two ways:
  1. Save the login cookie value to a .env file - log in to [the website](https://fantasyloi.leagueofireland.ie/) and press F12 to open developer tools. Navigate to `Application -> Cookies` if using chrome, or `Storage -> Cookies` if using firefox. Copy the value of the `LOIFF` cookie to a new file named `.env`, whose content is a single line `LOIFF=CfDJ...5XA`, where `CfDJ...5XA` is the copied value of your cookie
  1. Supply your current_squad and next_gw manually - add `"next_gw": gw,` and `"current_squad": [A, B, ..., N, O],` to `settings.json`, where `gw` is the gameweek you want to solve from and `[A, B, ..., N, O]` is a 15-integer-long list of the IDs of the 15 players in your current squad. These IDs must match up with the IDs in your `xpts.csv` file
> **_NOTE:_**  If you are using Vamps' xpts model, then you will need to use option 2 here as the IDs from that model and the fantasy LOI website are not the same. You can use either method with models that use the same IDs as the fantasy LOI website, e.g. the MooSwish model

- `python solve.py`

If you have the cookie saved to a `.env` file, you can then solve for any given team_id you like by adding the `"team_id": x,` setting to `settings.json`. This cookie value will have to be updated once every two weeks

# Options

Mostly the same as the FPL solver. To clarify the differences:

- `locked`: can specify a player ID to ban them for the entire horizon, or a `[player_id, gw]` pair to ban them for a specific gameweek, e.g. `[683665, [225100, 1]]` to ban ID 683665 for the entire horizon and ID 225100 for GW1
- `banned`: same as locked
- `current_squad`: a 15-integer-long list of the IDs of the 15 players in your current squad. These IDs must match up with the IDs in your `xpts.csv` file. This must be used when you haven't saved the cookie value to a .env file. Can be removed from `settings.json` or left as `[]` if you have saved the `LOIFF` cookie to `.env`. In this case, it will automatically fetch your squad of 15 players from the previous gameweek
- `next_gw`: the gameweek to solve from. This must be used when you haven't saved the cookie value to a .env file. Can be removed from `settings.json` or left as `null` if you have saved the `LOIFF` cookie to `.env`. In this case, it will default to the next available gameweek
- `formation`: the formation of the lineup. Can either be a string like "442" to lock 442 for every week in the horizon, or a list of lists like [["442", 2], ["541", 3]] to lock 442 in gw2 and 541 in gw3, for example

# Sensitivity Analysis

The aim of the sensitivity analysis is to randomise the xpts for all players based on their xmins and then solve with these new xpts. This is repeated many times and then you can find the transfers that come out as optimal most often. To do this,
-  ensure that the `results/` folder is empty of any solve solutions
- run `python simulations.py --no 50 --parallel 6` where 50 is the total number of simulations you want to run, and 6 is the number of solves to run in parallel. If not supplying a value for --parallel, it will use a value based on the number of CPU processes available
- when this has finished, run `python sensitivity.py` and answer the questions, or run e.g. `python sensitivity.py --all_gws n --gw 5 --wildcard n`. You can run `python sensitivity.py --help` for help with the optional command line arguments.