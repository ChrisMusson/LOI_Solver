import argparse
import csv
import datetime
import json
import os
import random
import string
import subprocess
import threading
import time

import highspy
import numpy as np
import pandas as pd
import requests
import sasoptpy as so
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from visualisation import create_squad_timeline

BASE_URL = "https://fantasyloi.leagueofireland.ie"


def get_live_data(options):
    load_dotenv()
    cookies = {"LOIFF": os.getenv("LOIFF")}
    with requests.Session() as s:
        r = s.get(f"{BASE_URL}/Team/TeamSheet", cookies=cookies)
        soup = BeautifulSoup(r.text, "html.parser")
        next_gw = int([x for x in soup.find_all("strong") if "GW" in x.text][-1].text.split(" ")[1])

        team_id = options.get("team_id")
        if not team_id:
            player_cards = soup.find_all("div", {"data-target": "#viewPlayer"})
            squad = [int(player["onclick"].split(",")[0].replace("'", "").split("(")[1]) for player in player_cards]

        else:
            url = f"{BASE_URL}/Results?gameweek={next_gw - 1}&teamId={team_id}"
            r = s.get(url, cookies=cookies)
            soup = BeautifulSoup(r.text, "html.parser")
            player_cards = soup.find_all("div", {"data-target": "#playerScore"})
            squad = [int(player["onclick"].split(",")[0].replace("'", "").split("(")[1]) for player in player_cards]

    return next_gw, squad


def get_random_id(n):
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(n))


def solve(runtime_options=None):
    with open("settings.json", "r") as f:
        options = json.load(f)

    solve_name = "sens_" + options["run_no"] if "run_no" in options else f"reg_{get_random_id(5)}"
    parser = argparse.ArgumentParser(add_help=False)
    cl_args = parser.parse_known_args()[1]

    for key, value in options.items():
        if value is None or isinstance(value, (list, dict)):
            parser.add_argument(f"--{key}", default=value)
            continue
        parser.add_argument(f"--{key}", type=type(value), default=value)

    # Parse cl args, which will take priority over default args in options dict
    args = vars(parser.parse_args(cl_args))

    # this code block is to look at command line arguments (read as a string) and determine what type
    # they should be when there is no default argument type set by the code above
    for key, value in args.items():
        if key not in options:
            continue
        if value == options[key]:  # skip anything that hasn't been edited by command line argument
            continue

        if options[key] is None or isinstance(options[key], (list, dict)):
            if value.isdigit():
                args[key] = int(value)
                continue

            try:
                args[key] = float(value)
                continue
            except ValueError:
                pass

            if value[0] in "[{":
                try:
                    args[key] = json.loads(value)
                    continue
                except json.JSONDecodeError:
                    value = value.replace("'", '"')
                    args[key] = json.loads(value)
                    continue
                finally:
                    pass
            print(f"Problem with CL argument: {key}. Original value: {options[key]}, New value: {value}")

    options.update(args)

    # runtime options take absolute priority other other option input methods. It is used in run_parallel.py and simulations.py
    if runtime_options is not None:
        options.update(runtime_options)

    horizon = options.get("horizon", 5)
    preseason = options.get("preseason", False)

    if preseason:
        next_gw = 1
        current_squad = []
    else:
        next_gw = options.get("next_gw")
        current_squad = options.get("current_squad", [])

        if len(current_squad) not in [0, 15]:
            raise ValueError(f"The length of your current squad list must be either 0 or 15. It is currently {len(current_squad)}")

        if not next_gw or not current_squad:
            live_data = get_live_data(options)
            if not next_gw:
                next_gw = live_data[0]
            if not current_squad:
                current_squad = live_data[1]
            options["next_gw"] = next_gw
            options["current_squad"] = current_squad

    target_gw = options.get("target_gw", next_gw)
    gws = list(range(next_gw, next_gw + horizon))
    all_gws = [next_gw - 1] + gws

    df = pd.read_csv("xpts.csv", index_col="ID", encoding="cp850").fillna(0)
    vamps = any("_xMin" in x for x in df.columns)
    if options.get("randomised", False):
        rng = np.random.default_rng(seed=options.get("seed"))
        for w in gws:
            noise = df[f"{w}_Pts"] * (92 - df[f"{w}_xMin" if vamps else "xMin"]) / 134 * rng.standard_normal(size=len(df))
            df[f"{w}_Pts"] = df[f"{w}_Pts"] + noise

    players = df.index.to_list()
    el_types = df["Pos"].unique().tolist()
    teams = df["Team"].unique().tolist()
    player_values = (df["Value" if "Value" in df.columns else "BV"] * 10).to_dict()

    model = so.Model(name="LOI")

    squad = model.add_variables(players, all_gws, name="squad", vartype=so.binary)
    lineup = model.add_variables(players, all_gws, name="lineup", vartype=so.binary)
    bench = model.add_variables(players, all_gws, range(4), name="bench", vartype=so.binary)
    cap = model.add_variables(players, gws, name="cap", vartype=so.binary)
    itb = model.add_variables(all_gws, name="itb", vartype=so.integer, lb=0)
    t_in = model.add_variables(players, gws, name="t_in", vartype=so.binary)
    t_out = model.add_variables(players, gws, name="t_out", vartype=so.binary)
    use_wc = model.add_variables(gws, name="use_wc", vartype=so.binary)
    use_bb = model.add_variables(gws, name="use_bb", vartype=so.binary)
    use_tc = model.add_variables(players, gws, name="use_tc", vartype=so.binary)
    use_tc_gw = {w: so.expr_sum(use_tc[p, w] for p in players) for w in gws}

    model.add_constraint(so.expr_sum(use_wc[w] for w in gws) <= 1, name="max_one_wc")
    model.add_constraint(so.expr_sum(use_bb[w] for w in gws) <= 1, name="max_one_bb")
    model.add_constraints((so.expr_sum(squad[p, w] for p in players) == 15 for w in gws), name="15_man_squad")
    model.add_constraints((so.expr_sum(lineup[p, w] for p in players) == 11 + 4 * use_bb[w] for w in gws), name="lineup_size")
    model.add_constraints((so.expr_sum(cap[p, w] for p in players) == 1 for w in gws), name="one_cap_per_gw")
    model.add_constraints((cap[p, w] <= lineup[p, w] for p in players for w in gws), name="cap_in_lineup")

    model.add_constraints((lineup[p, w] <= squad[p, w] for p in players for w in gws), name="lineup_subest_of_squad")
    model.add_constraints(
        (so.expr_sum(bench[p, w, n] for n in range(4)) <= squad[p, w] for p in players for w in gws),
        name="bench_subset_of_squad",
    )
    model.add_constraints(
        (so.expr_sum(bench[p, w, n] for n in range(4)) + lineup[p, w] <= 1 for p in players for w in gws),
        name="bench_not_in_lineup",
    )

    if preseason:
        model.add_constraint(so.expr_sum(squad[p, 0] for p in players) == 0, name="empty_preseason_squad")
        model.add_constraint(itb[0] == 1000, name="initial_itb")
    else:
        model.add_constraints((squad[p, next_gw - 1] == 1 for p in current_squad), name="initial_squad_players")
        model.add_constraints((squad[p, next_gw - 1] == 0 for p in players if p not in current_squad), name="initial_squad_others")
        model.add_constraint(itb[next_gw - 1] == 1000 - so.expr_sum(player_values[p] for p in current_squad), name="initial_itb")

    # Ensure correct amount of players in squad by position
    player_el_types = df["Pos"].to_dict()
    player_teams = df["Team"].to_dict()
    type_data = {
        "G": {"squad": 2, "min": 1, "max": 1},
        "D": {"squad": 5, "min": 3, "max": 5},
        "M": {"squad": 5, "min": 3, "max": 5},
        "F": {"squad": 3, "min": 1, "max": 3},
    }

    squad_type_count = {(t, w): so.expr_sum(squad[p, w] for p in players if player_el_types[p] == t) for t in el_types for w in gws}
    lineup_type_count = {(t, w): so.expr_sum(lineup[p, w] for p in players if player_el_types[p] == t) for t in el_types for w in gws}
    model.add_constraints(
        (squad_type_count[t, w] == type_data[t]["squad"] for t in el_types for w in gws),
        name="valid_squad_position",
    )
    model.add_constraints(
        (lineup_type_count[t, w] >= type_data[t]["min"] for t in el_types for w in gws),
        name="min_pos_in_lineup",
    )
    model.add_constraints(
        (lineup_type_count[t, w] <= type_data[t]["max"] + use_bb[w] for t in el_types for w in gws),
        name="max_pos_in_lineup",
    )
    model.add_constraints(
        (so.expr_sum(squad[p, w] for p in players if player_teams[p] == t) <= 3 for t in teams for w in gws),
        name="valid_squad_max_per_team",
    )

    model.add_constraints(
        (so.expr_sum(bench[p, w, n] for p in players) <= 1 for w in gws for n in range(4)),
        name="one_bench_slot_per_player",
    )

    model.add_constraints(
        (so.expr_sum(bench[p, w, n] for n in range(4)) <= 1 for w in gws for p in players),
        name="one_player_per_bench_slot",
    )

    # Transfer logic
    n_transfers = {w: so.expr_sum(t_in[p, w] for p in players) for w in gws}
    bought_amount = {w: so.expr_sum(player_values[p] * t_in[p, w] for p in players) for w in gws}
    sold_amount = {w: so.expr_sum(player_values[p] * t_out[p, w] for p in players) for w in gws}

    model.add_constraints(
        (t_in[p, w] + t_out[p, w] <= 1 for p in players for w in gws),
        name="tr_in_out_limit",
    )

    transfers_allowed = {w: (15 if w == 1 else 1 - use_wc[w]) * 3 + use_wc[w] * 15 for w in gws}
    model.add_constraints(
        (so.expr_sum(t_in[p, w] for p in players) <= transfers_allowed[w] for w in gws),
        name="max_transfers_per_gw",
    )

    model.add_constraints(
        (squad[p, w] == squad[p, w - 1] + t_in[p, w] - t_out[p, w] for p in players for w in gws),
        name="squad_t_rel",
    )

    model.add_constraints(
        (itb[w] == itb[w - 1] + sold_amount[w] - bought_amount[w] for w in gws),
        name="cont_budget",
    )

    total_transfers_allowed = options.get("total_transfers")
    if isinstance(total_transfers_allowed, int) and total_transfers_allowed > 0:
        model.add_constraint(so.expr_sum(n_transfers[w] for w in gws) <= total_transfers_allowed, name="total_transfers_allowed")

    # Chip constraints
    model.add_constraints((use_wc[w] + use_bb[w] + use_tc_gw[w] <= 1 for w in gws), name="single_chip")
    model.add_constraints((use_tc[p, w] <= cap[p, w] for p in players for w in gws), name="tc_same_as_cap")

    if options.get("use_tc"):
        model.add_constraint(so.expr_sum(use_tc[p, options["use_tc"]] for p in players) == 1, name="use_tc_in_gw")
        model.add_constraint(so.expr_sum(use_tc[p, w] for p in players for w in gws) == 1, name="only_one_tc")
    else:
        model.add_constraint(so.expr_sum(use_tc[p, w] for p in players for w in gws) == 0, name="no_tc_used")

    if options.get("use_wc"):
        gw = options["use_wc"]
        model.add_constraint(use_wc[gw] == 1, name="force_wc")
    else:
        model.add_constraint(so.expr_sum(use_wc[w] for w in gws) == 0, name="no_wc_used")

    if options.get("use_bb"):
        gw = options["use_bb"]
        model.add_constraint(use_bb[gw] == 1, name="force_bb")
    else:
        model.add_constraint(so.expr_sum(use_bb[w] for w in gws) == 0, name="no_bb_used")

    if options.get("locked"):
        print("OC - Locked")
        for x in options["locked"]:
            if isinstance(x, int):
                model.add_constraints((squad[x, w] == 1 for w in gws), name=f"locked_{x}")
            elif isinstance(x, list):
                model.add_constraint(squad[x[0], x[1]] == 1, name=f"locked_{x[0]}_{x[1]}")

    if options.get("banned"):
        print("OC - Banned")
        for x in options["banned"]:
            if isinstance(x, int):
                model.add_constraints((squad[x, w] == 0 for w in gws), name=f"banned_{x}")
            elif isinstance(x, list):
                model.add_constraint(squad[x[0], x[1]] == 0, name=f"banned_{x[0]}_{x[1]}")

    if options.get("formation"):
        print("OC - Formation")
        formation = options.get("formation")
        if isinstance(formation, str):
            # force given formation in every gameweek
            formation = [int(x) for x in "".join(formation.split("-"))]
            model.add_constraints(
                (
                    so.expr_sum(lineup[p, w] for p in players if player_el_types[p] == pos) == formation[i]
                    for i, pos in enumerate(el_types[1:])
                    if formation[i] > 0
                    for w in gws
                ),
                name="force_formation",
            )
        elif isinstance(formation, list):
            # will be a list of [formation, gameweek] pairs
            if [type(x) for x in formation] != [list] * len(formation):
                raise TypeError('Your list of formations is not of the form [["442", 2], ["541", 3]]')

            for f, w in formation:
                f = [int(x) for x in "".join(f.split("-"))]
                model.add_constraints(
                    (
                        so.expr_sum(lineup[p, w] for p in players if player_el_types[p] == pos) == f[i]
                        for i, pos in enumerate(el_types[1:])
                        if f[i] > 0
                    ),
                    name=f"force_formation_gw_{w}",
                )

        else:
            raise TypeError('Your formation setting must either be a string like "442" or a list of lists like [["442", 2], ["541", 3]]')

    if options.get("no_future_transfers"):
        model.add_constraints((n_transfers[w] == 0 for w in gws[1:]), name="no_future_transfers")

    for i, x in enumerate(options.get("position_constraints", [])):
        teams = x.get("teams", [])
        spec_gws = x.get("gws", [next_gw])
        spec_positions = x.get("pos", "GD")
        num = x.get("num", 1)
        typ = x.get("type", "min")
        sl = x.get("squad_lineup", "squad")

        print(x)

        sls = {"squad": squad, "lineup": lineup}
        if typ == "min":
            model.add_constraints(
                (
                    so.expr_sum(sls[sl][p, w] for p in players for team in teams if player_el_types[p] in spec_positions and player_teams[p] == team)
                    >= num
                    for w in spec_gws
                ),
                name=f"min_team_pos_constraint_{i}",
            )

        elif typ == "max":
            model.add_constraints(
                (
                    so.expr_sum(sls[sl][p, w] for p in players for team in teams if player_teams[p] == team and player_el_types[p] in spec_positions)
                    <= num
                    for w in spec_gws
                ),
                name=f"max_team_pos_constraint_{i}",
            )

    # Objective
    pts_player_week = {(p, w): df.loc[p, f"{w}_Pts"] for p in players for w in gws}
    lineup_pts = {w: so.expr_sum(pts_player_week[p, w] * (lineup[p, w] + cap[p, w] + use_tc[p, w]) for p in players) for w in gws}

    model.add_constraints((so.expr_sum(bench[p, w, n] for n in range(4) for p in players) == 4 - 4 * use_bb[w] for w in gws), name="4_subs")
    model.add_constraints((so.expr_sum(bench[p, w, 0] for p in players if player_el_types[p] == "G") == 1 - use_bb[w] for w in gws), name="GK_sub0")

    # workaround for now, setting the 0th sub (GK) and last sub to contribute sub_weight**2 to xp
    # TODO: add proper bench contributions
    sub_weight = options.get("sub_weight", 0.1)
    # bps = [0, 1, 2, 3]
    sub0 = {w: so.expr_sum(pts_player_week[p, w] * bench[p, w, 0] * sub_weight**2 for p in players) for w in gws}
    sub1 = {w: so.expr_sum(pts_player_week[p, w] * bench[p, w, 1] * sub_weight for p in players) for w in gws}
    sub2 = {w: so.expr_sum(pts_player_week[p, w] * bench[p, w, 2] * sub_weight for p in players) for w in gws}
    sub3 = {w: so.expr_sum(pts_player_week[p, w] * bench[p, w, 3] * sub_weight**2 for p in players) for w in gws}

    model.add_constraints(
        (so.expr_sum(bench[p, w, 1] + bench[p, w, 2] for p in players if player_el_types[p] == el_type) <= 1 for el_type in el_types for w in gws),
        name="sub1_sub2_diff_positions",
    )

    ft_penalty = options.get("ft_penalty", 0)
    ft_penalty_obj = {w: ft_penalty * n_transfers[w] for w in gws}

    # model.add_constraint(bench[17, 1, 0] == 1, name="sub0")
    # model.add_constraint(bench[3, 1, 1] == 1, name="sub1")
    # model.add_constraint(bench[4, 1, 2] == 1, name="sub2")
    # model.add_constraint(bench[6, 1, 3] == 1, name="sub3")

    bench_pts = {w: sub0[w] + sub1[w] + sub2[w] + sub3[w] for w in gws}
    gw_pts = {w: lineup_pts[w] + bench_pts[w] - ft_penalty_obj[w] for w in gws}

    decay = options.get("decay", 0.9)
    total_xp = so.expr_sum(gw_pts[w] * decay**i for i, w in enumerate(gws))
    model.set_objective(-total_xp, sense="N", name="total_regular_xp")

    solver = options.get("solver", "highs")
    use_cmd = options.get("use_cmd", False)
    if os.name != "nt" and not use_cmd:
        use_cmd = True
    solutions = []
    iterations = options.get("iterations", 1)
    for it in range(iterations):
        opt_file_name = os.path.join("tmp", f"{solve_name}.opt")
        sol_file_name = os.path.join("tmp", f"{solve_name}_sol.txt")
        mps_file_name = os.path.join("tmp", f"{solve_name}.mps")

        model.export_mps(mps_file_name)

        if solver == "highs":
            secs = options.get("secs", 20 * 60)
            presolve = options.get("presolve", "on")
            gap = options.get("gap", 0)
            random_seed = options.get("random_seed", 0)

            h = highspy.Highs()
            h.setOptionValue("random_seed", int(random_seed))
            h.setOptionValue("presolve", str(presolve))
            h.setOptionValue("time_limit", float(secs))
            h.setOptionValue("mip_rel_gap", float(gap))

            # Optional: mimic "--parallel on" (0 = automatic)
            h.setOptionValue("threads", int(options.get("threads", 0)))

            h.readModel(mps_file_name)
            h.run()

            # Write a HiGHS solution file that your existing parser reads ("# Columns" section)
            h.writeSolution(sol_file_name)

        elif solver == "copt":
            sol_file_name = sol_file_name.replace("_sol", "").replace("txt", "sol")
            gap = options.get("gap", 0)
            command = f'copt_cmd -c "set RelGap {gap};readmps {mps_file_name};optimize;writesol {sol_file_name};quit"'
            if use_cmd:
                os.system(command)
            else:

                def print_output(process):
                    while True:
                        output = process.stdout.readline()
                        if "Solving report" in output:
                            time.sleep(2)
                            process.kill()
                        elif output == "" and process.poll() is not None:
                            break
                        elif output:
                            print(output.strip())

                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                output_thread = threading.Thread(target=print_output, args=(process,))
                output_thread.start()
                output_thread.join()

            # Parsing
            with open(sol_file_name, "r") as f:
                for v in model.get_variables():
                    v.set_value(0)

                for line in f:
                    if line == "":
                        break
                    if line[0] == "#":
                        continue
                    words = line.split()
                    v = model.get_variable(words[0])
                    try:
                        if v.get_type() == so.INT:
                            v.set_value(round(float(words[1])))
                        elif v.get_type() == so.BIN:
                            v.set_value(round(float(words[1])))
                        elif v.get_type() == so.CONT:
                            v.set_value(round(float(words[1]), 3))
                    except Exception:
                        print("Error", words[0], line)

        elif solver == "gurobi":
            gap = options.get("gap", 0)
            sol_file_name = sol_file_name.replace("_sol", "").replace("txt", "sol")
            command = f"gurobi_cl MIPGap={gap} ResultFile={sol_file_name} {mps_file_name}"

            if use_cmd:
                os.system(command)
            else:

                def print_output(process):
                    while True:
                        output = process.stdout.readline()
                        if "Solving report" in output:
                            time.sleep(2)
                            process.kill()
                        elif output == "" and process.poll() is not None:
                            break
                        elif output:
                            print(output.strip())

                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                output_thread = threading.Thread(target=print_output, args=(process,))
                output_thread.start()
                output_thread.join()

            # Parsing
            with open(sol_file_name, "r") as f:
                for v in model.get_variables():
                    v.set_value(0)
                cols_started = False
                for line in f:
                    if line[0] == "#":
                        continue
                    if line == "":
                        break
                    words = line.split()
                    v = model.get_variable(words[0])
                    try:
                        if v.get_type() == so.INT:
                            v.set_value(round(float(words[1])))
                        elif v.get_type() == so.BIN:
                            v.set_value(round(float(words[1])))
                        elif v.get_type() == so.CONT:
                            v.set_value(round(float(words[1]), 3))
                    except Exception:
                        print("Error", words[0], line)

        # DataFrame generation
        picks = []
        for w in gws:
            for p in players:
                if squad[p, w].get_value() + t_out[p, w].get_value() > 0.5:
                    lp = df.loc[p]
                    is_cap = 1 if cap[p, w].get_value() > 0.5 else 0
                    is_squad = 1 if squad[p, w].get_value() > 0.5 else 0
                    is_lineup = 1 if lineup[p, w].get_value() > 0.5 else 0
                    is_tc = 1 if use_tc[p, w].get_value() > 0.5 else 0
                    is_t_in = 1 if t_in[p, w].get_value() > 0.5 else 0
                    is_t_out = 1 if t_out[p, w].get_value() > 0.5 else 0
                    position = player_el_types[p]
                    player_price = 0 if not is_t_in else player_values[p]
                    multiplier = 1 * (is_lineup == 1) + 1 * (is_cap == 1) + 1 * (is_tc == 1)
                    xp_cont = pts_player_week[p, w] * multiplier

                    bench_value = -1
                    for bp in [0, 1, 2, 3]:
                        if bench[p, w, bp].get_value() > 0.5:
                            bench_value = bp

                    # chip
                    if use_wc[w].get_value() > 0.5:
                        chip_text = "WC"
                    elif use_bb[w].get_value() > 0.5:
                        chip_text = "BB"
                    elif use_tc[p, w].get_value() > 0.5:
                        chip_text = "TC"
                    else:
                        chip_text = ""

                    picks.append(
                        {
                            "iter": it,
                            "id": p,
                            "week": w,
                            "name": lp["Name"],
                            "pos": position,
                            "type": lp["Pos"],
                            "team": lp["Team"],
                            "price": player_price,
                            "xP": round(pts_player_week[p, w], 2),
                            "squad": is_squad,
                            "lineup": is_lineup,
                            "bench": bench_value,
                            "cap": is_cap,
                            "t_in": is_t_in,
                            "t_out": is_t_out,
                            "multiplier": multiplier,
                            "xp_cont": xp_cont,
                            "chip": chip_text,
                            "t_count": n_transfers[w].get_value(),
                        }
                    )

        picks_df = pd.DataFrame(picks)
        picks_df["type"] = pd.Categorical(picks_df["type"], categories=["G", "D", "M", "F"], ordered=True)
        picks_df = picks_df.sort_values(by=["week", "lineup", "type", "xP"], ascending=[True, False, True, True])
        total_xp = so.expr_sum((lineup[p, w] + cap[p, w]) * pts_player_week[p, w] for p in players for w in gws).get_value()

        picks_df.sort_values(by=["week", "squad", "lineup", "bench", "type"], ascending=[True, False, False, True, True], inplace=True)

        # Writing summary
        summary_of_actions = ""
        move_summary = {"chip": [], "buy": [], "sell": []}
        cumulative_xpts = 0

        # collect statistics
        statistics = {next_gw - 1: {"itb": itb[next_gw - 1].get_value()}}
        for w in gws:
            summary_of_actions += f"** GW {w}:\n"
            chip_decision = (
                "WC" if use_wc[w].get_value() > 0.5 else "BB" if use_bb[w].get_value() > 0.5 else "TC" if use_tc_gw[w].get_value() > 0.5 else ""
            )
            if chip_decision != "":
                summary_of_actions += "CHIP " + chip_decision + "\n"
                move_summary["chip"].append(chip_decision + str(w))
            # summary_of_actions += f"ITB={itb[w - 1].get_value()}->{itb[w].get_value()}, NT={n_transfers[w].get_value()}\n"
            summary_of_actions += f"ITB={itb[w - 1].get_value() / 10}->{itb[w].get_value() / 10} {chip_decision}\n"
            for p in players:
                if t_in[p, w].get_value() > 0.5:
                    summary_of_actions += f"Buy {p} - {df['Name'][p]}\n"
                    if w == target_gw:
                        move_summary["buy"].append(df["Name"][p])

            for p in players:
                if t_out[p, w].get_value() > 0.5:
                    summary_of_actions += f"Sell {p} - {df['Name'][p]}\n"
                    if w == target_gw:
                        move_summary["sell"].append(df["Name"][p])

            gw_df = picks_df[picks_df["week"] == w]
            squad_players = gw_df.loc[gw_df["squad"] == 1]
            lineup_players = gw_df.loc[gw_df["lineup"] == 1]
            bench_players = picks_df[(picks_df["week"] == w) & (picks_df["bench"] >= 0)]
            bench_players = bench_players.sort_values(by="bench")
            # cap_name = gw_df.loc[gw_df["cap"] == 1].iloc[0]["name"]

            summary_of_actions += "---\n"
            summary_of_actions += "Lineup: \n"

            def get_display(row):
                return f"{row['name']} ({row['xP']}{', C' if row['cap'] == 1 else ''})"

            for pos in "GDMF":
                type_players = lineup_players.loc[lineup_players["type"] == pos]
                entries = type_players.apply(get_display, axis=1)
                summary_of_actions += "\t" + ", ".join(entries.tolist()) + "\n"
            summary_of_actions += "Bench: \n\t" + ", ".join(bench_players.apply(get_display, axis=1).to_list()) + "\n"
            summary_of_actions += "Lineup xPts: " + str(round(lineup_players["xp_cont"].sum(), 2)) + "\n"
            summary_of_actions += "---\n\n"
            cumulative_xpts = cumulative_xpts + round(lineup_players["xp_cont"].sum(), 2)

            statistics[w] = {
                "itb": itb[w].get_value(),
                # "nt": n_transfers[w].get_value(),
                "xP": round(squad_players["xp_cont"].sum(), 2),
                "obj": round(gw_pts[w].get_value(), 2),
                "chip": chip_decision if chip_decision != "" else None,
            }
            if options.get("delete_tmp", True):
                time.sleep(0.1)
                for file in [mps_file_name, sol_file_name, opt_file_name]:
                    try:
                        os.unlink(file)
                    except Exception:
                        pass
            buy_decisions = ", ".join(move_summary["buy"])
            sell_decisions = ", ".join(move_summary["sell"])
            chip_decisions = ", ".join(move_summary["chip"])
            if buy_decisions == "":
                buy_decisions = "-"
            if sell_decisions == "":
                sell_decisions = "-"
            if chip_decisions == "":
                chip_decisions = "-"

        # Add current solution to a list, and add a new cut
        solutions.append(
            {
                "iter": it,
                "model": model,
                "picks": picks_df,
                "total_xp": total_xp,
                "summary": summary_of_actions,
                "statistics": statistics,
                "buy": buy_decisions,
                "sell": sell_decisions,
                "chip": chip_decisions,
                "score": -model.get_objective_value(),
                # 'decay_metrics': {key: value.get_value() for key, value in decay_metrics.items()}
            }
        )

        # at least 1 different transfer in/out for the next gw
        target_gw = options.get("target_gw", next_gw)
        actions = (
            so.expr_sum(1 - t_in[p, target_gw] for p in players if t_in[p, target_gw].get_value() > 0.5)
            + so.expr_sum(t_in[p, target_gw] for p in players if t_in[p, target_gw].get_value() < 0.5)
            + so.expr_sum(1 - t_out[p, target_gw] for p in players if t_out[p, target_gw].get_value() > 0.5)
            + so.expr_sum(t_out[p, target_gw] for p in players if t_out[p, target_gw].get_value() < 0.5)
        )
        model.add_constraint(actions >= 1, name=f"cutoff_{it}")

    run_id = get_random_id(6)
    options["run_id"] = run_id
    # write solutions to csv
    for result in solutions:
        it = result["iter"]
        time_now = datetime.datetime.now()
        stamp = time_now.strftime("%Y-%m-%d_%H-%M-%S")
        solve_name = "sens_" + options["run_no"] if "run_no" in options else "regular"
        if options.get("binary_file_name"):
            bfn = options.get("binary_file_name")
            filename = f"{solve_name}_{bfn}_{stamp}_{it}"
        else:
            filename = f"{solve_name}_{stamp}_{it}"
        result["picks"].to_csv("results/" + filename + ".csv")

        if options.get("export_image", False):
            create_squad_timeline(current_squad=current_squad, statistics=result["statistics"], picks=result["picks"], filename=filename)

    print_solutions(options, solutions)


def print_solutions(options, solutions):
    horizon = options.get("horizon", 5)
    next_gw = 1 if options.get("preseason") else options.get("next_gw")
    gws = list(range(next_gw, next_gw + horizon))
    # Detailed print, e.g.
    # GW2: A, B, C -> D, E, F
    # GW3: G, H, I -> J, K, L
    # ......
    for result in solutions:
        # prints lineup and bench
        print(result["summary"])
        picks = result["picks"]
        for gw in gws:
            line_text = ""
            chip_df = picks.loc[(picks["week"] == gw) & (picks["chip"] != "")]
            chip_text = chip_df.iloc[0]["chip"] if len(chip_df) > 0 else ""
            if chip_text != "":
                line_text += "(" + chip_text + ") "
            sell_text = ", ".join(picks[(picks["week"] == gw) & (picks["t_out"] == 1)]["name"].to_list())
            buy_text = ", ".join(picks[(picks["week"] == gw) & (picks["t_in"] == 1)]["name"].to_list())
            if sell_text != "" or buy_text != "":
                line_text += sell_text + " -> " + buy_text
            else:
                line_text += "Roll"
            print(f"GW{gw}: {line_text}")

    result_table = pd.DataFrame(solutions)
    result_table = result_table.sort_values(by="score", ascending=False)
    print("\n", result_table[["iter", "sell", "buy", "chip", "score"]].to_string(index=False))

    solutions_file = options.get("solutions_file")
    if solutions_file:
        write_line_to_file(solutions_file, result, options)


def write_line_to_file(filename, result, options):
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    gw = min(result["picks"]["week"])
    score = round(result["score"], 3)
    picks = result["picks"]

    cap = picks[(picks["week"] == gw) & (picks["cap"] > 0.5)].iloc[0]["name"]
    run_id = options["run_id"]
    iter = result["iter"]
    team_id = options.get("team_id")
    chips = [options.get(x, 0) for x in ["use_wc", "use_bb", "use_tc"]]
    sell_text = ", ".join(picks[(picks["week"] == gw) & (picks["t_out"] == 1)]["name"].to_list())
    buy_text = ", ".join(picks[(picks["week"] == gw) & (picks["t_in"] == 1)]["name"].to_list())

    headers = ["run_id", "iter", "user_id", "wc", "bb", "tc", "cap", "sell", "buy", "score", "datetime"]
    data = [run_id, iter, team_id] + chips + [cap, sell_text, buy_text, score, t]
    if options.get("show_summary", False):
        headers.append("summary")
        data.append(result["summary"])

    if not os.path.exists(filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(data)


def main():
    solve()


if __name__ == "__main__":
    main()
