import argparse
import datetime
import json
import os
import random
import string
import subprocess
import threading
import time

import numpy as np
import pandas as pd
import sasoptpy as so

MAX_TRANSFERS_PER_GW = 3


def get_random_id(n):
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(n))


def solve(runtime_options=None):
    with open("settings.json", "r") as f:
        options = json.load(f)

    solve_name = "sens_" + options["run_no"] if "run_no" in options else f"reg_{get_random_id(5)}"

    parser = argparse.ArgumentParser(add_help=False)
    for key in options.keys():
        if isinstance(options[key], (list, dict)):
            continue

        parser.add_argument(f"--{key}", default=options[key], type=type(options[key]))

    args = parser.parse_known_args()[0]
    options = {**options, **vars(args)}
    if runtime_options is not None:
        options = {**options, **runtime_options}

    horizon = options.get("horizon", 5)
    preseason = options.get("preseason", False)
    # preseason hard coded value TODO: FIX
    if preseason:
        next_gw = 1
    gws = list(range(next_gw, next_gw + horizon))
    all_gws = [next_gw - 1] + gws

    df = pd.read_csv("xpts.csv", index_col="ID", encoding="latin-1").fillna(0)
    if options.get("randomised", False):
        rng = np.random.default_rng(seed=options.get("seed"))
        for w in gws:
            noise = df[f"{w}_Pts"] * (92 - df[f"{w}_xMins"]) / 134 * rng.standard_normal(size=len(df))
            df[f"{w}_Pts"] = df[f"{w}_Pts"] + noise

    players = df.index.to_list()
    el_types = df["Pos"].unique().tolist()
    teams = df["Team"].unique().tolist()
    player_values = (df["BV"] * 10).to_dict()

    model = so.Model(name="LOI")

    squad = model.add_variables(players, all_gws, name="squad", vartype=so.binary)
    lineup = model.add_variables(players, all_gws, name="lineup", vartype=so.binary)
    bench = model.add_variables(players, all_gws, range(4), name="bench", vartype=so.binary)
    cap = model.add_variables(players, gws, name="cap", vartype=so.binary)
    itb = model.add_variables(all_gws, name="itb", vartype=so.integer, lb=0)
    # model.add_constraint(itb[1] == 10, name="1.5itb")
    t_in = model.add_variables(players, gws, name="t_in", vartype=so.binary)
    t_out = model.add_variables(players, gws, name="t_out", vartype=so.binary)
    use_wc = model.add_variables(gws, name="use_wc", vartype=so.binary)
    use_bb = model.add_variables(gws, name="use_bb", vartype=so.binary)
    use_tc = model.add_variables(players, gws, name="use_tc", vartype=so.binary)
    use_tc_gw = {w: so.expr_sum(use_tc[p, w] for p in players) for w in gws}

    model.add_constraint(so.expr_sum(use_wc[w] for w in gws) <= 1, name="max_one_wc")
    model.add_constraints((so.expr_sum(squad[p, w] for p in players) == 15 for w in gws), name="15_man_squad")
    model.add_constraints((so.expr_sum(lineup[p, w] for p in players) == 11 for w in gws), name="11_man_lineup")
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
        (lineup_type_count[t, w] <= type_data[t]["max"] for t in el_types for w in gws),
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
    if preseason:
        model.add_constraint(itb[next_gw - 1] == 1000, name="initial_itb")
    n_transfers = {w: so.expr_sum(t_in[p, w] for p in players) for w in gws}
    bought_amount = {w: so.expr_sum(player_values[p] * t_in[p, w] for p in players) for w in gws}
    sold_amount = {w: so.expr_sum(player_values[p] * t_out[p, w] for p in players) for w in gws}

    model.add_constraints(
        (t_in[p, w] + t_out[p, w] <= 1 for p in players for w in gws),
        name="tr_in_out_limit",
    )

    transfers_allowed = {w: (15 if w == 1 else 1 - use_wc[w]) * MAX_TRANSFERS_PER_GW + use_wc[w] * 15 for w in gws}
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
                model.add_constraint(squad[x[0], x[w]] == 0, name=f"banned_{x[0]}_{x[1]}")

    # Objective
    pts_player_week = {(p, w): df.loc[p, f"{w}_Pts"] for p in players for w in gws}
    lineup_pts = {w: so.expr_sum(pts_player_week[p, w] * (lineup[p, w] + cap[p, w] + use_tc[p, w]) for p in players) for w in gws}

    model.add_constraints((so.expr_sum(bench[p, w, n] for n in range(4) for p in players) == 4 for w in gws), name="4_subs")
    model.add_constraints((so.expr_sum(bench[p, w, 0] for p in players if player_el_types[p] == "G") == 1 for w in gws), name="GK_sub0")

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

    # model.add_constraint(bench[17, 1, 0] == 1, name="sub0")
    # model.add_constraint(bench[3, 1, 1] == 1, name="sub1")
    # model.add_constraint(bench[4, 1, 2] == 1, name="sub2")
    # model.add_constraint(bench[6, 1, 3] == 1, name="sub3")

    bench_pts = {w: sub0[w] + sub1[w] + sub2[w] + sub3[w] for w in gws}
    gw_pts = {w: lineup_pts[w] + bench_pts[w] for w in gws}

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
            highs_exec = options.get("solver_path") or "highs"

            secs = options.get("secs", 20 * 60)
            presolve = options.get("presolve", "on")
            gap = options.get("gap", 0)
            random_seed = options.get("random_seed", 0)

            with open(opt_file_name, "w") as f:
                f.write(f"""mip_rel_gap = {gap}""")

            command = f"{highs_exec} --parallel on --options_file {opt_file_name} --random_seed {random_seed} --presolve {presolve} --model_file {mps_file_name} --time_limit {secs} --solution_file {sol_file_name}"

            if use_cmd:
                os.system(command)
            else:

                def print_output(process):
                    while True:
                        try:
                            output = process.stdout.readline()
                            if "Solving report" in output:
                                time.sleep(2)
                                process.kill()
                            elif output == "" and process.poll() is not None:
                                break
                            elif output:
                                print(output.strip())
                        except Exception:
                            print("File closed")
                            break
                    process.kill()

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
                    if not cols_started and "# Columns" not in line:
                        continue
                    elif "# Columns" in line:
                        cols_started = True
                        continue
                    elif cols_started and line[0] != "#":
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
                    elif line[0] == "#":
                        break

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

        picks_df = pd.DataFrame(picks).sort_values(by=["week", "lineup", "type", "xP"], ascending=[True, False, True, True])
        total_xp = so.expr_sum((lineup[p, w] + cap[p, w]) * pts_player_week[p, w] for p in players for w in gws).get_value()

        picks_df.sort_values(by=["week", "squad", "lineup", "bench", "type"], ascending=[True, False, False, True, True], inplace=True)

        # Writing summary
        summary_of_actions = ""
        move_summary = {"chip": [], "buy": [], "sell": []}
        cumulative_xpts = 0

        # collect statistics
        statistics = {}
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
                    if w == next_gw:
                        move_summary["buy"].append(df["Name"][p])

            for p in players:
                if t_out[p, w].get_value() > 0.5:
                    summary_of_actions += f"Sell {p} - {df['Name'][p]}\n"
                    if w == next_gw:
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
            summary_of_actions += "Bench: \n\t" + ", ".join(bench_players["name"].tolist()) + "\n"
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
        actions = (
            so.expr_sum(1 - t_in[p, next_gw] for p in players if t_in[p, next_gw].get_value() > 0.5)
            + so.expr_sum(t_in[p, next_gw] for p in players if t_in[p, next_gw].get_value() < 0.5)
            + so.expr_sum(1 - t_out[p, next_gw] for p in players if t_out[p, next_gw].get_value() > 0.5)
            + so.expr_sum(t_out[p, next_gw] for p in players if t_out[p, next_gw].get_value() < 0.5)
        )
        model.add_constraint(actions >= 1, name=f"cutoff_{it}")

    for result in solutions:
        it = result["iter"]
        print(result["summary"])
        time_now = datetime.datetime.now()
        stamp = time_now.strftime("%Y-%m-%d_%H-%M-%S")
        solve_name = "sens_" + options["run_no"] if "run_no" in options else "regular"
        if options.get("binary_file_name"):
            bfn = options.get("binary_file_name")
            filename = f"{solve_name}_{bfn}_{stamp}_{it}"
        else:
            filename = f"{solve_name}_{stamp}_{it}"
        result["picks"].to_csv("results/" + filename + ".csv")
    return solutions


def print_solutions(options, solutions):
    horizon = options.get("horizon", 5)
    preseason = options.get("preseason", False)
    if preseason:
        next_gw = 1
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
            print(f"\tGW{gw}: {line_text}")

    result_table = pd.DataFrame(solutions)
    result_table = result_table.sort_values(by="score", ascending=False)
    print(result_table[["iter", "sell", "buy", "chip", "score"]].to_string(index=False))


def main():
    with open("settings.json", "r") as f:
        options = json.load(f)
    solutions = solve(options)
    print_solutions(options, solutions)


if __name__ == "__main__":
    main()
