from concurrent.futures import ProcessPoolExecutor
from itertools import product

from solve import solve


def get_dict_combinations(my_dict):
    keys = my_dict.keys()
    for key in keys:
        if my_dict[key] is None or len(my_dict[key]) == 0:
            my_dict[key] = [None]
    all_combs = [dict(zip(my_dict.keys(), values)) for values in product(*my_dict.values())]
    feasible_combs = []
    for comb in all_combs:
        comb_copy = comb.copy()
        c_values = [i for i in comb_copy.values() if i is not None]
        if len(c_values) == len(set(c_values)):
            feasible_combs.append(comb)
    return feasible_combs


def run_parallel_solves(combinations, max_workers=8):
    jobs = [{f"use_{chip}": gw for chip, gw in combination.items() if gw} for combination in combinations]
    print(f"{len(jobs)} total jobs")
    print(jobs)
    # Use ProcessPoolExecutor to run commands in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(solve, jobs)


if __name__ == "__main__":
    chip_gameweeks = {"bb": [16, 19], "wc": [12, 13, 14, 15, 16, 17, 18, 19, 20], "tc": []}
    combinations = get_dict_combinations(chip_gameweeks)

    run_parallel_solves(combinations)
