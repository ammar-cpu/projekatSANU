import numpy as np


def parse_vrp(path):
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]

    dimension = None
    capacity = None
    section = None
    raw_coords = {}
    raw_demands = {}
    depot_raw_id = None

    for line in lines:
        if line == "NODE_COORD_SECTION":
            section = "coord"
            continue
        if line == "DEMAND_SECTION":
            section = "demand"
            continue
        if line == "DEPOT_SECTION":
            section = "depot"
            continue
        if line == "EOF":
            break

        if section is None:
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key == "DIMENSION":
                dimension = int(value)
            elif key == "CAPACITY":
                capacity = int(value)
        elif section == "coord":
            raw_id, x, y = line.split()
            raw_coords[int(raw_id)] = (float(x), float(y))
        elif section == "demand":
            raw_id, demand = line.split()
            raw_demands[int(raw_id)] = int(demand)
        elif section == "depot":
            value = int(line)
            if value == -1:
                section = None
            elif depot_raw_id is None:
                depot_raw_id = value

    # the depot moves to index 0, customers keep ascending file order after it
    order = [depot_raw_id] + sorted(i for i in raw_coords if i != depot_raw_id)

    coords = np.array([raw_coords[i] for i in order])
    return {
        "coords": coords,
        "demands": np.array([raw_demands[i] for i in order]),
        "dist": compute_distance_matrix(coords),
        "capacity": capacity,
        "depot_id": 0,
        "dimension": dimension,
    }


def compute_distance_matrix(coords):
    # TSPLIB EUC_2D convention: distances rounded to the nearest integer. The
    # published CVRPLIB optima assume this, so without the rounding the gap to
    # optimum is not comparable.
    diff = coords[:, None, :] - coords[None, :, :]
    return np.round(np.sqrt((diff ** 2).sum(axis=-1)))
