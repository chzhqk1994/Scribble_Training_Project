CLASS_NUM = 14

LABEL_DICT = {
    "bg": 0,
    "waterproof": 1,
    "facility": 2,
    "something": 3,
    "airconditioner": 4,
    "round_airconditioner": 5,
    #"wall": 6,
    "garden": 6,
    "corrugated": 7,
    "slate": 8,
    "solarpanel": 9,
    "heliport": 10,
    "tree": 11,
    "concrete": 12,
    "clay": 13
}

COLOR_MAP_DICT = {
    0: [0, 0, 0],
    1: [0, 255, 0],
    2: [255, 0, 0],
    3: [255, 0, 255],
    4: [153, 51, 0],
    5: [153, 153, 153],
    #6: [0, 0, 255],
    6: [25, 111, 61],
    7: [187, 143, 206],
    8: [142, 68, 173],
    9: [243, 156, 18],
    10: [0, 255, 255],
    11: [128, 128, 000],  # tree
    12: [64, 64, 64],
    13: [233, 153, 102]
    # 12: [64, 64, 64],
}
