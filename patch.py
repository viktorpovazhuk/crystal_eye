import numpy as np

class Patch:

    coordinate = ()
    patch = np.array[0:0, 0:0]

    def __init__(self, coordinate, patch) -> None:
        self.coordinate = coordinate
        self.patch = patch