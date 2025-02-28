from perlin_noise import PerlinNoise
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
import easing_functions
from typing import TypeVar

# Constants
NOISE_SPEED = 0.005

# Create PerlinNoise object once
noise = PerlinNoise(octaves=10, seed=1)

T = TypeVar("T", bound=easing_functions.easing.EasingBase)


def make_udf(
    dataset_len: int,
    easyFunction: T,
    min_val: float,
    max_val: float,
    noise_pct: float = 0.3,
) -> callable:
    """
    Returns a pyspark UDF that is a composite of an easing function
    modified with a noise function.

    Args:
        dataset_len (int): The length of the final dataset.
        easyFunction (T): The easing function to use. Must be a subtype of `easing_functions.easing.EasingBase`.
        min_val (float): The minimum value for the easing function.
        max_val (float): The maximum value for the easing function.
        noise_pct (float, optional): The amount of noise to apply to the easing function. Defaults to 0.3.

    Returns:
        callable: A pyspark UDF that applies the composite function to a given input.

    The noise function is a multiplier to the output of the easing function by
    `noise_pct`. In summary, if `e` is the easing function at a given `t`,
    `n(x)` is Perlin noise computed at `x`, and `p` is the noise amount,
    the UDF return will be `e * [1 + p * n(e)]`.
    """

    easing = easyFunction(start=min_val, end=max_val, duration=dataset_len)

    def get_val(x: float) -> float:
        y0 = easing(x)
        y1 = y0 * (1 + noise_pct * (1 + noise(x * NOISE_SPEED)))
        return y1

    return F.udf(get_val, FloatType(), useArrow=True)
