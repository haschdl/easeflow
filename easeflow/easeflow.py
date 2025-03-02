from perlin_noise import PerlinNoise
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
import easing_functions
from typing import TypeVar
from pyspark.sql import SparkSession


def get_spark() -> SparkSession:
    try:
        from databricks.connect import DatabricksSession

        return DatabricksSession.builder.getOrCreate()
    except ImportError:
        return SparkSession.builder.getOrCreate()


T = TypeVar("T", bound=easing_functions.easing.EasingBase)


def make_udf_noise(
    min_value: float,
    max_value: float,
    noise_speed: float,
    octaves: int = 10,
    seed: int = 1,
) -> callable:
    """
    Creates a PySpark UDF (User Defined Function) that returns a Perlin noise value
    between `min_value` and `max_value` based on the input `t` and `noise_speed`.

    Useful when you need generate data with smooth, continuous variation that is
    not strictly random.

    Args:
    min_value (float): The minimum value of the Perlin noise range.
    max_value (float): The maximum value of the Perlin noise range.
    noise_speed (float): Controls the speed/frequency of Perlin noise variation.
    noise_offset (float): The offset value for the Perlin noise generator-
    octaves (int): The number of octaves in the Perlin noise function.
    seed (int): The seed value for the Perlin noise generator.

    Returns:
    pyspark.sql.functions.udf: A PySpark UDF that takes one argument:
        - `t` (float): A normalized value in the range [0, 1].

    Example:
    --------
    ```python
    # Create a Perlin noise UDF
    noise_udf = make_udf_noise(min_value=500, max_value=1000, noise_speed=0.01)
    ```
    """
    _noise = PerlinNoise(octaves=octaves, seed=seed)

    def get_val(t: float, offset: float = 0) -> float:
        """
        Computes the Perlin noise value based on the input `t` and `noise_speed`.

        Args:
            t (float): Normalized input in the range [0, 1].

        Returns:
            float: Perlin noise value.
        """
        noise_factor = _noise(offset + t * noise_speed)  # [-1, 1]
        return min_value + (max_value - min_value) * (1 + noise_factor) / 2

    return F.udf(get_val, FloatType(), useArrow=True)


def make_udf(
    easing_function: T = None,
    start_value: float = 0.0,
    end_value: float = 1.0,
    noise_speed: float = 1.0,
    octaves: int = 10,  # Allow configuring Perlin noise complexity
    seed: int = 1,  # Ensure different seeds create independent noise patterns
) -> callable:
    """
    Creates a PySpark UDF (User Defined Function) that applies an easing function
    with optional Perlin noise. This version assumes that the input `t` is normalized
    between 0 and 1.

    The easing function generates smooth transitions between `start_value` and `end_value`,
    and Perlin noise is optionally added as a percentage-based variation.

    Args:
        easing_function (T): The easing function to apply. Must be a subclass of `easing_functions.easing.EasingBase`.
        start_value (float): The initial value of the transition.
        end_value (float): The final value of the transition.
        noise_speed (float): Controls the speed/frequency of Perlin noise variation.

    Returns:
        pyspark.sql.functions.udf: A PySpark UDF that takes two arguments:
            - `t` (float): A normalized value in the range [0, 1].
            - `noise_pct` (float): The percentage of noise to apply (0 = no noise, 1 = max noise).

    Example:
    --------
    ```python
    # Create an easing function UDF with a custom noise speed
    easing_udf = make_udf(
        easy.QuinticEaseIn, start_value=500, end_value=1000, noise_speed=0.01
    )

    # Apply to a DataFrame
    df = (
        norm_df(365)
        .withColumn("v", easing_udf(F.col("t"), F.lit(0)))  # No noise
        .withColumn("v_noise", easing_udf(F.col("t"), F.lit(0.3)))  # With noise
    )

    display(df)
    ```
    """
    # Create a new PerlinNoise instance per UDF
    _noise = PerlinNoise(octaves=octaves, seed=seed)

    # Initialize the easing function for the range [0, 1]
    easing = easing_function(start=start_value, end=end_value, duration=1)

    def get_val(t: float, noise_pct: float, offset: float = 0) -> float:
        """
        Computes the easing function value with optional Perlin noise.

        Args:
            t (float): Normalized input in the range [0, 1].
            noise_pct (float): Percentage of noise to apply.

        Returns:
            float: Smoothed output value with Perlin noise applied.
        """
        y0 = easing(t)  # Compute base easing value
        noise_factor = _noise(t * noise_speed + offset)  # [-1, 1]
        y1 = y0 * (1 + noise_pct * noise_factor)  # Apply scaled noise effect
        return y1

    return F.udf(get_val, FloatType(), useArrow=True)


def norm_df(n: int):
    """
    Generates a PySpark DataFrame with two columns:
        - 'id': A sequential integer index from 0 to n-1.
        - 't': A normalized value between 0 and 1.

    This function is useful for generating structured datasets where both
    raw index (`id`) and normalized (`t`) values are needed.

    Args:
        n (int): The number of rows in the DataFrame.

    Returns:
        pyspark.sql.DataFrame: A DataFrame with columns:
            - 'id' (int): Row index.
            - 't' (float): Normalized value between 0 and 1.

    Example:
    --------
    ```python
    df = norm_df(365)
    display(df)
    ```
    """
    return (
        get_spark()
        .range(0, n)
        .withColumn("t", (F.col("id") / (n - 1)).cast(FloatType()))
    )
