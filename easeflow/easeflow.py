from perlin_noise import PerlinNoise
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
import easing_functions
from typing import TypeVar
from pyspark.sql import SparkSession, DataFrame


def get_spark() -> SparkSession:
    try:
        from databricks.connect import DatabricksSession

        return DatabricksSession.builder.getOrCreate()
    except ImportError:
        return SparkSession.builder.getOrCreate()


# Constants
NOISE_SPEED = 0.005

# Create PerlinNoise object once
noise = PerlinNoise(octaves=10, seed=1)

T = TypeVar("T", bound=easing_functions.easing.EasingBase)


def make_udf(
    easing_function: T, min_val: float = 0.0, max_val: float = 1.0
) -> callable:
    """
    Creates a PySpark UDF (User Defined Function) that applies an easing function
    with optional Perlin noise. This version assumes that the input `t` is normalized
    between 0 and 1.

    The easing function generates smooth transitions between `min_val` and `max_val`,
    and Perlin noise is optionally added as a percentage-based variation.

    Args:
        easing_function (T): The easing function to apply. Must be a subclass of `easing_functions.easing.EasingBase`.
        min_val (float): The minimum value of the output range.
        max_val (float): The maximum value of the output range.

    Returns:
        pyspark.sql.functions.udf: A PySpark UDF that takes two arguments:
            - `t` (float): A normalized value in the range [0, 1].
            - `noise_pct` (float): The percentage of noise to apply (0 = no noise, 1 = max noise).

    Example 1: Using `norm_df` for Normalized Input
    -----------------------------------------------
    ```python
    # Create an easing function UDF
    easing_udf = make_udf(easy.QuinticEaseIn, min_val=500, max_val=1000)

    # Generate a DataFrame with easing function applied
    df = (
        norm_df(365)
        .withColumn("v", easing_udf(F.col("t"), lit(0)))  # No noise
        .withColumn("v_noise", easing_udf(F.col("t"), lit(0.3)))  # With noise
    )

    # Display DataFrame in Databricks
    display(df)
    ```

    Example 2: Generating a Time-Series with Dates
    ----------------------------------------------
    ```python
    import datetime

    # Define start date
    dataset_len = 365
    start_date = datetime.datetime.today().replace(day=1) - datetime.timedelta(days=dataset_len)

    # Generate normalized data with a date column
    df = (
        norm_df(dataset_len)
        .withColumn("date", F.date_add(lit(start_date), F.col("id").cast("integer")))  # Map ID to a date
        .withColumn("v", easing_udf(F.col("t"), lit(0)))  # No noise
        .withColumn("v_noise", easing_udf(F.col("t"), lit(0.3)))  # With noise
    )

    # Display the results in Databricks
    display(df)
    ```

    Notes:
    ------
    - **Best Practice**: Use `norm_df(n)` instead of manually normalizing values.
    - The `"id"` column allows easy **date mapping** for time-series data.
    - The `"t"` column should be used as input for easing functions.
    - Works well for **generating synthetic datasets** with smooth transitions and noise.
    """

    # Initialize the easing function for the range [0, 1]
    easing = easing_function(start=min_val, end=max_val, duration=1)

    def get_val(t: float, noise_pct: float) -> float:
        """
        Computes the easing function value with optional Perlin noise.

        Args:
            t (float): Normalized input in the range [0, 1].
            noise_pct (float): Percentage of noise to apply.

        Returns:
            float: Smoothed output value.
        """
        y0 = easing(t)  # Compute base easing value
        # Scale the noise to the easing range and apply it [-0.5, 0.5]

        y1 = y0 + y0 * noise_pct * 2.0 * (noise(t * NOISE_SPEED) - 0.5)

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

    Example Usage in Databricks:
    ----------------------------
    ```python
    df = norm_df(365)
    display(df)
    ```

    Expected Output:
    ----------------
    ```
    +---+------------------+
    | id|                 t|
    +---+------------------+
    |  0|  0.0             |
    |  1|  0.00274         |
    |  2|  0.00548         |
    |...|  ...             |
    |364|  1.0             |
    +---+------------------+
    ```
    """
    return (
        get_spark().range(0, n).withColumn("t", (F.col("id") / (n - 1)).cast("double"))
    )
