import numpy as np
from scipy.stats import gaussian_kde, wasserstein_distance
from itertools import combinations
import multiprocessing as mp
import math
import time
from tqdm import tqdm


# Kde functions ---------------------------------------------------------------
def kde_mode(values, num_points=1000):

    if len(values) < 2:
        return None  

    values = np.asarray(values)

    # Check for identical values (zero variance then)
    if np.allclose(values, values[0]):
        return values[0]

    try:
        kde = gaussian_kde(values)
    except np.linalg.LinAlgError:
        return None  

    x_range = np.linspace(values.min(), values.max(), num_points)
    density = kde(x_range)

    return x_range[np.argmax(density)]

def kde_basic(values, x_vals):
    kde = gaussian_kde(values)
    density = kde(x_vals)

    area = np.trapezoid(density, x_vals)

    if area > 0:
        density /= area

    return density


# process for one slice size----------------------------------------------------
def compute_single_distance(args):
    """inner parallelization of Wasserstein distances"""
    i, j, slice_i, slice_j, a = args
    dist = wasserstein_distance(slice_i, slice_j)
    return {"indices": (i, j), "d1": np.sqrt(a) * dist}


def process_slice_size_inner_parallel(data, a, n_cores):
    """
    Processes a single slice size sequentially, but parallelizes the heavy 
    distance matrix computation. Used for very small slice sizes"""

    remainder = len(data) % a
    if remainder != 0:
        data = data[remainder:]

    slices = [data[i:i+a] for i in range(0, len(data), a)]

    #arguments for the multiprocessing pool
    tasks_generator = (
        (i, j, slices[i], slices[j], a)
        for i, j in combinations(range(len(slices)), 2)
    )

    with mp.Pool(n_cores) as pool:
        g_results = list(pool.imap_unordered(
            compute_single_distance, 
            tasks_generator, 
            chunksize=1000))

    g_values = [e["d1"] for e in g_results]
    thresh = kde_mode(g_values) if g_values else None

    return a, {
        "g_results": g_results,
        "thresholds": [thresh]
    }

def process_slice_size_sequential(args):
    """
    Processes a single slice size completely sequentially. 
    Used when parallelizing heavily over the slice sizes themselves
    """
    data, a = args

    remainder = len(data) % a
    if remainder != 0:
        data = data[remainder:]

    slices = [data[i:i+a] for i in range(0, len(data), a)]
    
    
    g_results = []
    for i, j in combinations(range(len(slices)), 2):
        g_results.append({
            "indices": (i, j), 
            "d1": np.sqrt(a) * wasserstein_distance(slices[i], slices[j])})

    g_values = [e["d1"] for e in g_results]
    thresh = kde_mode(g_values) if g_values else None

    return a, {
        "g_results": g_results,
        "thresholds": [thresh]
    }



# Wasserstein dictionary---------------------------------------------
def wasserstein_dictionary_parallel(data, start_size, end_size, n_cores=None):
    """
    Builds the Wasserstein dictionary dynamically switching parallelization 
    strategies based on the slice size relative to the data length
    """
    if n_cores is None:
        n_cores = mp.cpu_count()

    total_length = len(data)
    threshold_limit = 0.005 * total_length # a 0.5% threshold for parallelization

    sizes = list(range(start_size, end_size + 1))

    # split sizes into two categories based on parallelization threshold
    small_sizes = [a for a in sizes if a < threshold_limit]
    large_sizes = [a for a in sizes if a >= threshold_limit]

    results_dict = {}

    # Inner parallelization for small slice sizes (many combinations)
    if small_sizes:
        for a in tqdm(small_sizes, desc=f"Small slice sizes up to {max(small_sizes)}"):
            _, res = process_slice_size_inner_parallel(data, a, n_cores)
            results_dict[a] = res

    # Outer parallelization for large slice sizes (slice process parallelization)
    if large_sizes:
        large_args_gen = ((data, a) for a in large_sizes)
        with mp.Pool(n_cores) as pool:
            large_results = []
            for res in tqdm(
                pool.imap_unordered(
                    process_slice_size_sequential,
                    large_args_gen,
                    chunksize=1
                ),
                total=len(large_sizes),
                desc="Large slice sizes",
                position=0):
                large_results.append(res)
        results_dict.update(dict(large_results))

    return results_dict



# Convergence analysis -----------------------------------------
# first a few global threshold candidates
def find_smallest_threshold(dictionary):

    smallest_threshold = float('inf')

    for result in dictionary.values():
        thresh = result["thresholds"][0]
        if thresh is not None and thresh < smallest_threshold:
            smallest_threshold = thresh

    return smallest_threshold

def find_biggest_threshold(dictionary):

    biggest_threshold = float('-inf')

    for result in dictionary.values():
        thresh = result["thresholds"][0]
        if thresh is not None and thresh > biggest_threshold:
            biggest_threshold = thresh

    return biggest_threshold

def find_mean_threshold(dictionary):
    threshs = [
        result["thresholds"][0]
        for result in dictionary.values()
        if result["thresholds"][0] is not None
    ]
    return np.mean(threshs) if threshs else None

def find_median_threshold(dictionary):
    threshs = [
        result["thresholds"][0]
        for result in dictionary.values()
        if result["thresholds"][0] is not None
    ]
    return np.median(threshs) if threshs else None

def find_good_x(array, y):
    """ function to find the index for which all later slice pairs are below the threshold"""
    array_set = set(array)
    good_x = y

    for i in range(y - 1, -1, -1):
        if all((i, j) in array_set for j in range(i + 1, y + 1)):
            good_x = i
        else:
            break

    return None if good_x == y else good_x


def find_convergence(dictionary, n_frames, threshold):
    """ find the estimated convergence point for each slice size and return the mean of the votes"""
    convergence_points = []

    for a, values in dictionary.items():

        number_of_slides = math.floor(n_frames / a)
        upper_index = number_of_slides - 1

        filtered_results = [
            entry for entry in values['g_results']
            if entry["d1"] <= threshold
        ]

        indices_list = [entry["indices"] for entry in filtered_results]

        if indices_list:
            c_point = find_good_x(indices_list, upper_index)

            if c_point is not None:
                convergence = a * (upper_index - c_point + 1)
                convergence_point = n_frames - convergence
                convergence_points.append([a, convergence_point])

    points = [x[1] for x in convergence_points]

    mean_co = int(np.mean(points)) if points else n_frames

    return mean_co, convergence_points



# MAIN FUNCTION-------------------------------------------------------
def estimate_convergence(
    x,
    y,
    slice_start=10,
    n_cores=None
):
    """
    Estimate convergence using the Wasserstein-dictionary approach

    Parameters:
    x, y : array-like
        Time array and trajectory
    slice_start : int
        Smallest slice size to consider (heavy influence on runtime)
    n_cores : int or None
        Number of processes for multiprocessing

    Returns:
    conv_points : dict
        Estimated convergence point for each threshold method as a frame
        index: the integer mean over all slice sizes that give an
        estimate, of the first frame from which the trajectory is considered
        converged. If no slice size gives an estimate, the value is the number of
        frames
    thresholds : dict
        global thresholds for each computation method
    details : dict
        Per-slice convergence details for each threshold computation method:
        a list of [slice_size, convergence_frame] pairs, one for each slice
        size that gives an estimate
    runtime : tuple
        (hours, minutes, seconds)
    """

    start_time = time.perf_counter()
    y = np.asarray(y)
    n_frames = len(x)

    slice_end = int(n_frames / 3)
    wd = wasserstein_dictionary_parallel(
        y,
        slice_start,
        slice_end,
        n_cores=n_cores)

    # collect all the different thresholds
    thresholds = {}
    thresholds.update({"min":find_smallest_threshold(wd)})
    thresholds.update({"max":find_biggest_threshold(wd)})
    thresholds.update({"mean":find_mean_threshold(wd)})
    thresholds.update({"median":find_median_threshold(wd)})
    threshs = [result["thresholds"][0]
            for result in wd.values()
            if result["thresholds"][0] is not None]
    if len(threshs) == 0:
        raise ValueError("No per-slice thresholds found")
    elif len(threshs) == 1:
        thresholds.update({"kde":threshs[0]})
    else:
        thresholds.update({"kde":kde_mode(threshs)})

    conv_points = {}
    details = {}
    for method in thresholds.keys():
        conv_point, detail = find_convergence(
            wd,
            n_frames,
            thresholds[method]
        )
        conv_points.update({method:conv_point})
        details.update({method:detail})

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    runtime = (hours, minutes, seconds)

    return conv_points, thresholds, details, runtime

# example usage:
if __name__=="__main__":
    import pandas as pd
    # block-averaged RMSD of the unfolding of triptophan zipper 2 (x: time in ps, RMSD in Angstrom)
    df = pd.read_csv("trpzip_unfolding_RMSD.csv")
    x = df["x"].to_numpy()
    rmsd = df["RMSD"].to_numpy()

    conv_points, thresholds, details, runtime = estimate_convergence(
        x,
        rmsd,
        slice_start=50,
        n_cores=None)

    print(f"runtime: {runtime[0]}h {runtime[1]}min {runtime[2]}s")
    for method, frame in conv_points.items():
        if frame >= len(x):
            print(f"{method:>6}: threshold {thresholds[method]:.3f} -> not converged")
        else:
            print(f"{method:>6}: threshold {thresholds[method]:.3f} -> "
                  f"converged from frame {frame} (t = {x[frame] / 1000:.1f} ns)")
    