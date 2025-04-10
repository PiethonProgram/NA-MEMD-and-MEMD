import numpy as np
from scipy.interpolate import CubicSpline
# from scipy.interpolate import interp1d
from math import pi, sqrt, sin, cos
import sys


def nth_prime(nth):
    """
       Computes the first nth prime numbers.

       For small nth (<= 5), returns a predefined list of primes.
       For larger nth, uses a sieve method based on an upper-limit estimate
       (derived from the prime number theorem).

       Parameters:
         nth (int): The number of prime numbers to generate. Must be >= 1.

       Returns:
         numpy.ndarray: An array containing the first nth prime numbers.

       Raises:
         ValueError: If nth is less than 1.
   """
    if nth > 0:
        if nth <= 5:    # call corner cases
            return small_primes(nth)
        else:
            return large_primes(nth)    # call general cases

    else:
        raise ValueError("n must be >= 1 for list of n prime numbers")


def small_primes(nth):      # corner cases for nth prime estimation using log
    """
        Returns the first nth prime numbers from a predefined list.

        Parameters:
          nth (int): Number of primes to return.

        Returns:
          numpy.ndarray: Array containing the first nth primes.
    """
    list_prime = np.array([2, 3, 5, 7, 11])
    return list_prime[:nth]


def large_primes(nth):
    """
        Generates the first nth prime numbers using a sieve algorithm.

        Estimates an upper limit based on the prime number theorem and then creates
        a sieve for numbers of the form 6k±1 to efficiently filter out non-prime numbers.

        Parameters:
          nth (int): Number of primes to generate.

        Returns:
          numpy.ndarray: Array containing the primes.
    """
    # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n/3035188#3035188
    lim = int(nth * (np.log(nth) + np.log(np.log(nth)))) + 1
    sieve = np.ones(lim // 3 + (lim % 6 == 2), dtype=bool)
    for i in range(1, int(lim ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False

    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def hamm(n, base):
    """
        Generates a Hammersley sequence of length n with a given base.

        The Hammersley sequence is used for quasi-random sampling. For base > 1, the function
        iteratively computes the digits of each number in the given base. For base <= 1, a mirrored
        approach is used.

        Parameters:
          n (int): Length of the sequence.
          base (int): The base used for generating the digits.

        Returns:
          numpy.ndarray: Generated Hammersley sequence.
    """
    sequence = np.zeros(n)
    if base > 1:    # generate initial seed and calculate inverse
        seed = np.arange(1, n + 1)
        invert_base = 1 / base

        while np.any(seed != 0):    # loop till all elements == 0
            digit = np.remainder(seed, base)
            sequence += digit * invert_base
            invert_base /= base
            seed = np.floor(seed / base)

    else:   # generate temporary array and calculate sequence for base values <= 1
        temp = np.arange(1, n + 1)
        sequence = (np.remainder(temp, (-base + 1)) + 0.5) / (-base)

    return sequence


def stop(mode, t_array, sd, sd2, tol, seq, ndir, N, N_dim):     # Stop criterion using Standard Deviation
    """
        Determines the stop criterion for the sifting process using standard deviation.

        This function computes the envelope mean of the current mode (via envelope_mean) and
        calculates a normalized measure sx. It then assesses whether the envelope variation and
        the number of extrema satisfy the stopping thresholds.

        Parameters:
          mode (numpy.ndarray): The current mode signal.
          t_array (numpy.ndarray): Array of time indices.
          sd (float): Threshold for the first standard deviation criterion.
          sd2 (float): Secondary threshold criterion.
          tol (float): Tolerance value for the proportion of points exceeding sd.
          seq (numpy.ndarray): Hammersley sequence used for envelope estimation.
          ndir (int): Number of directions (projections) for the envelope computation.
          N (int): Total number of data points.
          N_dim (int): Dimensionality (number of channels) of the signal.

        Returns:
          tuple: (stp, env_mean)
            stp (bool): True if the stopping condition is met.
            env_mean (numpy.ndarray): The computed envelope mean.

        In case of an exception, returns default values.
    """
    try:
        env_mean, nem, nzm, amp = envelope_mean(mode, t_array, seq, ndir, N, N_dim)  # envelope mean call
        sx = np.sqrt(np.sum(env_mean ** 2, axis=1))     # sum of squares
        sx = np.divide(sx, amp, where=amp != 0)     # normalize sx

        # stopping criteria calculation
        mean_sx_gt_sd = np.mean(sx > sd)
        any_sx_gt_sd2 = np.any(sx > sd2)
        any_nem_gt_2 = np.any(nem > 2)

        stp = not ((mean_sx_gt_sd > tol or any_sx_gt_sd2) and any_nem_gt_2)     # check stop criteria met

    except:     # return default values on exception
        env_mean = np.zeros((N, N_dim))
        stp = True

    return stp, env_mean


def fix(m, t, seq, ndir, stp_cnt, counter, N, N_dim):   # Stopping criterion based on number of iterations
    """
        Applies a fixed-number-of-iterations stopping criterion.

        This function computes the envelope mean and checks the absolute difference between the
        number of zero crossings and extrema. Depending on whether the difference exceeds a threshold,
        it either resets or increments a counter. Sifting stops when the counter exceeds stp_cnt.

        Parameters:
          m (numpy.ndarray): The current mode being processed.
          t (numpy.ndarray): Time indices.
          seq (numpy.ndarray): Hammersley sequence for interpolation.
          ndir (int): Number of directions.
          stp_cnt (int): Fixed number of iterations to perform.
          counter (int): Current iteration counter.
          N (int): Total number of data points.
          N_dim (int): Signal dimensionality.

        Returns:
          tuple: (stp, env_mean, counter)
            stp (bool or int): True if stop condition is met, or 0 if not.
            env_mean (numpy.ndarray): The computed envelope mean.
            counter (int): Updated counter value.

        On exception, returns default envelope and stops.
    """
    try:
        env_mean, nem, nzm, amp = envelope_mean(m, t, seq, ndir, N, N_dim)
        diff = np.abs(nzm - nem)

        if np.all(diff > 1):    # check if absolute difference b/w zero-crossings are greater than 1
            stp = 0
            counter = 0

        else:
            counter += 1
            stp = counter >= stp_cnt

    except:     # return default values
        env_mean = np.zeros((N, N_dim))
        stp = True

    return stp, env_mean, counter


def e_diff(prev_imf, curr_imf, t, seq, ndir, N, N_dim, threshold):  # stopping criterion based on energy difference
    try:
        env_mean, nem, nzm, amp = envelope_mean(curr_imf, t, seq, ndir, N, N_dim)
        prev_e = np.sum(np.abs(prev_imf) ** 2)
        curr_e = np.sum(np.abs(curr_imf) ** 2)
        difference = np.abs(curr_e - prev_e)

        if prev_e > 0:
            relative_diff = difference / prev_e
        else:
            relative_diff = difference

        stp = relative_diff < threshold

    except Exception as e:
        print(f"Error during e_diff computation: {e}")
        env_mean = np.zeros((N, N_dim))
        stp = True

    return stp, env_mean


def zero_crossings(x):
    """
        Identifies zero crossings in the input signal.

        A zero crossing occurs when the signal changes sign. This function finds the indices where this occurs.
        It also handles cases with exact zeros, including blocks of consecutive zeros, by using their midpoint.

        Parameters:
          x (numpy.ndarray): The input signal.

        Returns:
          numpy.ndarray: Indices corresponding to zero crossings.
    """
    izc_detect = np.where(np.diff(np.sign(x)))[0]

    if len(izc_detect) == 0:  # early exit if no zero crossing found
        return izc_detect

    exact_zeros = np.where(x == 0)[0]   # find and store exact zeros

    if len(exact_zeros) > 0:        # Check for consecutive zeros
        if np.any(np.diff(exact_zeros) == 1):   # consecutive zeros
            zero_array = (x == 0).astype(int)
            diff = np.diff(np.concatenate(([0], zero_array, [0])))
            start_block = np.where(diff == 1)[0]
            end_block = np.where(diff == -1)[0] - 1
            midpts = np.round((start_block + end_block) / 2).astype(int)

        else:
            midpts = exact_zeros    # non-consecutive zeros

        izc_detect = np.unique(np.concatenate((izc_detect, midpts)))    # midpoint + zc-index

    return izc_detect


def local_peaks(signal):
    """
        Determines the indices of local minima and maxima in a signal.

        The function computes the derivative (using sign changes) of the signal to identify
        points where the signal transitions from increasing to decreasing (local maximum) or vice versa.

        Parameters:
          signal (numpy.ndarray): The input signal.

        Returns:
          tuple: (indmin, indmax)
            indmin (numpy.ndarray): Indices of local minima.
            indmax (numpy.ndarray): Indices of local maxima.
    """
    def peaks(signal):
        dX = np.sign(np.diff(signal.transpose())).transpose()
        locs_max = np.where((dX[:-1] > 0) & (dX[1:] < 0))[0] + 1
        return signal[locs_max], locs_max

    if np.all(signal < 1e-5):
        x = np.zeros((1, len(signal)))

    m = len(signal) - 1
    dy = np.diff(signal.transpose()).transpose()
    a = np.where(dy != 0)[0]
    lm = np.where(np.diff(a) != 1)[0] + 1
    d = a[lm] - a[lm - 1]
    a[lm] -= - np.floor(d / 2).astype(int)
    a = np.append(a, m)
    ya = signal[a]

    if len(ya) > 1:
        pks_max, loc_max = peaks(ya)
        pks_min, loc_min = peaks(-ya)
        indmin = a[loc_min] if len(pks_min) > 0 else np.array([])
        indmax = a[loc_max] if len(pks_max) > 0 else np.array([])

    else:
        indmin = np.array([])
        indmax = np.array([])

    return indmin, indmax


# defines new extrema points to extend the interpolations at the edges of the signal (mainly mirror symmetry)
def boundary_conditions(indmin, indmax, t, x, z, nbsym):
    """
        Computes new extrema points by extending boundaries via mirror symmetry.

        This function determines additional extrema at the signal boundaries to improve
        the reliability of spline interpolation. The mirror symmetry is applied by reflecting
        the extrema about the boundary points.

        Parameters:
          indmin (numpy.ndarray): Indices of local minima.
          indmax (numpy.ndarray): Indices of local maxima.
          t (numpy.ndarray): Time indices of the signal.
          x (numpy.ndarray): The projected signal.
          z (numpy.ndarray): The original mode signal.
          nbsym (int): Number of symmetric points to consider.

        Returns:
          tuple: (tmin, tmax, zmin, zmax, mode)
            tmin, tmax: Combined time indices for interpolating lower and upper envelopes.
            zmin, zmax: Interpolated envelope values (minima and maxima).
            mode (int): Indicator whether the signal has adequate extrema (1) or not (0).
    """
    lx = len(x) - 1
    end_max = len(indmax) - 1
    end_min = len(indmin) - 1
    indmin = indmin.astype(int)
    indmax = indmax.astype(int)

    if len(indmin) + len(indmax) < 3:
        return None, None, None, None, 0

    mode = 1  # the projected signal has inadequate extrema
    lsym, rsym = 0, lx

    # boundary conditions for interpolations :
    if indmax[0] < indmin[0]:
        if x[0] > x[indmin[0]]:
            lmax = np.flipud(indmax[1:min(end_max + 1, nbsym + 1)])
            lmin = np.flipud(indmin[:min(end_min + 1, nbsym)])
            lsym = indmax[0]

        else:
            lmax = np.flipud(indmax[:min(end_max + 1, nbsym)])
            lmin = np.concatenate((np.flipud(indmin[:min(end_min + 1, nbsym - 1)]), ([0])))
            lsym = 0

    else:
        if x[0] < x[indmax[0]]:
            lmax = np.flipud(indmax[:min(end_max + 1, nbsym)])
            lmin = np.flipud(indmin[1:min(end_min + 1, nbsym + 1)])
            lsym = indmin[0]

        else:
            lmin = np.flipud(indmin[:min(end_min + 1, nbsym)])
            lmax = np.concatenate((np.flipud(indmax[:min(end_max + 1, nbsym - 1)]), ([0])))
            lsym = 0

    if indmax[-1] < indmin[-1]:
        if x[-1] < x[indmax[-1]]:
            rmax = np.flipud(indmax[max(end_max - nbsym + 1, 0):])
            rmin = np.flipud(indmin[max(end_min - nbsym, 0):-1])
            rsym = indmin[-1]

        else:
            rmin = np.flipud(indmin[max(end_min - nbsym + 1, 0):])
            rmax = np.concatenate((np.array([lx]), np.flipud(indmax[max(end_max - nbsym + 2, 0):])))
            rsym = lx

    else:
        if x[-1] > x[indmin[-1]]:
            rmax = np.flipud(indmax[max(end_max - nbsym, 0):-1])
            rmin = np.flipud(indmin[max(end_min - nbsym + 1, 0):])
            rsym = indmax[-1]

        else:
            rmax = np.flipud(indmax[max(end_max - nbsym + 1, 0):])
            rmin = np.concatenate((np.array([lx]), np.flipud(indmin[max(end_min - nbsym + 2, 0):])))
            rsym = lx

    tlmin, tlmax = 2 * t[lsym] - t[lmin], 2 * t[lsym] - t[lmax]
    trmin, trmax = 2 * t[rsym] - t[rmin], 2 * t[rsym] - t[rmax]

    if tlmin[0] > t[0] or tlmax[0] > t[0]:    # in case symmetrized parts do not extend enough
        if lsym == indmax[0]:
            lmax = np.flipud(indmax[:min(end_max + 1, nbsym)])
        else:
            lmin = np.flipud(indmin[:min(end_min + 1, nbsym)])

        sys.exit('bug') if lsym == 1 else None

        lsym = 0
        tlmin = 2 * t[lsym] - t[lmin]
        tlmax = 2 * t[lsym] - t[lmax]

    if trmin[-1] < t[lx] or trmax[-1] < t[lx]:
        if rsym == indmax[-1]:
            rmax = np.flipud(indmax[max(end_max - nbsym + 1, 0):])
        else:
            rmin = np.flipud(indmin[max(end_min - nbsym + 1, 0):])

        sys.exit('bug') if rsym == lx else None

        rsym = lx
        trmin = 2 * t[rsym] - t[rmin]
        trmax = 2 * t[rsym] - t[rmax]

    zlmin, zlmax = z[lmin, :], z[lmax, :]
    zrmin, zrmax = z[rmin, :], z[rmax, :]
    tmin, tmax = np.hstack((tlmin, t[indmin], trmin)), np.hstack((tlmax, t[indmax], trmax))
    zmin, zmax = np.vstack((zlmin, z[indmin, :], zrmin)), np.vstack((zlmax, z[indmax, :], zrmax))

    return (tmin, tmax, zmin, zmax, mode)


# computes the mean of the envelopes and the mode amplitude estimate
def envelope_mean(m, t, seq, ndir, N, N_dim):  # new
    """
        Computes the mean envelope and amplitude estimate for a mode.

        For each direction defined in the Hammersley sequence, this function:
          - Projects the mode signal m onto a unit vector.
          - Identifies local extrema in the projection.
          - Applies boundary conditions to extend the extrema for robust spline interpolation.
          - Computes the upper and lower envelopes via cubic splines.
          - Accumulates the averaged envelope (mean) and amplitude measure across directions.

        Parameters:
          m (numpy.ndarray): Mode signal.
          t (numpy.ndarray): Time indices.
          seq (numpy.ndarray): Hammersley sequence.
          ndir (int): Number of directions.
          N (int): Number of samples.
          N_dim (int): Number of dimensions/channels.

        Returns:
          tuple: (env_mean, nem, nzm, amp)
            env_mean (numpy.ndarray): Mean envelope computed from all directions.
            nem (numpy.ndarray): Number of extrema detected per direction.
            nzm (numpy.ndarray): Number of zero crossings per direction.
            amp (numpy.ndarray): Amplitude estimation based on the envelope differences.
    """
    NBSYM = 2
    count = 0

    env_mean = np.zeros((len(t), N_dim))
    amp = np.zeros((len(t)))
    nem = np.zeros((ndir))
    nzm = np.zeros((ndir))
    dir_vec = np.zeros((N_dim, 1))

    for it in range(ndir):
        if N_dim != 3:  # Multivariate signal (for N_dim ~=3) with hammersley sequence
            b = 2 * seq[it, :] - 1  # Linear normalisation of hammersley [-1,1]
            # Find angles corresponding to the normalised sequence
            tht = np.arctan2(np.sqrt(np.cumsum(b[:0:-1][::-1] ** 2))[::-1], b[:N_dim - 1])
            # Find coordinates of unit direction vectors on n-sphere
            dir_vec[:N_dim - 1, 0] = np.cos(tht) * np.cumprod(np.concatenate(([1], np.sin(tht)[:-1])))
            dir_vec[-1, 0] = np.prod(np.sin(tht))

        else:  # Trivariate signal with hammersley sequence
            # tt = 2 * seq[it, 0] - 1  # Linear normalisation of hammersley [-1,1]
            # tt = np.clip(tt, -1, 1)
            tt = np.clip(2 * seq[it, 0] - 1, -1, 1)
            phirad = seq[it, 1] * 2 * pi  # Normalize angle from 0 - 2*pi
            st = sqrt(1.0 - tt ** 2)
            dir_vec[0] = st * cos(phirad)
            dir_vec[1] = st * sin(phirad)
            dir_vec[2] = tt

        # Projection of input signal on nth (out of total ndir) direction vectors
        y = np.dot(m, dir_vec)

        # Calculates the extrema of the projected signal
        indmin, indmax = local_peaks(y)

        nem[it] = len(indmin) + len(indmax)
        indzer = zero_crossings(y)
        nzm[it] = len(indzer)

        tmin, tmax, zmin, zmax, mode = boundary_conditions(indmin, indmax, t, y, m, NBSYM)

        # Calculate multidimensional envelopes using spline interpolation
        # Only done if number of extrema of the projected signal exceed 3
        if mode:
            fmin = CubicSpline(tmin, zmin, bc_type='not-a-knot')
            env_min = fmin(t)
            fmax = CubicSpline(tmax, zmax, bc_type='not-a-knot')
            env_max = fmax(t)
            amp = amp + np.sqrt(np.sum((env_max - env_min)**2, axis=1)) / 2
            env_mean += (env_max + env_min) / 2
        else:  # if the projected signal has inadequate extrema
            count += 1

    if ndir > count:
        env_mean /= (ndir - count)
        amp /= (ndir - count)
    else:
        env_mean = np.zeros((N, N_dim))
        amp = np.zeros((N))
        nem = np.zeros((ndir))
        # env_mean.fill(0)
        # amp.fill(0)
        # nem.fill(0)
    return env_mean, nem, nzm, amp


def stop_emd(r, seq, ndir, N_dim):
    """
        Checks the stopping criterion for the empirical mode decomposition (EMD).

        Projects the residual signal r onto each direction defined by the Hammersley sequence
        and counts the extrema. If all projections yield less than 3 extrema, the EMD process stops.

        Parameters:
          r (numpy.ndarray): Residual signal.
          seq (numpy.ndarray): Hammersley sequence for direction selection.
          ndir (int): Number of directional projections.
          N_dim (int): Dimensionality (number of channels) of the signal.

        Returns:
          bool: True if the stopping condition is met (i.e., fewer than 3 extrema per projection), else False.
    """
    ner = np.zeros((ndir, 1))
    dir_vec = np.zeros((N_dim, 1))

    for it in range(ndir):

        if N_dim != 3:  # Multivariate signal (for N_dim ~=3) with hammersley sequence
            b = 2 * seq[it, :] - 1  # Linear normalisation of hammersley [-1,1]
            # Find angles corresponding to the normalised sequence
            tht = np.arctan2(np.sqrt(np.cumsum(b[:0:-1][::-1] ** 2))[::-1], b[:N_dim - 1])

            # Find coordinates of unit direction vectors on n-sphere
            dir_vec[:N_dim - 1, 0] = np.cos(tht) * np.cumprod(np.concatenate(([1], np.sin(tht)[:-1])))
            dir_vec[-1, 0] = np.prod(np.sin(tht))

        else:  # Trivariate signal with hammersley sequence
            tt = 2 * seq[it, 0] - 1     # Linear normalisation of hammersley [-1,1]
            tt = np.clip(tt, -1, 1)
            phirad = seq[it, 1] * 2 * pi  # Normalize angle from 0 - 2*pi
            st = sqrt(1.0 - tt ** 2)
            dir_vec[0] = st * cos(phirad)
            dir_vec[1] = st * sin(phirad)
            dir_vec[2] = tt

        y = np.dot(r, dir_vec)
        indmin, indmax = local_peaks(y)     # Calculates the extrema of the projected signal
        ner[it] = len(indmin) + len(indmax)

    stp = all(ner < 3)    # Stops if the all projected signals have less than 3 extrema

    return stp