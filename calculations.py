# Hammersley a prime calculations
import numpy as np
from scipy.interpolate import interp1d, CubicSpline


def hamm(n_dir, base):
    seq = np.zeros(n_dir)
    if base > 1:
        seed = np.arange(1, n_dir + 1)
        base_inv = 1 / base
        while np.any(seed != 0):
            digit = np.remainder(seed, base)
            seq += digit * base_inv
            base_inv /= base
            seed = np.floor(seed / base)
    else:
        temp = np.arange(1, n_dir + 1)
        seq = (np.remainder(temp, -base + 1) + 0.5) / -base
    return seq


def nth_prime(nth):  # Calculate and return the first nth primes.
    if nth > 0:
        if nth <= 5:    # call corner cases
            return small_primes(nth)
        else:
            return large_primes(nth)    # call general cases
    else:
        raise ValueError("n must be >= 1 for list of n prime numbers")


def small_primes(nth):      # corner cases for nth prime estimation using log
    list_prime = np.array([2, 3, 5, 7, 11])
    return list_prime[:nth]


def large_primes(nth):
    # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n/3035188#3035188
    lim = int(nth * (np.log(nth) + np.log(np.log(nth)))) + 1
    sieve = np.ones(lim // 3 + (lim % 6 == 2), dtype=bool)
    for i in range(1, int(lim ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def local_peaks(x):
    if np.all(x < 1e-5):
        return np.array([]), np.array([])

    dy = np.diff(x)
    a = np.where(dy != 0)[0]
    if len(a) == 0:
        return np.array([]), np.array([])

    lm = np.where(np.diff(a) != 1)[0] + 1
    d = a[lm] - a[lm - 1]
    a[lm] = a[lm] - d // 2
    a = np.append(a, len(x) - 1)
    ya = x[a]

    if len(ya) <= 1:
        return np.array([]), np.array([])

    dya = np.diff(ya)
    sign_change = np.sign(dya)

    loc_max = np.where((sign_change[:-1] > 0) & (sign_change[1:] < 0))[0] + 1
    loc_min = np.where((sign_change[:-1] < 0) & (sign_change[1:] > 0))[0] + 1

    indmax = a[loc_max] if len(loc_max) > 0 else np.array([])
    indmin = a[loc_min] if len(loc_min) > 0 else np.array([])

    return indmin, indmax


def stop_emd(r, seq, ndir, N_dim):
    ner = np.zeros(ndir)
    dir_vec = np.zeros(N_dim)

    for it in range(ndir):
        if N_dim != 3:  # Multivariate signal (for N_dim != 3) with Hammersley sequence
            # Linear normalization of Hammersley sequence in the range of -1.0 to 1.0
            b = 2 * seq[it, :] - 1

            # Find angles corresponding to the normalized sequence
            tht = np.arctan2(np.sqrt(np.flipud(np.cumsum(b[::-1][1:] ** 2))), b[:-1])

            # Find coordinates of unit direction vectors on n-sphere
            dir_vec = np.cumprod(np.concatenate(([1], np.sin(tht))))
            dir_vec[:-1] *= np.cos(tht)
        else:  # Trivariate signal with Hammersley sequence
            # Linear normalization of Hammersley sequence in the range of -1.0 to 1.0
            tt = 2 * seq[it, 0] - 1
            tt = np.clip(tt, -1, 1)

            # Normalize angle from 0 to 2*pi
            phirad = seq[it, 1] * 2 * np.pi
            st = np.sqrt(1.0 - tt * tt)

            dir_vec[0] = st * np.cos(phirad)
            dir_vec[1] = st * np.sin(phirad)
            dir_vec[2] = tt

        # Projection of input signal on nth (out of total ndir) direction vectors
        y = np.dot(r.T, dir_vec)

        # Calculates the extrema of the projected signal
        indmin, indmax = local_peaks(y)

        ner[it] = len(indmin) + len(indmax)

    # Stops if all projected signals have less than 3 extrema
    stp = np.all(ner < 3)
    return stp


def zero_crossings(x):
    indzer = np.where(x[:-1] * x[1:] < 0)[0]

    if np.any(x == 0):
        iz = np.where(x == 0)[0]
        if np.any(np.diff(iz) == 1):
            zer = x == 0
            dz = np.diff(np.concatenate(([0], zer, [0])))
            debz = np.where(dz == 1)[0]
            finz = np.where(dz == -1)[0] - 1
            indz = np.round((debz + finz) / 2).astype(int)
        else:
            indz = iz
        indzer = np.sort(np.concatenate((indzer, indz)))

    return indzer


def stop(m, t, sd, sd2, tol, seq, ndir, N, N_dim):
    try:
        env_mean, nem, nzm, amp = envelope_mean(m, t, seq, ndir, N, N_dim)
        sx = np.sqrt(np.sum(np.power(env_mean, 2), axis=1))

        if np.any(amp):
            sx = sx / amp

        if np.mean(sx > sd) <= tol and not np.any(sx > sd2) and np.any(nem > 2):
            stp = 1
        else:
            stp = 0
    except Exception as e:
        print(f"Exception in stop function: {e}")
        env_mean = np.zeros((N, N_dim))
        stp = 1

    return stp, env_mean


def fix(m, t, seq, ndir, stp_cnt, counter, N, N_dim):
    try:
        env_mean, nem, nzm, amp = envelope_mean(m, t, seq, ndir, N, N_dim)

        if np.all(np.abs(nzm - nem) > 1):
            stp = 0
            counter = 0
        else:
            counter += 1
            stp = (counter >= stp_cnt)
    except Exception as e:
        print(f"Exception in fix function: {e}")
        env_mean = np.zeros((N, N_dim))
        stp = 1

    return stp, env_mean, counter


def envelope_mean(m, t, seq, ndir, N, N_dim):
    NBSYM = 2
    count = 0

    env_mean = np.zeros((N, N_dim))
    amp = np.zeros(N)
    nem = np.zeros(ndir)
    nzm = np.zeros(ndir)

    for it in range(ndir):
        if N_dim != 3:
            b = 2 * seq[it, :] - 1
            tht = np.arctan2(np.sqrt(np.flipud(np.cumsum(b[::-1][1:]**2))), b[:-1])
            dir_vec = np.cumprod(np.concatenate(([1], np.sin(tht))))
            dir_vec[:-1] *= np.cos(tht)
        else:
            tt = np.clip(2 * seq[it, 0] - 1, -1, 1)
            phirad = seq[it, 1] * 2 * np.pi
            st = np.sqrt(1.0 - tt * tt)
            dir_vec = np.array([st * np.cos(phirad), st * np.sin(phirad), tt])

        y = np.dot(m.T, dir_vec)

        indmin, indmax = local_peaks(y)

        nem[it] = len(indmin) + len(indmax)
        indzer = zero_crossings(y)
        nzm[it] = len(indzer)

        tmin, tmax, zmin, zmax, mode = boundary_conditions(indmin, indmax, t, y, m, NBSYM)

        if mode:
            env_min = CubicSpline(tmin, zmin, bc_type='not-a-knot')(t)
            env_max = CubicSpline(tmax, zmax, bc_type='not-a-knot')(t)
            amp += np.linalg.norm(env_max - env_min, axis=1) / 2
            env_mean += (env_max + env_min) / 2
        else:
            count += 1

    if ndir > count:
        env_mean /= (ndir - count)
        amp /= (ndir - count)
    else:
        env_mean.fill(0)
        amp.fill(0)
        nem.fill(0)
        nzm.fill(0)

    return env_mean, nem, nzm, amp


def boundary_conditions(indmin, indmax, t, x, z, nbsym):
    lx = len(x) - 1
    indmin = indmin.astype(int)
    indmax = indmax.astype(int)

    if len(indmin) + len(indmax) < 3:
        return None, None, None, None, 0

    mode = 1  # the projected signal has adequate extrema

    if indmax[0] < indmin[0]:
        if x[0] > x[indmin[0]]:
            lmax = indmax[1:min(len(indmax), nbsym + 1)][::-1]
            lmin = indmin[:min(len(indmin), nbsym)][::-1]
            lsym = indmax[0]
        else:
            lmax = indmax[:min(len(indmax), nbsym)][::-1]
            lmin = np.append(indmin[:min(len(indmin), nbsym - 1)][::-1], 0)
            lsym = 0
    else:
        if x[0] < x[indmax[0]]:
            lmax = indmax[:min(len(indmax), nbsym)][::-1]
            lmin = indmin[1:min(len(indmin), nbsym + 1)][::-1]
            lsym = indmin[0]
        else:
            lmax = np.append(indmax[:min(len(indmax), nbsym - 1)][::-1], 0)
            lmin = indmin[:min(len(indmin), nbsym)][::-1]
            lsym = 0

    if indmax[-1] < indmin[-1]:
        if x[-1] < x[indmax[-1]]:
            rmax = indmax[max(len(indmax) - nbsym, 0):][::-1]
            rmin = indmin[max(len(indmin) - nbsym, 0):-1][::-1]
            rsym = indmin[-1]
        else:
            rmax = np.append(lx, indmax[max(len(indmax) - nbsym + 1, 0):][::-1])
            rmin = indmin[max(len(indmin) - nbsym, 0):][::-1]
            rsym = lx
    else:
        if x[-1] > x[indmin[-1]]:
            rmax = indmax[max(len(indmax) - nbsym, 0):-1][::-1]
            rmin = indmin[max(len(indmin) - nbsym + 1, 0):][::-1]
            rsym = indmax[-1]
        else:
            rmax = indmax[max(len(indmax) - nbsym + 1, 0):][::-1]
            rmin = np.append(lx, indmin[max(len(indmin) - nbsym + 2, 0):][::-1])
            rsym = lx

    tlmin = 2 * t[lsym] - t[lmin]
    tlmax = 2 * t[lsym] - t[lmax]
    trmin = 2 * t[rsym] - t[rmin]
    trmax = 2 * t[rsym] - t[rmax]

    if tlmin[0] > t[0] or tlmax[0] > t[0]:
        if lsym == indmax[0]:
            lmax = indmax[:min(len(indmax), nbsym)][::-1]
        else:
            lmin = indmin[:min(len(indmin), nbsym)][::-1]
        lsym = 0
        tlmin = 2 * t[lsym] - t[lmin]
        tlmax = 2 * t[lsym] - t[lmax]

    if trmin[-1] < t[lx] or trmax[-1] < t[lx]:
        if rsym == indmax[-1]:
            rmax = indmax[max(len(indmax) - nbsym + 1, 0):][::-1]
        else:
            rmin = indmin[max(len(indmin) - nbsym + 1, 0):][::-1]
        rsym = lx
        trmin = 2 * t[rsym] - t[rmin]
        trmax = 2 * t[rsym] - t[rmax]

    zlmax = z[lmax, :]
    zlmin = z[lmin, :]
    zrmax = z[rmax, :]
    zrmin = z[rmin, :]

    tmin = np.hstack((tlmin, t[indmin], trmin))
    tmax = np.hstack((tlmax, t[indmax], trmax))
    zmin = np.vstack((zlmin, z[indmin, :], zrmin))
    zmax = np.vstack((zlmax, z[indmax, :], zrmax))

    return tmin, tmax, zmin, zmax, mode
