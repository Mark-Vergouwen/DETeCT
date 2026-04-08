from __future__ import annotations

import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy.fft import rfft, rfftfreq # Fast Fourier Transforms 
from pathlib import Path


def closest_index(freqs, target):
    return int(np.argmin(np.abs(freqs - target)))

def safe_division(num, den, eps, default=0):
    """
    Protects ratios num/den if the denominator is too close to zero. 
    Defines such ratio as 0. 
    """
    if abs(den) < eps:
        return default
    else:
        return num / den

def compute_spectral_features(B: np.ndarray, k = 10) -> Dict[str, float]:
    """
    Extract spectral analysis and Fourier-based features from weekly consumption vector.
    All in one function as all features are based on the same FFT. 
    Inputs:
        B: flat array of 672 values (15-min x 7 days)
        k: an integer with the number of poles to extract as features, default is 10.  
    """

    features = {}

    n = B.size
    sample_spacing = 15 * 60                           # 15 minutes in seconds

    # Computing FFT 
    b_signal_fourier = rfft(B)                         # Computes the amplitudes 
    freqs = rfftfreq(n, d=sample_spacing)              # Gets the frequency poles of len(B) evenly spread datapoints with 15 minutes in between
    amplitude_spectrum = np.abs(b_signal_fourier)      # Amplitude_spectrum is sorted by increasing freqs, starting with 0 Hz (DC component, weekly average consumption)
    power_spectrum = amplitude_spectrum**2
    phase_spectrum = np.angle(b_signal_fourier)

    amplitude_spectrum_no_dc = amplitude_spectrum[1:]
    power_spectrum_no_dc = power_spectrum[1:]
    freqs_no_dc = freqs[1:]
    tot_energy_no_dc = np.sum(power_spectrum_no_dc)

    k = min(k, len(amplitude_spectrum) - 1)                # In case specified value is too large ; -1 to exclude pole 0 

    # Parameter for safe_division: 

    if tot_energy_no_dc > 0:
        max_power = np.max(power_spectrum_no_dc) 
    else:
       max_power = 1
       
    epsilon = 1e-9 * max_power  

    # Feature engineering

        # 0: DC component (N = 1 feature)
        # DC component: "baseline" over which harmonics are added
    features["fft_dc"] = amplitude_spectrum[0]

        # 1: Top k poles (N = 4k features)
        # Dominant frequencies: amp, freq, sin(phase), cos(phase)

    top_indices = np.argsort(-amplitude_spectrum[1:])[:k] + 1 # Extract top k indices (excluding DC at index 0) sorted by decreasing amplitude
    top_amplitudes = amplitude_spectrum[top_indices]
    top_freqs = freqs[top_indices]
    top_phases = phase_spectrum[top_indices]

    for i in range(len(top_indices)):
        features[f"top_amp_{i+1}"] = top_amplitudes[i]
        features[f"top_freq_{i+1}"] = top_freqs[i]
        features[f"top_sin_phi_{i+1}"] = np.sin(top_phases[i])
        features[f"top_cos_phi_{i+1}"] = np.cos(top_phases[i])

        # 2: Share of total energy in top k (N = 1 feature)
        # Energy concentration: periodic vs chaotic

    energy_top_k = np.sum(power_spectrum[top_indices]) 
    features["energy_top_k_ratio_noDC"] = safe_division(energy_top_k, tot_energy_no_dc, epsilon)

        # 3: Energy in bands (N = 6 features: 3 absolute + 3 ratios)
        # LF/MF/HF distribution: slow vs fast cycles

            # Computing useful frequencies

    freq_7day = 1/(7*24*60*60)
    freq_4day = 1/(4*24*60*60)
    freq_2day = 1/(2*24*60*60)
    freq_1day = 1/(24*60*60)
    freq_12h  = 1/(12*60*60)
    freq_6h = 1/(6*60*60)
    freq_15min = 1/(15*60)

            # Delimiting bands in the spectrum 

    id_freq_7day  = closest_index(freqs, freq_7day)
    id_freq_1day  = closest_index(freqs, freq_1day)
    id_freq_6h    = closest_index(freqs, freq_6h)
    id_freq_15min = closest_index(freqs, freq_15min)

            # Summing through squared amplitudes to get energy within bands 

    features["En_LF"] = np.sum(power_spectrum[id_freq_7day:id_freq_1day])
    features["En_MF"] = np.sum(power_spectrum[id_freq_1day:id_freq_6h])
    features["En_HF"] = np.sum(power_spectrum[id_freq_6h:id_freq_15min + 1])

            # Normalized 

    tot_energy_spectrum = np.sum(power_spectrum)
    features["ratio_En_LF_totnoDC"] = safe_division(features["En_LF"], tot_energy_no_dc, epsilon)
    features["ratio_En_MF_totnoDC"] = safe_division(features["En_MF"], tot_energy_no_dc, epsilon)
    features["ratio_En_HF_totnoDC"] = safe_division(features["En_HF"], tot_energy_no_dc, epsilon)

        # 4: Phase coherence + amplitude ratios (N = 11 features: 8 phases + 3 amps)
        # Harmonic relationships: independence vs coupling (redundant vs new info)

    phase_6h   = phase_spectrum[id_freq_6h]
    phase_12h  = phase_spectrum[closest_index(freqs, freq_12h)]
    phase_1day = phase_spectrum[id_freq_1day]
    phase_2day = phase_spectrum[closest_index(freqs, freq_2day)]
    phase_4day = phase_spectrum[closest_index(freqs, freq_4day)]

    delta_phi_6h_12h    = phase_6h - phase_12h
    delta_phi_12h_1day  = phase_12h - phase_1day
    delta_phi_1day_2day = phase_1day - phase_2day
    delta_phi_2day_4day = phase_2day - phase_4day

            # Encoded as sin and cos to avoid discontinuities

    features["phase_diff_6h_12h_sin"]    = np.sin(delta_phi_6h_12h)
    features["phase_diff_6h_12h_cos"]    = np.cos(delta_phi_6h_12h)

    features["phase_diff_12h_1day_sin"]  = np.sin(delta_phi_12h_1day)
    features["phase_diff_12h_1day_cos"]  = np.cos(delta_phi_12h_1day)

    features["phase_diff_1day_2day_sin"] = np.sin(delta_phi_1day_2day)
    features["phase_diff_1day_2day_cos"] = np.cos(delta_phi_1day_2day)

    features["phase_diff_2day_4day_sin"] = np.sin(delta_phi_2day_4day)
    features["phase_diff_2day_4day_cos"] = np.cos(delta_phi_2day_4day)

            # Similarly with amplitudes: 

    amp_24h = amplitude_spectrum[id_freq_1day]
    amp_12h = amplitude_spectrum[closest_index(freqs, freq_12h)]
    amp_2day = amplitude_spectrum[closest_index(freqs, freq_2day)]
    amp_4day = amplitude_spectrum[closest_index(freqs, freq_4day)]

    features["ratio_amp_12h_24h"] = safe_division(amp_12h, amp_24h, epsilon)
    features["ratio_amp_1d_2d"]   = safe_division(amp_24h, amp_2day, epsilon)
    features["ratio_amp_2d_4d"]   = safe_division(amp_2day, amp_4day, epsilon)

        # 5: Total energy (N = 1 feature)
        # Total spectral energy

    features["tot_energy_spec"] = tot_energy_spectrum           # Proportional to energy consumption

    if tot_energy_no_dc < 1e-12: # No variation: all spectral-shape features (w.r.t. non-DC) collapse to 0
        features["centroid_freq"] = 0.0
        features["spec_bw"] = 0.0
        features["spec_rolloff_85_freq"] = 0.0
        features["spec_entropy"] = 0.0
        features["spec_flatness"] = 0.0
        features["spec_crest_factor"] = 0.0
        features["spec_slope"] = 0.0

    else: # Compute them 
    
        # 6: Spectral centroid (N = 1 feature)
        # Center of gravity (in frequency terms) of the spectrum

        centroid_freq = np.dot(freqs_no_dc, power_spectrum_no_dc) / tot_energy_no_dc
        features["centroid_freq"] = centroid_freq

        # 7: Spectral bandwidth (N = 1 feature)
        # Spectral dispersion around centroid: concentrated vs diffuse

        bw = np.sqrt(np.dot((freqs_no_dc-centroid_freq)**2, power_spectrum_no_dc) / tot_energy_no_dc)
        features["spec_bw"] = bw

        # 8: Spectral roll-off at 85% (N = 1 feature)
        # Frequency threshold: is energy concentrated or not 

        cumulative_energy = np.cumsum(power_spectrum_no_dc)
        threshold_p85 = 0.85 * cumulative_energy[-1]
        index_p85 = min(np.searchsorted(cumulative_energy, threshold_p85), len(freqs_no_dc) - 1) # min and -1 to protect if weird signal 
        features["spec_rolloff_85_freq"] = freqs_no_dc[index_p85]

        # 9: Spectral entropy (N = 1 feature)
        # Disorder: structure vs noise in consumption

        freq_proba_array = power_spectrum_no_dc / tot_energy_no_dc
        mask = freq_proba_array > 0                                                 # Convention 0 * log(0) -> 0, otherwise: NaNs
        entropy = - np.dot(freq_proba_array[mask],np.log2(freq_proba_array[mask]))  # Log2 convention in signal processing (but any other base works)
        features["spec_entropy"] = entropy

        # 10: Spectral flatness (N = 1 feature)
        # Uniformity: tonal vs white noise

        mask = power_spectrum_no_dc > 0

        if np.sum(mask) > 0:
            geometric_mean = np.exp(np.mean(np.log(power_spectrum_no_dc[mask])))
            arithmetic_mean = np.mean(power_spectrum_no_dc[mask])
            features["spec_flatness"] = safe_division(geometric_mean, arithmetic_mean, epsilon)
        else:
            features["spec_flatness"] = 0

        # 11: Crest factor (N = 1 feature)
        # Peak/RMS ratio: pulses in the spectrum? 

        if tot_energy_no_dc > 0:
            features["spec_crest_factor"] = safe_division(np.max(amplitude_spectrum_no_dc), np.sqrt(np.mean(power_spectrum_no_dc)), epsilon)
        else:
            features["spec_crest_factor"] = 0

        # 12: Spectral slope (N = 1 feature)
        # Decay rate of the (global) spectrum on a log-log scale (exponant of frequency is power law)  
        
        x = np.log10(freqs_no_dc[1:])
        y = np.log10(power_spectrum_no_dc[1:] + epsilon)
        
        mask_xy = np.isfinite(x) & np.isfinite(y)
        
        if mask_xy.sum() < 2:
            features["spec_slope"] = np.nan
        elif np.std(y[mask_xy]) < 1e-8:
            features["spec_slope"] = 0.0  
        else: 
            spec_slope = np.polyfit(x[mask_xy], y[mask_xy], 1)[0]
            features["spec_slope"] = spec_slope
        
        # 13: Odd/Even harmonic ratio (N = 1 feature)
        # Compares energy in odd vs even harmonics of the daily frequency to describe daily cycle shape: 
        #   - ratio >> 1 : more "square-like" or asymmetric daily patterns (strong odd harmonics)
        #   - ratio around 1 : mixed harmonic structure, more complex shape
        #   - ratio << 1 : more symmetric patterns with strong even harmonics

    fund_freq = freq_1day
    odd_harmonics  = [closest_index(freqs, fund_freq * (2*i+1)) for i in range(3)] # Let's look at odd harmonics 1f, 3f, 5f close to 24h 
    even_harmonics = [closest_index(freqs, fund_freq * 2*(i+1)) for i in range(3)] # Let's look at the even ones: 2f, 4f, 6f
    odd_energy  = np.sum(power_spectrum[odd_harmonics]) 
    even_energy = np.sum(power_spectrum[even_harmonics])
    features["odd_even_harmonic_ratio"] = safe_division(odd_energy, even_energy, epsilon)

        # 14: Temporal periodicities (N = 1 feature)
        # Autocorrelation at 1-day.
        # Similarity of daily patterns (routine vs irregular days)

    B_centered = B - np.mean(B)
    acf = np.correlate(B_centered, B_centered, mode="full")[len(B)-1:]
    acf = acf / acf[0] if acf[0] != 0 else acf
    features["periodicity_1day"] = acf[24*4] if len(acf) > 24*4 else 0.0

    # Total: 4k + 29 features (with default k=10 -> 69 features)

    return features 

# --------------------------- Collecting features -----------------------------

"""
Workflow: 
    
    1) Convert dataframe  (df) to weekly vector (B): build_weekly_features_dataframe, ensure_week_vector
    2) Convert weekly vector to weekly vector (B) ánd numeric matrix (dm15): build_dm15, daytime_indices
    3) Apply feature engineering function: calc_fourier_features_consumption_py (D)
    4) Convert feature engineering dictionary pack to dataframe (features_df)
"""

EXPECTED_WEEK_LEN = 96 * 7  # 672 samples (15-min × 7 days)

def build_weekly_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply calc_features15_consumption_py to each household x week combination
    and return a DataFrame with features as columns. Change the function for 
    Fourier.
    """
    df = df.sort_values(["EAN_ID", "year", "week", "datetime"])
    results = []

    for (ean, year, week), group in df.groupby(["EAN_ID", "year", "week"]):
        B = group["afname_kwh"].to_numpy() 
        T = group["T"].to_numpy()
        N = group["N"].to_numpy()
        GHI = group["GHI"].to_numpy()
        P = group["P"].to_numpy()
        B = ensure_week_vector(B)
        D: Dict[str, float] = calc_fourier_features_consumption_py(B, T, N, GHI, P)
        D.update({"EAN_ID": ean, "year": year, "week": int(week)})
        results.append(D)

    features_df = pd.DataFrame(results)
    features_df = features_df.sort_values(["EAN_ID", "year", "week"]).reset_index(drop=True)
    return features_df

def ensure_week_vector(B: np.ndarray, expected_len: int = EXPECTED_WEEK_LEN) -> np.ndarray:
    """Ensure B is a numeric 1D array of length `expected_len` (pad/truncate)."""
    B = np.asarray(B, dtype=float).flatten()
    if B.size > expected_len:
        return B[:expected_len]
    elif B.size < expected_len:
        pad = np.full(expected_len - B.size, np.nan)
        return np.concatenate([B, pad])
    return B

def calc_fourier_features_consumption_py(B: np.ndarray, T: np.ndarray, N: np.ndarray, GHI: np.ndarray, P: np.ndarray) -> Dict[str, float]:
    
    """
    Generate features for 15-minute consumption over a week (length 672).
    Returns a dictionary of numeric features. 
    """

    dm15, B = build_dm15(B)
    idx = daytime_indices()

    D: Dict[str, float] = {}

    # ---- Fourier features ----
    D.update(compute_spectral_features(B, k=10)) 

    return D

def build_dm15(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape a 672-vector into a (96, 7) matrix with columns = days (Mon..Sun)."""
    B = ensure_week_vector(B)
    dm15 = B.reshape(96, 7, order="F")  # column-major like R
    return dm15, B

def daytime_indices() -> Dict[str, np.ndarray]:
    """Return index ranges for different dayparts (0-based)."""
    weekday = np.arange(0, 5 * 96)       # Mon-Fri
    weekend = np.arange(5 * 96, 7 * 96)  # Sat-Sun
    night = np.arange(0*4, 6*4)          # 00:00-06:00
    morning = np.arange(6*4, 10*4)       # 06:00-10:00
    noon = np.arange(10*4, 14*4)         # 10:00-14:00
    afternoon = np.arange(14*4, 18*4)    # 14:00-18:00
    evening = np.arange(18*4, 24*4)      # 18:00-24:00

    return dict(
        weekday=weekday, weekend=weekend,
        night=night, morning=morning, noon=noon,
        afternoon=afternoon, evening=evening
    )



def aggregate_fourierfeatures(df, agg):
 
    if agg == "yearly":
        
        feature_cols = [c for c in df.columns if c not in ['EAN_ID', 'week', 'moy', 'year']]
    
        collapsed_year = (
            df.groupby(['EAN_ID'])[feature_cols]
            .agg(['min', 'max', 'median', 'mean', 'std'])
            .reset_index()
        )
    
        collapsed_year.columns = [
            '_'.join(filter(None, col)).strip() for col in collapsed_year.columns.to_flat_index()
        ]
        
        return collapsed_year
    
    if agg == "monthly":
        
        moy_to_weeks = {
            1:  [1,2,3,4,5],
            2:  [6,7,8,9],
            3:  [10,11,12,13],
            4:  [14,15,16,17,18],
            5:  [19,20,21,22],
            6:  [23,24,25,26],
            7:  [27,28,29,30,31],
            8:  [32,33,34,35],
            9:  [36,37,38,39,40],
            10: [41,42,43,44],
            11: [45,46,47,48],
            12: [49,50,51,52]
        }

        week_to_moy = {w: m for m, weeks in moy_to_weeks.items() for w in weeks}
    
        df["moy"] = df["week"].map(week_to_moy)

        feature_cols = [c for c in df.columns if c not in ['EAN_ID', 'week', 'moy', 'year']]

        collapsed_month = (
            df.groupby(['EAN_ID', 'moy'])[feature_cols]
            .agg(['min', 'max', 'median', 'mean', 'std'])
            .reset_index()
        )

        collapsed_month.columns = [
            '_'.join(filter(None, col)).strip() for col in collapsed_month.columns.to_flat_index()
        ]
        
        return collapsed_month



