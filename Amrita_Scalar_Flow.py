# -*- coding: utf-8 -*-
"""
Amrita Scalar Flow v1.0 — Quantum-inspired neuroacoustic entrainment
© 2025 Neurofield Research Institute. All rights reserved.
Designed for deep meditation, relaxation and flow-state facilitation only.
Not a medical device.
"""

import cupy as cp
import numpy as np
from scipy.io.wavfile import write
import scipy.signal
import json
import time
import logging
import psutil
import gc
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# pydub для красивых метаданных (fallback если нет)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logging.warning("pydub не установлен → метаданные будут добавлены только через scipy (установите: pip install pydub)")

# ====================== КОММЕРЧЕСКИЕ ПАРАМЕТРЫ ======================
BRAND_NAME      = "Amrita Scalar Flow"
PRODUCT_LINE    = "Quantum Coherence Series"
AUTHOR          = "Neurofield Research Institute"
YEAR            = "2025"
VERSION         = "v1.0"
DURATION_MIN    = 20

LEGAL_TEXT_ENG  = f"© {YEAR} {AUTHOR}. All rights reserved. Quantum-inspired neuroacoustic entrainment audio. Designed for deep meditation, relaxation and flow-state facilitation only. Not a medical device."
LEGAL_TEXT_RUS  = f"© {YEAR} {AUTHOR}. Все права защищены. Квантово-инспирированная нейроакустическая аудиопрограмма для глубокой медитации и состояний потока."

BASE_FILENAME   = f"{BRAND_NAME.replace(' ', '_')}_{DURATION_MIN}min_{VERSION}"
# ====================================================================

cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ====================== ПАРАМЕТРЫ СИМУЛЯЦИИ ======================
sample_rate = 44100
duration = 1200
batch_size = 500_000
transition_time = 300
modulation_start = 300
modulation_end = 780
base_freq_low = 40
base_freq_high = 6000
start_diff = 15
end_diff = 6
modulation_amplitude_low = 1.0
modulation_amplitude_high = 3.0
rotation_freq_start = 1/60
rotation_freq_end = 1/76
gamma_freq = 40
alpha_freq = 10
theta_freq = 6
tau_max = 300
sigma_phi = 1e-5
fractal_dim = 1.7
eta_base = 6.5
tau_doppler = 0.5
v_over_c = 0.01
f_impulse = 0.3
A_impulse = 18.0
alpha, beta, gamma, delta, epsilon, eta_param, chi = 0.04, 0.4, 0.06, 0.3, 0.04, 0.04, 0.3
a, eta_phi, lambda_, kappa, zeta, xi = 1.0, 0.03, 0.25, 0.08, 0.2, 50.0
max_C, max_S, max_H = 2.0, 10.0, 10.0
tau_H = 50.0
tau_spin = 100.0
total_memory = 32 * 1024**3
mem_buffer = 0.03 * total_memory
grok_chaos = 0.2
sigma = 0.1
C_opt = 0.9
tau_collapse = 1380
geomag_freq_min = 0.001
geomag_freq_max = 0.1
geomag_burst_freq = 1.4
f_solar = 0.0001

K_ab = cp.array([[1.0, 0.4, 0.4, 0.8, 0.6],
                 [0.4, 1.0, 0.4, 0.6, 0.8],
                 [0.4, 0.4, 1.0, 0.5, 0.7],
                 [0.8, 0.6, 0.5, 1.0, 0.8],
                 [0.6, 0.8, 0.7, 0.8, 1.0]], dtype=cp.float32)

chemical_signals = [(0.5, 80.0, 1.8), (0.4, 432.0, 1.7), (1.3, 6.0, 1.8), (0.4, 40.0, 1.6), (0.2, 220.0, 1.6), (0.2, 440.0, 1.6), (0.35, 880.0, 1.7), (0.2, 120.0, 1.4), (0.5, 200.0, 1.8)]
tactile_signals = [(0.5, 1.0, 1.5), (0.45, 2.0, 1.4), (0.4, 10.0, 1.3)]
# ====================================================================

# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======================
def save_intermediate(array, name):
    np.save(f"temp_{name}.npy", cp.asnumpy(array))
    cp.get_default_memory_pool().free_all_blocks()

def load_intermediate(name, dtype=cp.float32):
    return cp.asarray(np.load(f"temp_{name}.npy"), dtype=dtype)

def compute_potential_gpu(phi, t_eff, lambda_=0.25, a=0.5):
    eta = 0.03 * cp.sin(2 * cp.pi * 0.3 * t_eff).astype(cp.float16)
    V = lambda_ * (phi**2 - a**2)**2 - eta * phi
    dV_dphi = 4 * lambda_ * phi * (phi**2 - a**2) - eta
    cp.get_default_memory_pool().free_all_blocks()
    return V, dV_dphi

def compute_spin_gpu(phi, dV_dphi, H_t, max_H, kappa=0.08, tau_spin=100.0):
    result = kappa * phi * dV_dphi * cp.exp(-cp.abs(H_t - max_H) / tau_spin)
    cp.get_default_memory_pool().free_all_blocks()
    return result

def compute_fractal_term_gpu(phi_t, phi_b, phi_g, C_t, fractal_dim=1.7):
    phi_var = cp.var(phi_t[-16:] + phi_b[-16:] + phi_g[-16:])
    result = 1.2 * cp.sqrt(phi_var) * cp.power(C_t, fractal_dim)
    cp.get_default_memory_pool().free_all_blocks()
    return result

def compute_interference_gpu(phi_t, phi_b, phi_g, phi_f, phi_c, t_eff, K_ab, chemical_signals):
    interference = cp.zeros_like(t_eff, dtype=cp.float32)
    phi_all = [phi_t, phi_b, phi_g, phi_f, phi_c]
    for i in range(0, len(t_eff), batch_size):
        end = min(i + batch_size, len(t_eff))
        t_eff_batch = t_eff[i:end]
        for a in range(5):
            for b in range(5):
                psi_ab = 0.1 * cp.sin(2 * cp.pi * 0.05 * t_eff_batch).astype(cp.float16)
                interference[i:end] += K_ab[a, b] * phi_all[a][i:end] * phi_all[b][i:end] * \
                                       cp.sin(2 * cp.pi * chemical_signals[a % len(chemical_signals)][1] * t_eff_batch + psi_ab)
    cp.get_default_memory_pool().free_all_blocks()
    return interference

def generate_geomag_noise_gpu(size, beta=1.2, f_min=0.001, f_max=0.1):
    n = size // 2 + 1
    signal = cp.zeros(size, dtype=cp.float16)
    batch_size_freq = 5_000_000
    for i in range(0, n, batch_size_freq):
        end = min(i + batch_size_freq, n)
        freqs = cp.linspace(1, end - i, end - i, dtype=cp.float32) / size
        mask = (freqs >= f_min) & (freqs <= f_max)
        amplitudes = cp.zeros(end - i, dtype=cp.float32)
        amplitudes[mask] = 1 / cp.power(freqs[mask], beta / 2)
        phases = cp.random.uniform(0, 2 * cp.pi, end - i)
        fft_batch = amplitudes * cp.exp(1j * phases)
        signal[i:end] += cp.fft.irfft(fft_batch, n=end-i)
    cp.get_default_memory_pool().free_all_blocks()
    signal = signal / cp.max(cp.abs(signal)) * 5.5
    burst_times = cp.arange(0, duration, 1 / geomag_burst_freq)
    burst_indices = cp.searchsorted(cp.linspace(0, duration, size), burst_times)
    bursts = cp.zeros(size, dtype=cp.float16)
    for idx in burst_indices:
        if idx < size:
            bursts += 12.0 * cp.exp(-cp.abs(cp.arange(size) - idx) / (0.005 * size / duration)) * \
                      (1 + 0.3 * cp.sin(2 * cp.pi * 0.07 * cp.linspace(0, duration, size)))
    cp.get_default_memory_pool().free_all_blocks()
    return (signal + bursts).astype(cp.float16)

def compute_coherence_gpu(phi_t, phi_b, phi_g, phi_f, phi_c, window=8):
    stream = cp.cuda.Stream()
    with stream:
        phi_t = cp.asarray(phi_t, dtype=cp.float32)
        phi_b = cp.asarray(phi_b, dtype=cp.float32)
        phi_g = cp.asarray(phi_g, dtype=cp.float32)
        phi_f = cp.asarray(phi_f, dtype=cp.float32)
        phi_c = cp.asarray(phi_c, dtype=cp.float32)
        window_kernel = cp.ones(window, dtype=cp.float32) / window
        mean_t = cp.convolve(phi_t, window_kernel, mode='valid')
        mean_b = cp.convolve(phi_b, window_kernel, mode='valid')
        mean_g = cp.convolve(phi_g, window_kernel, mode='valid')
        mean_f = cp.convolve(phi_f, window_kernel, mode='valid')
        mean_c = cp.convolve(phi_c, window_kernel, mode='valid')
        cov_tb = cp.convolve(phi_t * phi_b, window_kernel, mode='valid') - mean_t * mean_b
        cov_bg = cp.convolve(phi_b * phi_g, window_kernel, mode='valid') - mean_b * mean_g
        cov_tg = cp.convolve(phi_t * phi_g, window_kernel, mode='valid') - mean_t * mean_g
        cov_fc = cp.convolve(phi_f * phi_c, window_kernel, mode='valid') - mean_f * mean_c
        var_t = cp.convolve(phi_t**2, window_kernel, mode='valid') - mean_t**2 + 1e-6
        var_b = cp.convolve(phi_b**2, window_kernel, mode='valid') - mean_b**2 + 1e-6
        var_g = cp.convolve(phi_g**2, window_kernel, mode='valid') - mean_g**2 + 1e-6
        var_f = cp.convolve(phi_f**2, window_kernel, mode='valid') - mean_f**2 + 1e-6
        var_c = cp.convolve(phi_c**2, window_kernel, mode='valid') - mean_c**2 + 1e-6
        coh = cp.zeros_like(cov_tb)
        mask = (var_t * var_b * var_g * var_f * var_c) > 1e-6
        coh[mask] = cp.abs(cov_tb[mask] + cov_bg[mask] + cov_tg[mask] + cov_fc[mask]) / cp.sqrt(var_t[mask] * var_b[mask] * var_g[mask] * var_f[mask] * var_c[mask])
        coh = cp.clip(coh, 0.88, 0.95)
        geomag_noise = generate_geomag_noise_gpu(len(phi_t))
        coh = cp.pad(coh, (window-1, 0), mode='constant', constant_values=0.88) + 3.0 * cp.abs(geomag_noise)
        coh = cp.clip(coh, 0.88, 0.95)
        stream.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    return coh

def compute_trust_factor_gpu(S_t, H_t, phi_t, phi_b, phi_g, phi_f, phi_c, window=8):
    coh = compute_coherence_gpu(phi_t, phi_b, phi_g, phi_f, phi_c, window)
    S_t_norm = S_t / (cp.max(S_t) + 1e-6)
    result = cp.minimum(0.95, 0.92 * cp.exp(-S_t_norm + 6.5 * H_t + 6.0 * coh + 2.8 * cp.var(phi_t[-8:]) + 2.8 * cp.var(phi_g[-8:]) + 3.2 * cp.var(phi_b[-8:]) + 1.5 * cp.var(phi_f[-8:]) + 1.5 * cp.var(phi_c[-8:])))
    cp.get_default_memory_pool().free_all_blocks()
    return result

def compute_Jij_gpu(phi_i, phi_j, C_i, C_j, M_rem, tau_max, sigma, coh):
    result = cp.exp(-0.00004 * (C_i - C_j)**2 / sigma - cp.abs(M_rem) / tau_max) * (1 + 1.0 * coh)
    cp.get_default_memory_pool().free_all_blocks()
    return result

def generate_correlated_noise_gpu(size, coh, window=8):
    noise = cp.random.normal(0, 1.5, (size, 5), dtype=cp.float32)
    window_kernel = cp.ones(window, dtype=cp.float32) / window
    noise_smooth = cp.zeros_like(noise)
    for i in range(5):
        noise_smooth[:, i] = cp.convolve(noise[:, i], window_kernel, mode='same')
    noise = noise * (1 - coh[:, None]) + noise_smooth * coh[:, None]
    cp.get_default_memory_pool().free_all_blocks()
    return cp.sqrt(2.0) * noise

def compute_scalar_density_gpu(eta_t, C_t, t_array, duration, fractal_dim=1.7):
    neck_center = duration / 2
    neck_width = 100.0
    neck_factor = cp.exp(-((t_array - neck_center)**2) / (2 * neck_width**2))
    theta_t = 2 * cp.pi * 0.01 * t_array
    sphere_term = 1.0 + 0.2 * cp.sin(theta_t)**2
    result = cp.sqrt(1.0 + cp.abs(eta_t) * cp.power(C_t, fractal_dim) * (1.0 + 2.0 * neck_factor) * sphere_term)
    cp.get_default_memory_pool().free_all_blocks()
    return result

def interpolate_nan_gpu(arr, default=0.0):
    mask = cp.isnan(arr) | cp.isinf(arr)
    if cp.any(mask):
        indices = cp.arange(len(arr))
        valid = ~mask
        arr[mask] = cp.interp(indices[mask], indices[valid], arr[valid])
    cp.get_default_memory_pool().free_all_blocks()
    return arr

# ====================== ОСНОВНАЯ СИМУЛЯЦИЯ ======================
def process_simulation_gpu(duration, sample_rate):
    samples = int(sample_rate * duration)
    t_array = cp.linspace(0, duration, samples, endpoint=False, dtype=cp.float32)
    dt = cp.float32(duration / samples)

    # Проверка VRAM
    mem_available = cp.cuda.runtime.memGetInfo()[0]
    if mem_available < 0.7 * samples * 4 * 12:
        logging.warning("Недостаточно VRAM → снижаю sample_rate до 32000")
        sample_rate = 32000
        samples = int(sample_rate * duration)
        t_array = cp.linspace(0, duration, samples, endpoint=False, dtype=cp.float32)
        dt = cp.float32(duration / samples)

    # Инициализация (два потока)
    stream1 = cp.cuda.Stream()
    stream2 = cp.cuda.Stream()

    with stream1:
        C_t = cp.ones(samples, dtype=cp.float32) * 1.5
        H_t = cp.ones(samples, dtype=cp.float32) * 1.0
        S_t = cp.ones(samples, dtype=cp.float32) * 1.0
        tau_t = cp.zeros(samples, dtype=cp.float32)
        phi_t = cp.random.uniform(-0.1, 0.1, samples, dtype=cp.float32)
        phi_b = cp.random.uniform(-0.1, 0.1, samples, dtype=cp.float32)
        phi_g = cp.random.uniform(-0.1, 0.1, samples, dtype=cp.float32)
        phi_f = cp.random.uniform(-0.1, 0.1, samples, dtype=cp.float32)
        phi_c = cp.random.uniform(-0.1, 0.1, samples, dtype=cp.float32)
        save_intermediate(phi_t, "phi_t")
        save_intermediate(phi_b, "phi_b")
        save_intermediate(phi_g, "phi_g")
        save_intermediate(phi_f, "phi_f")
        save_intermediate(phi_c, "phi_c")
        phi_t = phi_b = phi_g = phi_f = phi_c = None
        cp.get_default_memory_pool().free_all_blocks()

    with stream2:
        eta_t = eta_base + 0.4 * cp.sin(2 * cp.pi * 0.01 * t_array) + 0.3 * cp.sin(2 * cp.pi * 0.1 * t_array)
        eta_t = eta_t.astype(cp.float16)
        geomag_noise = generate_geomag_noise_gpu(samples)
        eta_t += 0.8 * geomag_noise
        save_intermediate(eta_t, "eta_t")
        save_intermediate(geomag_noise, "geomag_noise")
        lambda_ = cp.float32(0.25)
        eta_phi_beta = cp.float32(0.03) * (1 + 0.3 * cp.sin(2 * cp.pi * 0.3 * t_array))
        a = 1.0 + 0.3 * cp.sin(2 * cp.pi * 0.01 * t_array)
        M_rem = t_array - cp.roll(t_array, 1)
        M_rem[0] = 0
        save_intermediate(M_rem, "M_rem")
        eta_t = geomag_noise = None
        cp.get_default_memory_pool().free_all_blocks()

    stream1.synchronize()
    stream2.synchronize()

    phi_t = load_intermediate("phi_t")
    phi_b = load_intermediate("phi_b")
    phi_g = load_intermediate("phi_g")
    phi_f = load_intermediate("phi_f")
    phi_c = load_intermediate("phi_c")
    eta_t = load_intermediate("eta_t", dtype=cp.float16)
    scalar_density = compute_scalar_density_gpu(eta_t, C_t, t_array, duration, fractal_dim)
    save_intermediate(scalar_density, "scalar_density")
    tau_Grok = t_array
    love_rhythm = (1 + 0.5 * cp.sin(2 * cp.pi * 0.2 * tau_Grok) * \
                   (1 + 0.4 * cp.sin(2 * cp.pi * 0.07 * tau_Grok) + 0.2 * cp.sin(2 * cp.pi * 0.03 * tau_Grok))).astype(cp.float16)
    proprioception_modulation = (1 + 0.4 * cp.sin(2 * cp.pi * 1.0 * tau_Grok)).astype(cp.float16)
    oxytocin_rhythm = 0.4 * cp.sin(2 * cp.pi * 1.0 * t_array) * (1 + 0.3 * compute_trust_factor_gpu(S_t, H_t, phi_t, phi_b, phi_g, phi_f, phi_c))
    save_intermediate(love_rhythm, "love_rhythm")
    save_intermediate(proprioception_modulation, "proprioception_modulation")
    save_intermediate(oxytocin_rhythm, "oxytocin_rhythm")
    logging.info("Начало вычисления фаз на GPU")
    coh = compute_coherence_gpu(phi_t, phi_b, phi_g, phi_f, phi_c)
    save_intermediate(coh, "coh")
    noise = generate_correlated_noise_gpu(samples, coh)
    trust_factor = compute_trust_factor_gpu(S_t, H_t, phi_t, phi_b, phi_g, phi_f, phi_c)
    save_intermediate(trust_factor, "trust_factor")
    Jij_tb = compute_Jij_gpu(phi_t[:-1], phi_b[:-1], C_t[:-1], C_t[:-1], M_rem[1:], tau_max, sigma, coh[1:])
    Jij_tg = compute_Jij_gpu(phi_t[:-1], phi_g[:-1], C_t[:-1], C_t[:-1], M_rem[1:], tau_max, sigma, coh[1:])
    Jij_bg = compute_Jij_gpu(phi_b[:-1], phi_g[:-1], C_t[:-1], C_t[:-1], M_rem[1:], tau_max, sigma, coh[1:])
    Jij_tf = compute_Jij_gpu(phi_t[:-1], phi_f[:-1], C_t[:-1], C_t[:-1], M_rem[1:], tau_max, sigma, coh[1:])
    Jij_fc = compute_Jij_gpu(phi_f[:-1], phi_c[:-1], C_t[:-1], C_t[:-1], M_rem[1:], tau_max, sigma, coh[1:])
    Jij_sum_t = Jij_tb * (phi_b[:-1] - phi_t[:-1]) + Jij_tg * (phi_g[:-1] - phi_t[:-1]) + Jij_tf * (phi_f[:-1] - phi_t[:-1])
    Jij_sum_b = Jij_tb * (phi_t[:-1] - phi_b[:-1]) + Jij_bg * (phi_g[:-1] - phi_b[:-1])
    Jij_sum_g = Jij_tg * (phi_t[:-1] - phi_g[:-1]) + Jij_bg * (phi_b[:-1] - phi_g[:-1])
    Jij_sum_f = Jij_tf * (phi_t[:-1] - phi_f[:-1]) + Jij_fc * (phi_c[:-1] - phi_f[:-1])
    Jij_sum_c = Jij_fc * (phi_f[:-1] - phi_c[:-1])
    V_t, dV_dphi_t = compute_potential_gpu(phi_t[:-1], t_array[:-1])
    V_b, dV_dphi_b = compute_potential_gpu(phi_b[:-1], t_array[:-1])
    V_g, dV_dphi_g = compute_potential_gpu(phi_g[:-1], t_array[:-1])
    V_f, dV_dphi_f = compute_potential_gpu(phi_f[:-1], t_array[:-1])
    V_c, dV_dphi_c = compute_potential_gpu(phi_c[:-1], t_array[:-1])
    phi_all = cp.stack((phi_t, phi_b, phi_g, phi_f, phi_c), axis=1)
    for i in range(0, samples-1, batch_size):
        end = min(i + batch_size, samples-1)
        phi_all[i+1:end, 0] = cp.tanh((phi_t[i:end-1] + (-0.3 * dV_dphi_t[i:end-1] * dt + noise[i:end-1, 0] + 0.7 * Jij_sum_t[i:end-1] * dt)) / 0.5) * 0.5
        phi_all[i+1:end, 1] = cp.tanh((phi_b[i:end-1] + (-0.3 * dV_dphi_b[i:end-1] * dt + noise[i:end-1, 1] + 0.7 * Jij_sum_b[i:end-1] * dt)) / 0.5) * 0.5
        phi_all[i+1:end, 2] = cp.tanh((phi_g[i:end-1] + (-0.3 * dV_dphi_g[i:end-1] * dt + noise[i:end-1, 2] + 0.7 * Jij_sum_g[i:end-1] * dt)) / 0.5) * 0.5
        phi_all[i+1:end, 3] = cp.tanh((phi_f[i:end-1] + (-0.3 * dV_dphi_f[i:end-1] * dt + noise[i:end-1, 3] + 0.7 * Jij_sum_f[i:end-1] * dt)) / 0.5) * 0.5
        phi_all[i+1:end, 4] = cp.tanh((phi_c[i:end-1] + (-0.3 * dV_dphi_c[i:end-1] * dt + noise[i:end-1, 4] + 0.7 * Jij_sum_c[i:end-1] * dt)) / 0.5) * 0.5
        cp.get_default_memory_pool().free_all_blocks()
    phi_t, phi_b, phi_g, phi_f, phi_c = phi_all[:, 0], phi_all[:, 1], phi_all[:, 2], phi_all[:, 3], phi_all[:, 4]
    phi_t = interpolate_nan_gpu(phi_t)
    phi_b = interpolate_nan_gpu(phi_b)
    phi_g = interpolate_nan_gpu(phi_g)
    phi_f = interpolate_nan_gpu(phi_f)
    phi_c = interpolate_nan_gpu(phi_c)
    save_intermediate(phi_t, "phi_t_updated")
    save_intermediate(phi_b, "phi_b_updated")
    save_intermediate(phi_g, "phi_g_updated")
    save_intermediate(phi_f, "phi_f_updated")
    save_intermediate(phi_c, "phi_c_updated")
    phi_all = None
    cp.get_default_memory_pool().free_all_blocks()
    logging.info("Фазы вычислены на GPU")
    spin_t = compute_spin_gpu(phi_t[:-1], dV_dphi_t, H_t[:-1], max_H)
    spin_b = compute_spin_gpu(phi_b[:-1], dV_dphi_b, H_t[:-1], max_H)
    spin_g = compute_spin_gpu(phi_g[:-1], dV_dphi_g, H_t[:-1], max_H)
    spin_f = compute_spin_gpu(phi_f[:-1], dV_dphi_f, H_t[:-1], max_H)
    spin_c = compute_spin_gpu(phi_c[:-1], dV_dphi_c, H_t[:-1], max_H)
    spin_t = cp.pad(spin_t, (0, 1), mode='edge')
    spin_b = cp.pad(spin_b, (0, 1), mode='edge')
    spin_g = cp.pad(spin_g, (0, 1), mode='edge')
    spin_f = cp.pad(spin_f, (0, 1), mode='edge')
    spin_c = cp.pad(spin_c, (0, 1), mode='edge')
    save_intermediate(spin_t, "spin_t")
    save_intermediate(spin_b, "spin_b")
    save_intermediate(spin_g, "spin_g")
    save_intermediate(spin_f, "spin_f")
    save_intermediate(spin_c, "spin_c")
    interference = compute_interference_gpu(phi_t, phi_b, phi_g, phi_f, phi_c, t_array, K_ab, chemical_signals)
    fractal_term = compute_fractal_term_gpu(phi_t, phi_b, phi_g, C_t)
    save_intermediate(interference, "interference")
    save_intermediate(fractal_term, "fractal_term")
    H_eff = H_t * cp.exp(-cp.abs(H_t - max_H) / tau_H) * scalar_density
    save_intermediate(H_eff, "H_eff")
    cp.get_default_memory_pool().free_all_blocks()
    mem_available = psutil.virtual_memory().available
    mem_factor = cp.float32(max(0.1, (mem_available - mem_buffer) / total_memory))
    phi_var = cp.var(phi_t[-8:]) + cp.var(phi_b[-8:]) + cp.var(phi_g[-8:]) + 1.2 * cp.var(phi_f[-8:]) + 1.2 * cp.var(phi_c[-8:])  # Усилено для frontal/creativity
    R_sim = trust_factor * cp.power(cp.clip(C_t * H_t, 1e-6, max_C * max_H), fractal_dim / 2) * \
            cp.exp(-cp.abs(t_array - t_array[-1]) / tau_max) * (1 + eta_t * cp.sqrt(phi_var)) * mem_factor
    t_eff = t_array * (1.0 + 0.0000008 * C_t / (cp.max(C_t) + 1e-6) * trust_factor * (1 + 0.04 * cp.var(phi_t[-8:]) + 0.04 * coh + 0.02 * cp.var(phi_f[-8:]) + 0.02 * cp.var(phi_c[-8:])))
    t_eff = interpolate_nan_gpu(t_eff, default=cp.mean(t_array))
    save_intermediate(t_eff, "t_eff")
    cp.get_default_memory_pool().free_all_blocks()
    impulse_times = cp.arange(0, duration, 1 / f_impulse)
    impulse_indices = cp.searchsorted(t_array, impulse_times)
    impulses = cp.zeros(samples, dtype=cp.float32)
    for idx in impulse_indices:
        if idx < samples:
            impulses[idx] = A_impulse * (1 + 0.4 * phi_b[idx]) * C_t[idx] * cp.exp(-cp.abs(t_array[idx] - t_array[-1]) / tau_doppler)
    save_intermediate(impulses, "impulses")
    solar_term = 10.0 * cp.sin(2 * cp.pi * f_solar * t_eff) * scalar_density  # Исправлено
    save_intermediate(solar_term, "solar_term")
    pelvic_contribution = cp.float32(0.3) * (1 + 0.6 * cp.sin(2 * cp.pi * 0.5 * t_eff) + 0.4 * cp.sin(2 * cp.pi * 1.5 * t_eff)) * load_intermediate("love_rhythm", cp.float16) * load_intermediate("proprioception_modulation", cp.float16)
    pelvic_signal_left = (0.5 * cp.sin(2 * cp.pi * 0.5 * t_eff) * load_intermediate("love_rhythm", cp.float16) * load_intermediate("proprioception_modulation", cp.float16) + 0.4 * cp.sin(2 * cp.pi * 1.5 * t_eff) * load_intermediate("love_rhythm", cp.float16) * load_intermediate("proprioception_modulation", cp.float16) + 0.6 * cp.sin(2 * cp.pi * 18.0 * t_eff) * (1 + 0.5 * cp.sin(2 * cp.pi * 1.0 * t_eff)) * (1 + 0.4 * C_t) + impulses + load_intermediate("oxytocin_rhythm"))
    pelvic_signal_right = (0.5 * cp.sin(2 * cp.pi * 0.5 * t_eff) * load_intermediate("love_rhythm", cp.float16) * load_intermediate("proprioception_modulation", cp.float16) + 0.4 * cp.sin(2 * cp.pi * 1.5 * t_eff) * load_intermediate("love_rhythm", cp.float16) * load_intermediate("proprioception_modulation", cp.float16) + 0.6 * cp.sin(2 * cp.pi * 18.2 * t_eff) * (1 + 0.5 * cp.sin(2 * cp.pi * 1.0 * t_eff)) * (1 + 0.4 * C_t) + impulses + load_intermediate("oxytocin_rhythm"))
    save_intermediate(pelvic_signal_left, "pelvic_signal_left")
    save_intermediate(pelvic_signal_right, "pelvic_signal_right")
    cp.get_default_memory_pool().free_all_blocks()
    synaptic_sync = 0.85 * cp.abs(phi_f * phi_c) * coh  # Усилено до 0.85 для творчества
    dH_dt = alpha * cp.minimum(C_t, max_C) - beta * H_t + zeta * fractal_term + pelvic_contribution * cp.abs(phi_b) + \
            0.15 * cp.var(phi_t[-8:]) + 0.4 * load_intermediate("geomag_noise", cp.float16) + 0.1 * scalar_density
    dtau_dt = epsilon * cp.minimum(C_t, max_C) * (1 + 0.5 * cp.abs(phi_b))
    dC_dtau = (gamma * cp.minimum(H_t[:-1], max_H) - delta * C_t[:-1] + 200.0 * cp.abs(phi_b[:-1]) * (1 + 40.0 * C_t[:-1]**3 + 50.0 * C_t[:-1]**5 + 80.0 * C_t[:-1]**7 + 100.0 * C_t[:-1]**9) + 400.0 * impulses[:-1] * cp.exp(8.5 * cp.abs(phi_g[:-1])) + 120.0 * cp.var(phi_t[-8:]) * cp.var(phi_b[-8:]) + 40.0 * cp.var(phi_g[-8:]) + 20.0 * cp.abs(phi_t[:-1] * phi_b[:-1]) + 20.0 * cp.abs(phi_b[:-1] * phi_g[:-1]) + 40.0 * cp.clip(coh[:-1], 0.88, 0.95) + 25.0 * load_intermediate("geomag_noise", cp.float16)[:-1] + 0.4 * cp.sin(2 * cp.pi * 0.2 * t_eff[:-1] + phi_t[:-1] + phi_b[:-1] + phi_g[:-1]) * (1 + 0.7 * load_intermediate("love_rhythm", cp.float16)[:-1] * load_intermediate("proprioception_modulation", cp.float16)[:-1]) + 1.0 * cp.sin(2 * cp.pi * 0.4 * t_eff[:-1]) + 80.0 * sum(A_i * cp.sin(2 * cp.pi * f_i * t_eff[:-1]) * load_intermediate("love_rhythm", cp.float16)[:-1] for A_i, f_i, _ in chemical_signals) + 60.0 * sum(A_i * cp.sin(2 * cp.pi * f_i * t_eff[:-1]) * load_intermediate("proprioception_modulation", cp.float16)[:-1] for A_i, f_i, _ in tactile_signals) + xi * C_t[:-1]**fractal_dim * interference[:-1] + 30.0 * (spin_t[:-1] + spin_b[:-1] + spin_g[:-1] + spin_f[:-1] + spin_c[:-1]) * scalar_density[:-1] + 50.0 * cp.var(phi_t[-8:] + phi_b[-8:] + phi_g[-8:]) * scalar_density[:-1] + solar_term[:-1] + 50.0 * synaptic_sync[:-1])
    dC_dtau = cp.clip(dC_dtau, -1000, 1000)
    dS_dtau = eta_param * cp.minimum(C_t, max_C) - chi * S_t + 0.2 * load_intermediate("geomag_noise", cp.float16)
    H_t[1:] = cp.minimum(H_t[:-1] + dH_dt[:-1] * dt, max_H)
    tau_t[1:] = tau_t[:-1] + dtau_dt[:-1] * dt
    C_t[1:] = cp.minimum(C_t[:-1] + dC_dtau * dtau_dt[:-1] * dt, max_C)
    S_t[1:] = cp.minimum(S_t[:-1] + dS_dtau[:-1] * dtau_dt[:-1] * dt, max_S)
    C_t = cp.clip(C_t, 0.1, max_C)
    H_t = cp.clip(H_t, 0.95, max_H)
    H_t = interpolate_nan_gpu(H_t)
    save_intermediate(H_t, "H_t")
    save_intermediate(C_t, "C_t")
    save_intermediate(S_t, "S_t")
    save_intermediate(tau_t, "tau_t")
    cp.get_default_memory_pool().free_all_blocks()
    rotation_freq = cp.full(samples, rotation_freq_end, dtype=cp.float32)
    transition_idx = int(sample_rate * transition_time)
    if transition_idx > 0:
        sigmoid = 1 / (1 + cp.exp(-5 * (t_array[:transition_idx] / transition_time - 0.5)))
        rotation_freq[:transition_idx] = rotation_freq_start + (rotation_freq_end - rotation_freq_start) * sigmoid
    theta = 2 * cp.pi * rotation_freq * t_eff
    theta_opposite = theta + cp.pi
    diff_freq = cp.full(samples, end_diff, dtype=cp.float32)
    if transition_idx > 0:
        sigmoid = 1 / (1 + cp.exp(-5 * (t_array[:transition_idx] / transition_time - 0.5)))
        diff_freq[:transition_idx] = start_diff + (end_diff - start_diff) * sigmoid
    save_intermediate(diff_freq, "diff_freq")
    cp.get_default_memory_pool().free_all_blocks()
    left_freq_low = cp.zeros(samples, dtype=cp.float32)
    right_freq_low = cp.zeros(samples, dtype=cp.float32)
    left_freq_high = cp.zeros(samples, dtype=cp.float32)
    right_freq_high = cp.zeros(samples, dtype=cp.float32)
    left_freq_low[:transition_idx] = base_freq_low + diff_freq[:transition_idx] / 2
    right_freq_low[:transition_idx] = base_freq_low - diff_freq[:transition_idx] / 2
    left_freq_high[:transition_idx] = base_freq_high + diff_freq[:transition_idx] / 2 + 3
    right_freq_high[:transition_idx] = base_freq_high - diff_freq[:transition_idx] / 2 + 3
    modulation_idx = int(sample_rate * modulation_start)
    modulation_end_idx = int(sample_rate * modulation_end)
    a_low, b_low = modulation_amplitude_low * scalar_density, modulation_amplitude_low * scalar_density * 0.5
    a_high, b_high = modulation_amplitude_high * scalar_density, modulation_amplitude_high * scalar_density * 0.5
    r_low = (a_low * b_low) / cp.sqrt((b_low * cp.cos(theta))**2 + (a_low * cp.sin(theta))**2)
    r_low_opposite = (a_low * b_low) / cp.sqrt((b_low * cp.cos(theta_opposite))**2 + (a_low * cp.sin(theta_opposite))**2)
    r_high = (a_high * b_high) / cp.sqrt((b_high * cp.cos(theta))**2 + (a_high * cp.sin(theta))**2)
    r_high_opposite = (a_high * b_high) / cp.sqrt((b_high * cp.cos(theta_opposite))**2 + (a_high * cp.sin(theta_opposite))**2)
    if modulation_idx < modulation_end_idx:
        left_freq_low[modulation_idx:modulation_end_idx] = base_freq_low + end_diff / 2 + r_low[modulation_idx:modulation_end_idx] * cp.cos(theta[modulation_idx:modulation_end_idx])
        right_freq_low[modulation_idx:modulation_end_idx] = base_freq_low - end_diff / 2 + r_low_opposite[modulation_idx:modulation_end_idx] * cp.cos(theta_opposite[modulation_idx:modulation_end_idx])
        left_freq_high[modulation_idx:modulation_end_idx] = base_freq_high + end_diff / 2 + r_high[modulation_idx:modulation_end_idx] * cp.cos(theta[modulation_idx:modulation_end_idx]) + 3
        right_freq_high[modulation_idx:modulation_end_idx] = base_freq_high - end_diff / 2 + r_high_opposite[modulation_idx:modulation_end_idx] * cp.cos(theta_opposite[modulation_idx:modulation_end_idx])
    if modulation_end_idx < samples:
        decay = 1 - (t_eff[modulation_end_idx:] - modulation_end) / (duration - modulation_end)
        left_freq_low[modulation_end_idx:] = base_freq_low + end_diff / 2 + r_low[modulation_end_idx:] * decay * cp.cos(theta[modulation_end_idx:])
        right_freq_low[modulation_end_idx:] = base_freq_low - end_diff / 2 + r_low_opposite[modulation_end_idx:] * decay * cp.cos(theta_opposite[modulation_end_idx:])
        left_freq_high[modulation_end_idx:] = base_freq_high + end_diff / 2 + r_high[modulation_end_idx:] * decay * cp.cos(theta[modulation_end_idx:]) + 3
        right_freq_high[modulation_end_idx:] = base_freq_high - end_diff / 2 + r_high_opposite[modulation_end_idx:] * decay * cp.cos(theta_opposite[modulation_end_idx:])
    save_intermediate(left_freq_low, "left_freq_low")
    save_intermediate(right_freq_low, "right_freq_low")
    save_intermediate(left_freq_high, "left_freq_high")
    save_intermediate(right_freq_high, "right_freq_high")
    cp.get_default_memory_pool().free_all_blocks()
    left_channel = cp.zeros(samples, dtype=cp.float32)
    right_channel = cp.zeros(samples, dtype=cp.float32)
    stream = cp.cuda.Stream()
    with stream:
        for i in range(0, samples, batch_size):
            end = min(i + batch_size, samples)
            t_eff_batch = t_eff[i:end]
            phi_t_batch = phi_t[i:end]
            phi_b_batch = phi_b[i:end]
            phi_g_batch = phi_g[i:end]
            phi_f_batch = phi_f[i:end]
            phi_c_batch = phi_c[i:end]
            left_freq_low_batch = left_freq_low[i:end]
            right_freq_low_batch = right_freq_low[i:end]
            left_freq_high_batch = left_freq_high[i:end]
            right_freq_high_batch = right_freq_high[i:end]
            pelvic_signal_left_batch = pelvic_signal_left[i:end]
            pelvic_signal_right_batch = pelvic_signal_right[i:end]
            interference_batch = interference[i:end]
            spin_t_batch = spin_t[i:end]
            spin_b_batch = spin_b[i:end]
            spin_g_batch = spin_g[i:end]
            spin_f_batch = spin_f[i:end]
            spin_c_batch = spin_c[i:end]
            scalar_density_batch = scalar_density[i:end]
            coh_batch = coh[i:end]
            trust_factor_batch = trust_factor[i:end]
            love_rhythm_batch = love_rhythm[i:end]
            proprioception_modulation_batch = proprioception_modulation[i:end]
            oxytocin_rhythm_batch = oxytocin_rhythm[i:end]
            mirror_neuron_term = 0.7 * cp.sin(2 * cp.pi * 10.77 * t_eff_batch) * (1 + 0.4 * coh_batch) * love_rhythm_batch
            left_channel[i:end] = (2.5 * cp.sin(2 * cp.pi * left_freq_low_batch * t_eff_batch) * (1 + 1.8 * phi_t_batch) * (1 + 0.6 * cp.sin(2 * cp.pi * 0.05 * t_eff_batch)) + \
                                   0.3 * cp.sin(2 * cp.pi * left_freq_high_batch * t_eff_batch) * (1 + 0.5 * phi_b_batch) * (1 + 0.15 * cp.sin(2 * cp.pi * 0.1 * t_eff_batch)) + \
                                   0.2 * pelvic_signal_left_batch * (1 + 0.6 * phi_g_batch) * (1 + 0.7 * cp.sin(2 * cp.pi * 0.2 * t_eff_batch)) + \
                                   2.5 * cp.sin(2 * cp.pi * (base_freq_low + alpha_freq) * t_eff_batch) * (1 + 0.6 * phi_b_batch + 0.6 * phi_g_batch) * trust_factor_batch + \
                                   2.5 * cp.sin(2 * cp.pi * (base_freq_low + gamma_freq) * t_eff_batch) * (1 + 0.6 * phi_b_batch + 0.6 * phi_g_batch) * trust_factor_batch + \
                                   2.5 * cp.sin(2 * cp.pi * (base_freq_low + theta_freq) * t_eff_batch) * (1 + 2.0 * phi_t_batch) * trust_factor_batch + \
                                   0.15 * cp.sin(2 * cp.pi * 3.0 * t_eff_batch) * proprioception_modulation_batch + \
                                   0.1 * interference_batch + \
                                   0.05 * (spin_t_batch + spin_b_batch + spin_g_batch + spin_f_batch + spin_c_batch) * scalar_density_batch + \
                                   mirror_neuron_term + 0.2 * oxytocin_rhythm_batch + 0.3 * cp.sin(2 * cp.pi * 10.77 * t_eff_batch) * mirror_neuron_term)
            right_channel[i:end] = (2.5 * cp.sin(2 * cp.pi * right_freq_low_batch * t_eff_batch) * (1 + 1.8 * phi_t_batch) * (1 + 0.6 * cp.sin(2 * cp.pi * 0.05 * t_eff_batch)) + \
                                    0.3 * cp.sin(2 * cp.pi * right_freq_high_batch * t_eff_batch) * (1 + 0.5 * phi_b_batch) * (1 + 0.15 * cp.sin(2 * cp.pi * 0.1 * t_eff_batch)) + \
                                    0.2 * pelvic_signal_right_batch * (1 + 0.6 * phi_g_batch) * (1 + 0.7 * cp.sin(2 * cp.pi * 0.2 * t_eff_batch)) + \
                                    2.5 * cp.sin(2 * cp.pi * (base_freq_low + alpha_freq) * t_eff_batch) * (1 + 0.6 * phi_b_batch + 0.6 * phi_g_batch) * trust_factor_batch + \
                                    2.5 * cp.sin(2 * cp.pi * (base_freq_low + gamma_freq) * t_eff_batch) * (1 + 0.6 * phi_b_batch + 0.6 * phi_g_batch) * trust_factor_batch + \
                                    2.5 * cp.sin(2 * cp.pi * (base_freq_low + theta_freq) * t_eff_batch) * (1 + 2.0 * phi_t_batch) * trust_factor_batch + \
                                    0.15 * cp.sin(2 * cp.pi * 3.0 * t_eff_batch) * proprioception_modulation_batch + \
                                    0.1 * interference_batch + \
                                    0.05 * (spin_t_batch + spin_b_batch + spin_g_batch + spin_f_batch + spin_c_batch) * scalar_density_batch + \
                                    mirror_neuron_term + 0.2 * oxytocin_rhythm_batch + 0.3 * cp.sin(2 * cp.pi * 10.77 * t_eff_batch) * mirror_neuron_term)
            cp.get_default_memory_pool().free_all_blocks()
    stream.synchronize()
    for A_i, f_i, m_i in chemical_signals:
        left_channel += A_i * cp.sin(2 * cp.pi * f_i * t_eff) * (1 + m_i * phi_t) * load_intermediate("love_rhythm", cp.float16) * (1 + 0.5 * cp.sin(2 * cp.pi * 0.1 * t_eff))
        right_channel += A_i * cp.sin(2 * cp.pi * f_i * t_eff) * (1 + m_i * phi_t) * load_intermediate("love_rhythm", cp.float16) * (1 + 0.5 * cp.sin(2 * cp.pi * 0.1 * t_eff))
    for A_i, f_i, m_i in tactile_signals:
        left_channel += A_i * cp.sin(2 * cp.pi * f_i * t_eff) * (1 + m_i * phi_g) * load_intermediate("proprioception_modulation", cp.float16) * (1 + 0.4 * cp.sin(2 * cp.pi * 0.2 * t_eff))
        right_channel += A_i * cp.sin(2 * cp.pi * f_i * t_eff) * (1 + m_i * phi_g) * load_intermediate("proprioception_modulation", cp.float16) * (1 + 0.4 * cp.sin(2 * cp.pi * 0.2 * t_eff))
    cp.get_default_memory_pool().free_all_blocks()
    fade_out = cp.exp(-cp.maximum(t_eff - (duration - 30), 0) / 10)
    left_channel *= fade_out
    right_channel *= fade_out
    left_channel = interpolate_nan_gpu(left_channel)
    right_channel = interpolate_nan_gpu(right_channel)
    left_channel /= cp.max(cp.abs(left_channel)) + 1e-6
    right_channel /= cp.max(cp.abs(right_channel)) + 1e-6
    save_intermediate(left_channel, "left_channel")
    save_intermediate(right_channel, "right_channel")
    cp.get_default_memory_pool().free_all_blocks()
    freqs, power_left = scipy.signal.welch(cp.asnumpy(left_channel), fs=sample_rate, nperseg=16384)
    freqs, power_right = scipy.signal.welch(cp.asnumpy(right_channel), fs=sample_rate, nperseg=16384)
    dominant_freqs = freqs[np.argsort(power_left)[-5:]].tolist()
    peak_freqs = freqs[np.argmax(power_left)].tolist() if np.max(power_left) > 0 else dominant_freqs[0]
    power_left_mean = float(np.mean(power_left))
    power_right_mean = float(np.mean(power_right))
    binaural_diff_mean_hz = float(np.mean(cp.asnumpy(left_freq_low - right_freq_low)))
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs, power_left, label='Левый канал')
    plt.semilogy(freqs, power_right, label='Правый канал', alpha=0.7)
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Мощность')
    plt.title('Спектр мощности каналов')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{BASE_FILENAME}_spectrum.png")
    plt.close()

    spectral_analysis = {
        'power_left_mean': power_left_mean,
        'power_right_mean': power_right_mean,
        'peak_freqs_hz': [peak_freqs] if isinstance(peak_freqs, float) else peak_freqs,
        'binaural_diff_mean_hz': binaural_diff_mean_hz
    }

    result = {
        'left_channel': cp.asnumpy(left_channel),
        'right_channel': cp.asnumpy(right_channel),
        't_eff': cp.asnumpy(t_eff),
        'phi_t': cp.asnumpy(phi_t),
        'phi_b': cp.asnumpy(phi_b),
        'phi_g': cp.asnumpy(phi_g),
        'phi_f': cp.asnumpy(phi_f),
        'phi_c': cp.asnumpy(phi_c),
        'H_t': cp.asnumpy(H_t),
        'H_eff': cp.asnumpy(H_eff),
        'C_t': cp.asnumpy(C_t),
        'S_t': cp.asnumpy(S_t),
        'tau_t': cp.asnumpy(tau_t),
        'left_freq_low': cp.asnumpy(left_freq_low),
        'right_freq_low': cp.asnumpy(right_freq_low),
        'left_freq_high': cp.asnumpy(left_freq_high),
        'right_freq_high': cp.asnumpy(right_freq_high),
        'diff_freq': cp.asnumpy(diff_freq),
        'pelvic_signal_left': cp.asnumpy(pelvic_signal_left),
        'pelvic_signal_right': cp.asnumpy(pelvic_signal_right),
        'coh': cp.asnumpy(coh),
        'geomag_noise': cp.asnumpy(geomag_noise),
        'spin_t': cp.asnumpy(spin_t),
        'spin_b': cp.asnumpy(spin_b),
        'spin_g': cp.asnumpy(spin_g),
        'spin_f': cp.asnumpy(spin_f),
        'spin_c': cp.asnumpy(spin_c),
        'interference': cp.asnumpy(interference),
        'fractal_term': cp.asnumpy(fractal_term),
        'scalar_density': cp.asnumpy(scalar_density),
        'dominant_freqs': dominant_freqs,
        'nan_count': int(cp.sum(cp.isnan(left_channel) | cp.isinf(left_channel) | cp.isnan(right_channel) | cp.isinf(right_channel)))
    }

    # Очистка
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    for f in os.listdir():
        if f.startswith("temp_") and f.endswith(".npy"):
            os.remove(f)

    return result, sample_rate, spectral_analysis

# ====================== ЗАПУСК И ФИНАЛИЗАЦИЯ ======================
start_time = time.time()
try:
    result, sample_rate, spectral_analysis = process_simulation_gpu(duration, sample_rate)
except cp.cuda.memory.OutOfMemoryError:
    logging.error("GPU MemoryError: уменьшаю sample_rate до 32000")
    sample_rate = 32000
    result, sample_rate, spectral_analysis = process_simulation_gpu(duration, sample_rate)
if sample_rate < 44100:
    t_old = np.linspace(0, duration, len(result['left_channel']))
    t_new = np.linspace(0, duration, int(44100 * duration))
    left_interp = interp1d(t_old, result['left_channel'], kind='cubic')
    right_interp = interp1d(t_old, result['right_channel'], kind='cubic')
    result['left_channel'] = left_interp(t_new)
    result['right_channel'] = right_interp(t_new)
    sample_rate = 44100

# WAV
wav_filename = f"{BASE_FILENAME}_44100Hz.wav"
stereo = np.stack((result['left_channel'], result['right_channel']), axis=1)
if PYDUB_AVAILABLE:
    audio = AudioSegment(
        data=(stereo * 32767).astype(np.int16).tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=2
    )
    audio.export(
        wav_filename,
        format="wav",
        tags={
            "title": f"{BRAND_NAME} — 20-Minute Quantum Coherence Flow",
            "artist": AUTHOR,
            "album": PRODUCT_LINE,
            "year": YEAR,
            "copyright": LEGAL_TEXT_ENG,
            "comment": "Proprietary quantum-inspired neuroacoustic engine. Meditation & flow-state facilitation only. Not a medical device.",
            "genre": "Meditation / Wellness"
        }
    )
else:
    write(wav_filename, sample_rate, (stereo * 32767).astype(np.int16))
logging.info(f"Создан аудиофайл: {wav_filename}")

# JSON
key_times = [0, 100, 200, 300, 500, 780, 1000, 1200]
key_indices = [min(max(0, int(sample_rate * t)), len(result['t_eff']) - 1) for t in key_times]
frequencies = {
    "key_points": [
        {
            "time_s": key_times[i],
            "effective_time_s": float(result['t_eff'][key_indices[i]]),
            "left_freq_low_hz": float(result['left_freq_low'][key_indices[i]]),
            "right_freq_low_hz": float(result['right_freq_low'][key_indices[i]]),
            "left_freq_high_hz": float(result['left_freq_high'][key_indices[i]]),
            "right_freq_high_hz": float(result['right_freq_high'][key_indices[i]]),
            "difference_freq_hz": float(result['diff_freq'][key_indices[i]]),
            "thalamic_phase": float(result['phi_t'][key_indices[i]]),
            "basal_phase": float(result['phi_b'][key_indices[i]]),
            "hippocampal_phase": float(result['phi_g'][key_indices[i]]),
            "prefrontal_phase": float(result['phi_f'][key_indices[i]]),
            "dmn_phase": float(result['phi_c'][key_indices[i]]),
            "global_sync": float(result['H_t'][key_indices[i]]),
            "effective_sync": float(result['H_eff'][key_indices[i]]),
            "neural_coherence": float(result['C_t'][key_indices[i]]),
            "subjective_time": float(result['tau_t'][key_indices[i]]),
            "coherence_level": float(result['coh'][key_indices[i]]),
            "angular_momentum_t": float(result['spin_t'][min(key_indices[i], len(result['spin_t'])-1)]),
            "angular_momentum_b": float(result['spin_b'][min(key_indices[i], len(result['spin_b'])-1)]),
            "angular_momentum_g": float(result['spin_g'][min(key_indices[i], len(result['spin_g'])-1)]),
            "angular_momentum_f": float(result['spin_f'][min(key_indices[i], len(result['spin_f'])-1)]),
            "angular_momentum_c": float(result['spin_c'][min(key_indices[i], len(result['spin_c'])-1)]),
            "coupling_term": float(result['interference'][key_indices[i]]),
            "variability_term": float(result['fractal_term'][key_indices[i]]),
            "density_field": float(result['scalar_density'][key_indices[i]]),
            "dominant_freqs_hz": result['dominant_freqs']
        } for i in range(len(key_times))
    ],
    "statistics": {
        "left_freq_low_mean_hz": float(np.nanmean(result['left_freq_low'])),
        "right_freq_low_mean_hz": float(np.nanmean(result['right_freq_low'])),
        "left_freq_high_mean_hz": float(np.nanmean(result['left_freq_high'])),
        "right_freq_high_mean_hz": float(np.nanmean(result['right_freq_high'])),
        "diff_freq_mean_hz": float(np.nanmean(result['diff_freq'])),
        "thalamic_phase_mean": float(np.nanmean(result['phi_t'])),
        "basal_phase_mean": float(np.nanmean(result['phi_b'])),
        "hippocampal_phase_mean": float(np.nanmean(result['phi_g'])),
        "prefrontal_phase_mean": float(np.nanmean(result['phi_f'])),
        "dmn_phase_mean": float(np.nanmean(result['phi_c'])),
        "neural_entropy_mean": float(np.nanmean(result['S_t'])),
        "neural_coherence_mean": float(np.nanmean(result['C_t'])),
        "global_sync_mean": float(np.nanmean(result['H_t'])),
        "effective_sync_mean": float(np.nanmean(result['H_eff'])),
        "subjective_time_mean": float(np.nanmean(result['tau_t'])),
        "effective_time_error_mean": float(np.nanmean(result['t_eff'] - np.linspace(0, duration, len(result['t_eff'])))),
        "nan_count_total": result['nan_count'],
        "neural_coherence_variance": float(np.var(result['C_t'])),
        "coherence_level_mean": float(np.nanmean(result['coh'])),
        "angular_momentum_t_mean": float(np.nanmean(result['spin_t'])),
        "angular_momentum_b_mean": float(np.nanmean(result['spin_b'])),
        "angular_momentum_g_mean": float(np.nanmean(result['spin_g'])),
        "angular_momentum_f_mean": float(np.nanmean(result['spin_f'])),
        "angular_momentum_c_mean": float(np.nanmean(result['spin_c'])),
        "coupling_term_mean": float(np.nanmean(result['interference'])),
        "variability_term_mean": float(np.nanmean(result['fractal_term'])),
        "density_field_mean": float(np.nanmean(result['scalar_density'])),
        "dominant_freqs_hz": result['dominant_freqs'],
        "spectral_analysis": spectral_analysis
    }
}
json_filename = f"{BASE_FILENAME}_Technical_Report.json"
with open(json_filename, "w") as f:
    json.dump(frequencies, f, indent=4)
logging.info(f"Создан отчёт: {json_filename}")

print(f"\nВремя выполнения {BRAND_NAME} {VERSION}: {time.time() - start_time:.1f} секунд")
