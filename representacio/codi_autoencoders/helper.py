import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import tensorflow as tf
import numpy as np

duration = 1024./16000.
rate = 16000
time = np.linspace(0,duration, num=rate*duration)

try:
    import IPython.display as display
except:
    pass


def plot_wave(audio, audio_gen):
   
    fig, ax = plt.subplots(figsize=(18,3))

    plt.subplot(1,3,1)
    plt.plot(time, audio)
    plt.xlim([0, duration])
    plt.title("Senyal Original")

    plt.subplot(1,3,2)
    plt.plot(time, audio_gen)
    plt.xlim([0, duration])
    plt.title("Senyal Generat")

    audio_noise = audio-audio_gen
    plt.subplot(1,3,3)
    plt.plot(time, audio_noise)
    plt.xlim([0, duration])
    plt.title("Soroll")

def mult_plot_wave(audio, audio_gen):
    fig, ax = plt.subplots(figsize=(24,4))
    plt.plot(time, audio, label='Original')
    plt.plot(time, audio_gen, label='Generat')
    plt.xlim([0, duration])
    legend = ax.legend(loc='best', shadow=True)
    plt.show()


def error_quadratic(original, generada):
    return np.mean(np.power(generada - original, 2))

def error_snr(original, generada):
    signal = np.mean(np.power(original, 2))
    noise = np.mean(np.power(generada - original, 2))
    return 20 * np.log10(signal/noise)

def best_results(originals, generades):
    i = 0
    aux = set()
    _dict = set()
    for orig, gen in zip(originals, generades):
        snr, qua = error_snr(orig, gen), error_quadratic(orig, gen)
        if not (snr, qua) in aux:
            _dict.add((snr, qua, i))
        aux.add((snr, qua))
        i+=1

    return sorted(_dict)

def best_results_quadratic(originals, generades):
    i = 0
    aux = set()
    _dict = set()
    for orig, gen in zip(originals, generades):
        snr, qua = error_snr(orig, gen), error_quadratic(orig, gen)
        if not (snr, qua) in aux:
            _dict.add((qua, snr, i))
        aux.add((snr, qua))
        i+=1

    return sorted(_dict)

def specgram(x,  NFFT=256, Fs=16000, Fc=None, detrend=mlab.detrend_none,
             window=mlab.window_hanning, noverlap=128,
             pad_to=None, sides='default', scale_by_freq=None, 
             mode='default', scale='default',**kwargs):
    
    if Fc is None:
        Fc = 0

    if mode == 'complex':
        raise ValueError('Cannot plot a complex specgram')

    if scale is None or scale == 'default':
        if mode in ['angle', 'phase']:
            scale = 'linear'
        else:
            scale = 'dB'
    elif mode in ['angle', 'phase'] and scale == 'dB':
        raise ValueError('Cannot use dB scale with angle or phase mode')

    spec, freqs, t = mlab.specgram(x=x, NFFT=NFFT, Fs=Fs,
                                   detrend=detrend, window=window,
                                   noverlap=noverlap, pad_to=pad_to,
                                   sides=sides, scale_by_freq=scale_by_freq,
                                   mode=mode)
    return spec, freqs, t

def show_specgram(subplt, spec, freqs, t, title = None, NFFT=256, Fs=16000,
            Fc=0, noverlap=128, cmap=None, xextent=None, 
            mode='default', scale='dB', vmin=None, vmax=None, colorbar = False, **kwargs ):

    if Fc is None:
        Fc = 0

    if mode == 'complex':
        raise ValueError('Cannot plot a complex specgram')

    if scale is None or scale == 'default':
        if mode in ['angle', 'phase']:
            scale = 'linear'
        else:
            scale = 'dB'
    elif mode in ['angle', 'phase'] and scale == 'dB':
        raise ValueError('Cannot use dB scale with angle or phase mode')
        
    if scale == 'linear':
        Z = spec
    elif scale == 'dB':
        if mode is None or mode == 'default' or mode == 'psd':
            Z = 10. * np.log10(spec)
        else:
            Z = 20. * np.log10(spec)
    else:
        raise ValueError('Unknown scale %s', scale)

    Z = np.flipud(Z)

    if xextent is None:
        # padding is needed for first and last segment:
        pad_xextent = (NFFT-noverlap) / (Fs / 2)
        xextent = np.min(t) - pad_xextent, np.max(t) + pad_xextent
        
    xmin, xmax = xextent
    freqs += Fc
    extent = xmin, xmax, freqs[0], freqs[-1]
    
    im = plt.imshow(Z, cmap, extent=extent, vmin=vmin, vmax=vmax, **kwargs)
    subplt.axis('auto')
    if colorbar:
        subplt.colorbar()
    subplt.xlabel("time")
    subplt.ylabel("frequency (hz)")
    subplt.xlim([xmin, xmax])
    subplt.ylim([freqs[0], freqs[-1]])
    if title:
        subplt.title(title)
        
def plot_3spec (audio, audio_gen, test):
    
    fig, ax = plt.subplots(figsize=(18,3))

    plt.subplot(1,3,1)
    show_specgram(plt, audio.reshape(129,7), test[1], test[2])
    plt.title("Espectrograma Original")

    plt.subplot(1,3,2)
    show_specgram(plt, audio_gen.reshape(129,7), test[1], test[2])
    plt.title("Espectrograma Generat")
    
    plt.subplot(1,3,3)
    plt.plot(time, test[3])
    plt.xlim([0, duration])
    plt.title("Senyal Original")
    plt.show()
    