import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

def getpinpu(audio):
    spectro = librosa.core.stft(audio, hop_length=256, n_fft=1022, center=True)
    spectro_mag, spectro_phase = librosa.core.magphase(spectro)
    spectro_mag = np.expand_dims(spectro_mag, axis=0)
    return spectro_mag, spectro_phase

def clip_audio(audio):
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

audio_path1 = '/data/mashuo/work/study/co-separation/dataset/music_dataset/solo/solo_audio_resample/violin/10RkjPIguAY_6.wav'
audio_path2 = '/data/mashuo/work/study/co-separation/dataset/music_dataset/solo/solo_audio_resample/erhu/37HdHAzJrOQ_11.wav'
audio1, audio_rate1 = librosa.load(audio_path1, sr=11025)
audio2, audio_rate2 = librosa.load(audio_path2, sr=11025)

audio_length = min(len(audio1), len(audio2))
audio1 = audio1[:audio_length]
audio2 = audio2[:audio_length]
audioMix = (audio1 + audio2)



spectro1, _ = getpinpu(audio1)
spectro2, _ = getpinpu(audio2)
spectroMix, _ = getpinpu(audioMix)
spectroMix = spectroMix+1e-10


f = librosa.core.stft(audio1)
print(f)
print(abs(f))
Xdb = librosa.amplitude_to_db(f)
plt.figure(figsize=(15, 15))
librosa.display.specshow(Xdb, sr=11025, x_axis='time', y_axis='hz')
plt.colorbar()
plt.savefig('/data/whs/5656.png')

gt_ratio_mask = spectro1 / spectroMix
gt_binary_mask = spectro1 >= spectro2
print(spectro1)
print(spectro2)
print(gt_binary_mask)

ratio_mask_value = sum((spectro1-(spectroMix * gt_ratio_mask)).flatten())
binary_mask_value = sum((spectro1-(spectroMix * gt_binary_mask)).flatten())
print('ratio_mask_value:'+str(ratio_mask_value))
print('binary_mask_value:'+str(binary_mask_value))
