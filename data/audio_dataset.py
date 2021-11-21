import os.path
import librosa
import h5py
import random
from random import randrange
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.utils.data as torchdata
import json
import pickle


def generate_spectrogram_magphase(audio, stft_frame, stft_hop, with_phase=True):
    spectro = librosa.core.stft(audio, hop_length=stft_hop, n_fft=stft_frame, center=True)
    spectro_mag, spectro_phase = librosa.core.magphase(spectro)
    spectro_mag = np.expand_dims(spectro_mag, axis=0)
    if with_phase:
        spectro_phase = np.expand_dims(np.angle(spectro_phase), axis=0)
        return spectro_mag, spectro_phase
    else:
        return spectro_mag

def augment_audio(audio):
    audio = audio * (random.random() + 0.5)  # 0.5 - 1.5
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

def sample_audio(audio, window):
    # repeat if audio is too short
    if audio.shape[0] < window:
        n = int(window / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio_start = randrange(0, audio.shape[0] - window + 1)
    audio_sample = audio[audio_start:(audio_start+window)]
    return audio_sample

def augment_image(image):
	if(random.random() < 0.5):
		image = image.transpose(Image.FLIP_LEFT_RIGHT)
	enhancer = ImageEnhance.Brightness(image)
	image = enhancer.enhance(random.random()*0.6 + 0.7)
	enhancer = ImageEnhance.Color(image)
	image = enhancer.enhance(random.random()*0.6 + 0.7)
	return image



# 这里的声音加载方式不一样，感觉两篇论文都不一样，先这么加载，以后再更改。
class AudioVisual_audioset(torchdata.Dataset):
    def __init__(self, mode, opt):
        super(AudioVisual_audioset, self).__init__()
        self.mode = mode
        self.opt = opt
        self.imgSize = opt.image_size
        self.data_dir = opt.data_dir
        self.image_dir = opt.image_dir

        if self.mode == 'train':
            json_file = open('/data/whs/dataset/MUSIC-synthetic/synthetic/train_sy_path.json', 'r')
        else:
            json_file = open('/data/whs/dataset/MUSIC-synthetic/synthetic/test_sy_path.json', 'r')
            self.data_dir = '/data/whs/dataset/MUSIC-synthetic/synthetic/test1/'
            self.image_dir = '/data/whs/dataset/MUSIC-synthetic/synthetic/test1/video/'
        self.dict = json.load(json_file)
        self.wav_file = list(self.dict.keys())



        print('total data length: ' + str(len(self.wav_file)))

        self.stft_frame = opt.stft_frame
        self.stft_hop = opt.stft_hop
        self.audio_window = opt.audio_window
        random.seed(opt.seed)
        self._init_transform()

        if opt.stage == 'two':
            label_file = open('/data/whs/object/MUSIC_SF/class.json', 'r')
            self.pseudo_label = json.load(label_file)

        if opt.mode == 'test':
            pass


    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.mode == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(self.imgSize, Image.BICUBIC),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

    def create_id(self, wavFile):
        return wavFile.split('.')[0]

    def create_label(self, vid):
        return vid.split('/')[0]

    def __getitem__(self, index):
        wavFile = self.wav_file[index]
        sub_path = self.dict.get(wavFile).get('path')
        audio_path = os.path.join(self.data_dir, sub_path)
        # if self.mode == 'test':
        #     # audio_path = "/data/whs/checkpoint/SI_MUSIC_DUET/silence.wav"
        #     audio_path = audio_path.split('.')[0] + '.wav'

        # obtain spectrogram
        # audio, audio_rate = librosa.load(audio_path, sr=self.opt.audio_sampling_rate)
        file_audio = open(audio_path, "rb")
        audio = pickle.load(file_audio)
        audio_segment = sample_audio(audio, self.audio_window)
        audio_mag, audio_phase = generate_spectrogram_magphase(audio_segment, self.stft_frame, self.stft_hop)

        # audio_mag = audio_mag[np.newaxis, :, :]
        audio_mag = torch.tensor(audio_mag)



        # obtain image
        image_path = os.path.join(self.image_dir, sub_path.split('.')[0].split('/')[1] + '.jpg')
        if self.mode == 'test':
            image_path = os.path.join(self.image_dir, sub_path.split('.')[0].split('/')[1] + '.jpg')
        image = Image.open(image_path).convert('RGB')
        image = self.img_transform(image)

        data = {'audio': audio_mag, 'image': image, 'id': wavFile}


        str = self.dict.get(wavFile).get('label').split(',')
        tig = np.zeros(14)
        for t in str:
            t = int(t)
            tig[t] = 1
        zong_tag = tig.reshape(1, 14)
        data['label'] = torch.tensor(zong_tag, dtype=torch.float32)

        if self.opt.stage == 'two' and self.mode == 'train':
            tag = self.pseudo_label.get(wavFile)
            p_label = np.zeros(self.opt.num_class)
            for i in tag:
                p_label[i] = 1
            image_label = p_label.reshape(1, self.opt.num_class)
            data['image_label'] = torch.tensor(image_label, dtype=torch.float32)
            tag_1 = tag[0]
            tag_2 = tag[1]
            data['pseudo_label1'] = tag_1  # [X]
            data['pseudo_label2'] = tag_2  # [X]
        elif self.mode == 'test':
            data['image_label'] = torch.rand(1, self.opt.num_class)
            data['pseudo_label1'] = torch.tensor(0)  # [X]
            data['pseudo_label2'] = torch.tensor(0)  # [X]


        if self.mode == 'test':
            data['vid'] = sub_path
        else:
            data['vid'] = wavFile
        return data

    def __len__(self):
        return len(self.wav_file)

    def name(self):
        return 'AudioVisual_AudioSet'
