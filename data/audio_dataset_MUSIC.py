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

def get_vid_name(npy_path):
    #first 11 chars are the video id
    return os.path.basename(npy_path)[0:11]

def get_class_tag(npy_path):
    # class_tag = str(npy_path,encoding='utf-8').split('/')[-2]
    class_tag = npy_path.split('/')[-2]
    return class_tag

def get_clip_name(npy_path):
    return os.path.basename(npy_path)[0:-4]

def get_frame_root(npy_path):
    a = os.path.dirname(os.path.dirname(npy_path))
    return os.path.join(os.path.dirname(a), 'solo_extract')

def get_ins_name(npy_path):
    return os.path.basename(os.path.dirname(npy_path))

def get_audio_root(npy_path):
    a = os.path.dirname(os.path.dirname(npy_path))
    return os.path.join(os.path.dirname(a), 'solo_audio_resample')



class AudioVisual_audioset(torchdata.Dataset):
    def __init__(self, mode, opt):
        super(AudioVisual_audioset, self).__init__()
        self.mode = mode
        self.opt = opt
        self.imgSize = opt.image_size
        self.data_dir = opt.data_dir
        self.image_dir = opt.image_dir

        if self.mode == 'train':
            json_file = open("/data/whs/dataset/MUSIC_Duet/SoloDuet_Train.json", 'r')


        else:
            json_file = open('/data/whs/dataset/MUSIC_Duet/Test/MUSIC-Duet_test_paths.json', 'r')
            self.data_dir = '/data/whs/dataset/MUSIC_Duet/Train/duet_audio_resample/'
            self.image_dir = '/data/whs/dataset/MUSIC_Duet/Test/'
            # json_file = open('/home/whs/programs/Localization_AudioSet/data/paths_train.json', 'r')
        self.dict = json.load(json_file)
        self.wav_file = list(self.dict.keys())



        print('total data length: ' + str(len(self.wav_file)))

        self.stft_frame = opt.stft_frame
        self.stft_hop = opt.stft_hop
        self.audio_window = opt.audio_window
        random.seed(opt.seed)
        self._init_transform()

        if opt.stage == 'two':
            label_file = open("/data/whs/checkpoint/Synthetic/class.json", 'r')
            self.pseudo_label = json.load(label_file)




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
        item = self.dict.get(wavFile)
        audio_path = item.get('path')
        if self.mode == 'test':
            audio_path = audio_path.split('.')[0] + '.wav'

        # obtain spectrogram
        audio, audio_rate = librosa.load(audio_path, sr=self.opt.audio_sampling_rate)
        audio_segment = sample_audio(audio, self.audio_window)
        audio_mag, audio_phase = generate_spectrogram_magphase(audio_segment, self.stft_frame, self.stft_hop)

        # audio_mag = audio_mag[np.newaxis, :, :]
        audio_mag = torch.tensor(audio_mag)



        # obtain image
        image_path = item.get('image_path')

        image = Image.open(image_path).convert('RGB')
        image = self.img_transform(image)

        data = {'audio': audio_mag, 'image': image, 'id': wavFile}


        if self.opt.stage == 'two' and self.mode == 'train':
            tag = self.pseudo_label.get(wavFile)
            p_label = np.zeros(self.opt.num_class)
            for i in tag:
                p_label[i] = 1
            image_label = p_label.reshape(1, self.opt.num_class)
            data['image_label'] = torch.tensor(image_label, dtype=torch.float32)
            #

            tag_1 = tag[0]
            tag_2 = tag[1]
            data['pseudo_label1'] = tag_1  # [X]
            data['pseudo_label2'] = tag_2  # [X]
        elif self.mode == 'test':  #
            data['image_label'] = torch.rand(1, self.opt.num_class)
            data['pseudo_label1'] = torch.tensor(0)  # [X]
            data['pseudo_label2'] = torch.tensor(0)  # [X]


        data['vid'] = wavFile
        return data

    def __len__(self):
        return len(self.wav_file)

    def name(self):
        return 'AudioVisual_Synthetic'
