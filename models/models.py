import torch
import torchvision
# from models.image_extract import imageExtract
from models.image_branch import ImageBranch
from models.image_extract import imageExtract
from models.audio_extract import audio_extract
from models.Heatmap import Heatmap
from models.Classify import Classify, ClassifyOnImage, ClassifyOnAudio
from models.audio_extract import BasicBlock
from models.Align import Align
from models.GuidedSeparation import AudioSeparation

class ModelBuilder():

    def build_AudioExtract(self, weights=''):
        net = audio_extract()

        if len(weights) > 0:
            print('Loading weights for Audio Extract')
            net.load_state_dict(torch.load(weights))
        return net

    def build_ImageExtract(self, weights=''):
        net = imageExtract()

        if len(weights) > 0:
            print('Loading weights for Image Extract')
            net.load_state_dict(torch.load(weights))
        return net




    def build_Image_branch(self, weights=''):
        net = ImageBranch()
        if len(weights) > 0:
            print('Loading weights for image branch')
            net.load_state_dict(torch.load(weights))
        return net



    def build_AVattention(self, weights=''):
        net = AVattention()
        if len(weights) > 0:
            print('Loading weights for AVattention')
            net.load_state_dict(torch.load(weights))
        return net

    def build_Heatmap(self, weights=''):
        net = Heatmap()
        if len(weights) > 0:
            print('Loading weights for Heatmap')
            net.load_state_dict(torch.load(weights))
        return net

    def build_ConstraintCluster(self, weights=''):
        net = ContraintCluster()
        if len(weights) > 0:
            print('Loading weights for ContraintCluster')
            net.load_state_dict(torch.load(weights))
        return net

    def build_Classify(self, weights=''):
        net = Classify()
        if len(weights) > 0:
            print('Loading weights for Classify')
            net.load_state_dict(torch.load(weights))
        return net

    def build_ClassifyOnImage(self, weights='', num_class=30):
        net = ClassifyOnImage(num_class=num_class)
        if len(weights) > 0:
            print('Loading weights for ClassifyOnImage')
            net.load_state_dict(torch.load(weights))
        return net

    def build_ClassifyOnAudio(self, weights='', num_class=30):
        net = ClassifyOnAudio(num_class=num_class)
        if len(weights) > 0:
            print('Loading weights for ClassifyOnAudio')
            net.load_state_dict(torch.load(weights))
        return net

    def build_Align(self, mode='train', weights=''):
        net = Align(mode=mode)
        if len(weights) > 0:
            print('Loading weights for Align')
            net.load_state_dict(torch.load(weights))
        return net

    def build_AudioSeparation(self, weights=''):
        net = AudioSeparation()
        if len(weights) > 0:
            print('Loading weights for AudioSeparation')
            net.load_state_dict(torch.load(weights))
        return net