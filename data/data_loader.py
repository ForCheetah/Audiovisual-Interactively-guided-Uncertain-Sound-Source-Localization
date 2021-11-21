from utils.utils import object_collate
# from data.audio_dataset import AudioVisual_audioset
from data.audio_dataset_MUSIC import AudioVisual_audioset
import torch.utils.data

def createDataloader(mode, opt):
    dataset = AudioVisual_audioset(mode, opt)
    dataloader = None
    if opt.mode == "train":
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.nThreads),
            collate_fn=object_collate)
    elif opt.mode == 'val':
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,
            shuffle=False,
            num_workers=2,
            collate_fn=object_collate)
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,
            shuffle=False,
            )
    return dataloader
