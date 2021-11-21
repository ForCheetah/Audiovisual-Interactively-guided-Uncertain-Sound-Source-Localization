import os
import numpy as np
from data.data_loader import createDataloader
from models.Localization_Current_Model import LocalizationModel
import torch
from options.test_options import TestOptions
from models.models import ModelBuilder
import sklearn.metrics
import json
import cv2

def clip_audio(audio):
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

def cal_AUC(cious):
	results = []
	for i in range(21):
		result = np.sum(np.array(cious)>=0.05*i)
		result = result / len(cious)
		results.append(result)
	x = [0.05*i for i in range(21)]
	auc = sklearn.metrics.auc(x, results)
	return auc

def get_ciou(boxs, heat):

	gtmap = np.zeros((224, 224))
	for i in range(len(boxs)):
		box = boxs[i]
		box = np.array(box) * 224
		gtmap[int(box[1]): int(box[1] + box[3]), int(box[0]): int(box[0] + box[2])] = 1
	# resize predited avmap to (224, 224)
	predmap = heat

	thresvalue = 0.1 * np.max(predmap)
	ciou = np.sum((predmap > thresvalue) * (gtmap > 0)) / (np.sum(gtmap) + np.sum((predmap > thresvalue) * (gtmap == 0)))
	return ciou

def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    return value

def main(visualization):
	# parse arguments
	opt = TestOptions().parse()
	opt.device = torch.device("cuda")
	opt.mode = 'test'
	opt.stage = 'two'
	data_loader = createDataloader(mode='test', opt=opt)
	json_file = open("/data/whs/dataset/MUSIC_Duet/Test/MUSIC-Duet_test_paths.json", 'r')
	dict = json.load(json_file)

	builder = ModelBuilder()
	net_image_extract = builder.build_ImageExtract(
		weights="")
	net_audio_extract = builder.build_AudioExtract(
		weights="")
	net_AVattention = builder.build_Heatmap(
		weights="")
	net_Audio_Separation = builder.build_AudioSeparation(
		weights="")
	net_Align = builder.build_Align(
		mode='train',
		weights="")
	net_ClassifyOnAudio = builder.build_ClassifyOnAudio(
		weights="",
		num_class=opt.num_class)
	net_ClassifyOnImage = builder.build_ClassifyOnImage(
		weights="",
		num_class=opt.num_class)
	nets = (net_image_extract, net_audio_extract, net_AVattention,
			net_Audio_Separation, net_Align, net_ClassifyOnAudio, net_ClassifyOnImage)

	# construct our audio-visual model
	model = LocalizationModel(nets, opt)
	model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
	model.to(opt.device)

	cious = []
	nosound = []
	thresvalue = -0.3
	for i, data in enumerate(data_loader):
		print(i)
		output = model.forward(data)
		heatmap1 = output['heatmap1'].squeeze(1).detach().cpu().numpy()
		heatmap2 = output['heatmap2'].squeeze(1).detach().cpu().numpy()
		class_pre = torch.sigmoid(output['imagePC']).detach().cpu().numpy()   #  [X,30]


		i = 0
		for id in data['id']:
			sub_path = data['vid'][i]
			path = os.path.join('/data/whs/dataset/MUSIC_Duet/Test/', sub_path)
			img = cv2.imread(path)
			img = cv2.resize(img, (224, 224))
			heatmap = np.minimum(heatmap1[i], heatmap2[i])
			label = class_pre[i]

			heatmapc = (heatmap - np.min(heatmap))/(np.max(heatmap) - np.min(heatmap))

			heat = cv2.resize(heatmapc, (224, 224), interpolation=cv2.INTER_LINEAR)

			list = dict.get(id).get('box')
			boxs = []
			for item in list:
				boxs.append(item.get('normbox'))
			ciou = get_ciou(boxs, heat)
			i += 1
			cious.append(ciou)


			if np.max(label) < 0.3:
				nosound.append(np.sum(heatmap <= thresvalue) / (14 * 14))


	auc = cal_AUC(cious)
	print('cIoU', np.sum(np.array(cious) >= 0.3) / len(cious))
	print('auc: ' + str(auc))

	print('*****************************************************')
	print(nosound)
	print(np.mean(nosound))


if __name__ == '__main__':
	main(False)



