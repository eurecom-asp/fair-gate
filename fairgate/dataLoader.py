'''
DataLoader for training
'''

import glob, numpy, os, random, soundfile, torch
from scipy import signal

class train_loader(object):
	def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
		self.train_path = train_path
		self.num_frames = num_frames
		# Load and configure augmentation files
		self.noisetypes = ['noise','speech','music']
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		self.noiselist = {}
		augment_files = glob.glob(os.path.join(musan_path,'*/*/*/*.wav')) if os.path.isdir(musan_path) else []
		for file in augment_files:
			if file.split('/')[-4] not in self.noiselist:
				self.noiselist[file.split('/')[-4]] = []
			self.noiselist[file.split('/')[-4]].append(file)
		self.rir_files = glob.glob(os.path.join(rir_path,'*/*/*.wav')) if os.path.isdir(rir_path) else []
		# Load data & labels
		# -------- 读取 train_list，并兼容一列/两列格式 --------
		self.data_list = []
		spk_names = []

		with open(train_list, 'r') as f:
			lines = [ln.strip() for ln in f if ln.strip()]

		for ln in lines:
			parts = ln.split()
			if len(parts) == 2:
				# 两列：spk  相对/绝对路径
				spk, path = parts[0], parts[1]
				wav = path if os.path.isabs(path) else os.path.join(train_path, path)
			elif len(parts) == 1:
				# 一列：绝对/相对路径；从路径里取倒数第3层当作说话人ID（VoxCeleb2 是 idXXXXX）
				path = parts[0]
				wav = path if os.path.isabs(path) else os.path.join(train_path, path)
				comps = os.path.normpath(wav).split(os.sep)
				spk = comps[-3] if len(comps) >= 3 else os.path.basename(os.path.dirname(wav))
			else:
				continue

			self.data_list.append(wav)
			spk_names.append(spk)

		# 映射到连续的类别ID
		spk2id = {s: i for i, s in enumerate(sorted(set(spk_names)))}
		self.data_label = [spk2id[s] for s in spk_names]

	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		# audio, sr = soundfile.read(self.data_list[index])
		try:
			audio, sr = soundfile.read(self.data_list[index])
		except Exception as e:
			raise RuntimeError(
				f"[AUDIO-READ-ERROR] idx={index} file={self.data_list[index]} -> {type(e).__name__}: {e}")

		length = self.num_frames * 160 + 240
		if audio.shape[0] <= length:
			shortage = length - audio.shape[0]
			audio = numpy.pad(audio, (0, shortage), 'wrap')
		start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
		audio = audio[start_frame:start_frame + length]
		audio = numpy.stack([audio],axis=0)

		# Data Augmentation（可选：根据资源是否可用动态决定）
		candidates = [0]  # 0=原始
		if len(self.rir_files) > 0: candidates.append(1)
		if self.noiselist.get('speech'): candidates.append(2)
		if self.noiselist.get('music'):  candidates.append(3)
		if self.noiselist.get('noise'):  candidates.append(4)
		if self.noiselist.get('speech') and self.noiselist.get('music'): candidates.append(5)
		augtype = random.choice(candidates)
		if augtype == 0:   # Original
			audio = audio
		elif augtype == 1: # Reverberation
			audio = self.add_rev(audio)
		elif augtype == 2: # Babble
			audio = self.add_noise(audio, 'speech')
		elif augtype == 3: # Music
			audio = self.add_noise(audio, 'music')
		elif augtype == 4: # Noise
			audio = self.add_noise(audio, 'noise')
		elif augtype == 5: # Television noise
			audio = self.add_noise(audio, 'speech')
			audio = self.add_noise(audio, 'music')
		return torch.FloatTensor(audio[0]), self.data_label[index]

	def __len__(self):
		return len(self.data_list)

	def add_rev(self, audio):
		# NEW: 若没有 RIR 资源，直接不做增强
		if not self.rir_files:
			return audio
		rir_file = random.choice(self.rir_files)

		rir, sr     = soundfile.read(rir_file)
		rir         = numpy.expand_dims(rir.astype(numpy.float32),0)  # CHG: 避免 numpy.float 警告
		rir         = rir / numpy.sqrt(numpy.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:,:self.num_frames * 160 + 240]

	def add_noise(self, audio, noisecat):
		# NEW: 若没有对应类别的噪声，直接返回原音
		if not self.noiselist.get(noisecat):
			return audio
		clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)

		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
		noises = []
		for noise in noiselist:
			noiseaudio, sr = soundfile.read(noise)
			length = self.num_frames * 160 + 240
			if noiseaudio.shape[0] <= length:
				shortage = length - noiseaudio.shape[0]
				noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
			start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))
			noiseaudio = noiseaudio[start_frame:start_frame + length]
			noiseaudio = numpy.stack([noiseaudio],axis=0)
			noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio
# Public alias with standard class naming.
VoxCelebTrainDataset = train_loader
