import datetime
import shutil
import sys

import torch

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

from os import listdir, path
import os

if not os.path.isfile('face_detection/detection/sfd/s3fd.pth'):
	raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
							before running this script!')

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
from glob import iglob
import audio
from hparams import hparams as hp

import face_detection
import ffmpeg
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", required=True)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)

argv = ['--batch_size', '7', '--data_root', '/home/sean/Downloads/AVSpeech/', '--preprocessed_root', '/home/sean/data/AVSpeech']
args = parser.parse_args(argv)

fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
									device='cuda:{}'.format(id)) for id in range(args.ngpu)]

template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -filter:v fps=fps=25 -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'
template3 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -filter:v fps=fps=25 -vsync 1 -async 1 -ac 1 -acodec pcm_s16le -ar 16000 {}'

def repl_test():
	import numpy as np
	import matplotlib.pyplot as plt
	from PIL import Image
	from ISR.models import RDN
	from ISR.models import RRDN

	rdn = RDN(weights='psnr-large')
	rrdn = RRDN(weights='gans')

	img = Image.open('/home/sean/data/AVSpeech/TUyDankfTsY/TUyDankfTsY_1/63.jpg')
	lr_img = np.array(img)
	sr_img = rdn.predict(lr_img, by_patch_of_size=50)
	test = Image.fromarray(sr_img)
	f, axarr = plt.subplots(1,2)
	axarr[0].imshow(img)
	axarr[1].imshow(test)
#	plt.show()
	plt.savefig('foo.png', dpi = 300)
	test.save('foo.png')




def process_video_file(vfile, args, gpu_id):
	video_stream = cv2.VideoCapture(vfile)
	
	frames = []
	time_now = datetime.datetime.now()
	while 1:
		still_reading, frame = video_stream.read()
		if not still_reading:
			video_stream.release()
			break
		frames.append(frame)
	print("Read all video frames: " + str(datetime.datetime.now() - time_now))
	
	vidname = os.path.basename(vfile).split('.')[0]
	dirname = vfile.split('/')[-2]

	fulldir = os.path.join(args.preprocessed_root, dirname, vidname)
	os.makedirs(fulldir, exist_ok=True)

	batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]

	i = -1
	for fb in batches:
		time_now = datetime.datetime.now()
		#torch.cuda.empty_cache()
		preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))
		print("Batch face detections: " + str(datetime.datetime.now() - time_now))

		time_now = datetime.datetime.now()
		for j, f in enumerate(preds):
			i += 1
			if f is None:
				continue

			x1, y1, x2, y2 = f

			# hack!  skip processing if it's a low res image
			if(x2-x1 < 96 or y2-y1 < 96):
				print("LOWRES:", x2-x1, y2-y1, vfile)
				raise Exception("LOWRES")

			cv2.imwrite(os.path.join(fulldir, '{}.jpg'.format(i)), fb[j][y1:y2, x1:x2])
		print("Batch write to disk: " + str(datetime.datetime.now() - time_now))


def process_audio_file(vfile, args):
	time_now = datetime.datetime.now()
	vidname = os.path.basename(vfile).split('.')[0]
	dirname = vfile.split('/')[-2]

	fulldir = os.path.join(args.preprocessed_root, dirname, vidname)
	os.makedirs(fulldir, exist_ok=True)

	wavpath = os.path.join(fulldir, 'audio.wav')

	command = template.format(vfile, wavpath)
	subprocess.call(command, shell=True)
	print("Process audio: " + str(datetime.datetime.now() - time_now))


def mp_handler(job):
	vfile, args, gpu_id = job

	# if the destination folder already exists don't process it
	vidname = os.path.basename(vfile).split('.')[0]
	dirname = vfile.split('/')[-2]
	fulldir = os.path.join(args.preprocessed_root, dirname, vidname)
	if(os.path.exists(fulldir)):
		print("already processed: {}".format(fulldir))
		return

	path = Path(vfile)
	basename = os.path.basename(path).split('.')[0]
	basename_parent = os.path.basename(path.parent)
	vfile_tmp_path = os.path.join('/tmp/', basename_parent)
	vfile_tmp = os.path.join(vfile_tmp_path, basename + '.mkv')
	os.makedirs(vfile_tmp_path, exist_ok=True)

	time_now = datetime.datetime.now()
	command = template3.format(vfile, vfile_tmp)
	subprocess.call(command, shell=True)
	print("Media conversion: " + str(datetime.datetime.now() - time_now))

	try:
		process_video_file(vfile_tmp, args, gpu_id)
		process_audio_file(vfile_tmp, args)
	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()

	shutil.rmtree(vfile_tmp_path)

def main(args):
	print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))

	filelist = []
	for filename in iglob(args.data_root + '**/*.mp4', recursive=True):
		filelist.append(filename)

	#filelist = [filelist[0]]

	jobs = [(vfile, args, i%args.ngpu) for i, vfile in enumerate(filelist)]
	p = ThreadPoolExecutor(args.ngpu)
	futures = [p.submit(mp_handler, j) for j in jobs]
	_ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

	# print('Dumping audios...')
	#
	# for vfile in tqdm(filelist):
	# 	try:
	# 		process_audio_file(vfile, args)
	# 	except KeyboardInterrupt:
	# 		exit(0)
	# 	except:
	# 		traceback.print_exc()
	# 		continue

if __name__ == '__main__':
	main(args)



def repl_test2():
	filelist = []
	for filename in iglob(args.data_root + '**/*.mp4', recursive=True):
		filelist.append(filename)

	i, vfile = next(iter(enumerate(filelist)))
