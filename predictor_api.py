import math
import tempfile
import urllib
from os import listdir, path
from urllib.parse import urlparse

import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string

from pydub import AudioSegment
from skimage.draw import ellipse
from sklearn.metrics import euclidean_distances
from tqdm import tqdm
from glob import glob
import torch

# constants
N_MEL_CHANNELS = 80.
MAX_WAV_VALUE = 32767.0
MEL_STEP_SIZE = 16

# useful for keeping track of face bounding boxes
class BoundingBox(object):
    """
    A 2D bounding box
    """
    def __init__(self, x1:int, y1:int, x2:int, y2:int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    # def __init__(self, points):
    #     if len(points) == 0:
    #         raise ValueError("Can't compute bounding box of empty list")
    #
    #     # standardize to ints
    #     converted = np.floor(points).astype(np.int)
    #
    #     # use argmax to find the min and max boundaries
    #     c_min = converted.argmin(axis=0)
    #     c_max = converted.argmax(axis=0)
    #     self.x1 = converted[c_min[0]][0]
    #     self.y1 = converted[c_min[1]][1]
    #     self.x2 = converted[c_max[0]][0]
    #     self.y2 = converted[c_max[1]][1]

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    def __str__(self):
        return "(x1: {0}, y1: {1})\t(x2: {2}, y2: {3})\n".format(self.x1, self.y1, self.x2, self.y2)

    def __repr__(self):
        return "(x1: {0}, y1: {1})\t(x2: {2}, y2: {3})\n".format(self.x1, self.y1, self.x2, self.y2)


def extract_video_frames(video_path):
    orig_video_stream = cv2.VideoCapture(video_path)
    orig_video_fps = orig_video_stream.get(cv2.CAP_PROP_FPS)
    orig_images_cv = []
    while True:
        still_reading, frame = orig_video_stream.read()
        if not still_reading:
            orig_video_stream.release()
            break
        orig_images_cv.append(frame)

    return orig_images_cv, orig_video_fps

# downloads and extracts the audio and video frames
def download_and_extract_audio_and_video_frames(video_url):

    # download the video to a tempfile
    video_url_parsed = urlparse(video_url)
    video_file_path, video_file_ext = os.path.splitext(video_url_parsed.path)
    video_file_name = os.path.basename(video_url_parsed.path)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_video_path = os.path.join(tmp_dir, video_file_name)
        with urllib.request.urlopen(video_url) as response, open(tmp_video_path, 'wb') as video_file:
            data = response.read()  # a `bytes` object
            video_file.write(data)

            # extract audio and convert to proper frame rate
            orig_audio_pydub = AudioSegment.from_file(tmp_video_path, video_file_ext[1:])
            #converted_audio_pydub = orig_audio_pydub.set_frame_rate(AUDIO_FRAME_RATE)
            #assert(converted_audio_pydub.frame_rate == AUDIO_FRAME_RATE)

            # extract video into frames
            orig_video_stream = cv2.VideoCapture(tmp_video_path)
            orig_video_fps = orig_video_stream.get(cv2.CAP_PROP_FPS)
            orig_video_frames = []
            while True:
                still_reading, frame = orig_video_stream.read()
                if not still_reading:
                    orig_video_stream.release()
                    break
                orig_video_frames.append(frame)

            return orig_audio_pydub, orig_video_frames, orig_video_fps

# landmarks
#
# center nose = 30
# bottom nose = 33
# left cheeck = 4
# right cheek = 12
# bottom face = 8
def draw_mouth_elipsis(box, landmarks):

    # rescale the landmark points to be [0-1]
    h = box.y2 - box.y1
    w = box.x2 - box.x1

    # converts points to be relative
    def normalize(landmark):
        return (landmark - [box.x1, box.y1]) / [w, h]

    center_nose = normalize(landmarks[30])
    top_edge = normalize(landmarks[33])
    bottom_edge = normalize(landmarks[8])
    left_edge = normalize(landmarks[4])
    right_edge = normalize(landmarks[12])
    c, r = normalize(landmarks[62])
    c_radius = min(c - left_edge[0], right_edge[0] - c)
    r_radius = max(r - top_edge[1], bottom_edge[1] - r)

    # create an empty image mask at 1/10th the size
    mh = int(h / 10)
    mw = int(w / 10)
    mouth_mask = np.zeros((mh, mw))

    # fill ellipse on the smaller mask
    rr, cc = ellipse(r * mh, c * mw, r_radius * mh, c_radius * mw, mouth_mask.shape)

    # calculate euclidean distances in one pass
    point_sets = np.asarray(list(zip(rr, cc)))
    distances = euclidean_distances(point_sets, [[r * mh, c * mw]])
    distances_scale_min = distances.max() * 0.70
    distances[distances > distances_scale_min] = distances[distances > distances_scale_min] * 1.3
    mouth_mask[rr, cc] = distances.flatten()


    # rescale to the max value and invert
    mouth_mask *= 1.0 / mouth_mask.max()
    mouth_mask[rr, cc] = 1 - mouth_mask[rr, cc]

    # set any values above the nose to 0
    mouth_mask[0:int(center_nose[1] * mh), :] = 0

    # now resize back to original proportions
    # TODO: speed this up
    resized = cv2.resize(mouth_mask, (w,h))
    # import matplotlib.pyplot as plt
    # plt.imshow(resized)
    # plt.show()

    return resized

def create_mask(images_cv, frame_id_to_landmarks, frame_id_to_box):
    import numpy as np
    from matplotlib import pyplot as plt

    import matplotlib.pyplot as plt

    from skimage.draw import line, polygon, circle, ellipse
    import numpy as np

    img = np.zeros((500, 500, 1), 'uint8')



    # for each frame with landmarks
    for f_id, landmarks in frame_id_to_landmarks.items():

        # using the face bounding box make the lower half of the face_mask to be NaN
        box = frame_id_to_box[f_id]

        # create an empty image mask
        h = box.y2 - box.y1
        w = box.x2 - box.x1
        mouth_mask = np.zeros((h, w))

        # draw an oval from left cheek to right cheek and nose tip to chin
        plt.imshow(mouth_mask)

        # Translate landmark coordinates to the bounding box
        for p_idx, prediction in enumerate(frame_id_to_landmarks[f_id][0]):
            x = math.floor(prediction[0]) - box.x1
            y = math.floor(prediction[1]) - box.y1

            plt.text(x, y, str(p_idx), fontsize=5)

            # large oval
            # bottom nose = 33
            # left cheeck = 4
            # right cheek = 12
            # bottom face = 8

            # smaller oval
            # left lips = 48
            # right lips = 54
            # center = 62

            if p_idx > 48:
                mouth_mask[y, x] = 255
            else:
                mouth_mask[y, x] = 0

        plt.show()

        # set the lower half of the face to nan
        box_y_mid = box.y1 + ((box.y2 - box.y1) // 2)
        mouth_mask[box_y_mid:box.y2, box.x1:box.x2] = np.nan


        # crop to the face
        mouth_mask = mouth_mask[box.y1:box.y2, box.x1:box.x2]

        # resize to 96x96
        #mouth_mask_img = cv2.from(mouth_mask)
        # cv2.cvtColor(mouth_mask, cv2.COLOR_GRAY2BGR)
        # mask_gray = cv2.normalize(src=mouth_mask, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
        #                           dtype=cv2.CV_8UC1)

        # mask invalid values
        # x = np.arange(0, mouth_mask.shape[1])
        # y = np.arange(0, mouth_mask.shape[0])
        # array = np.ma.masked_invalid(mouth_mask)
        # xx, yy = np.meshgrid(x, y)
        # x1 = xx[~array.mask]
        # y1 = yy[~array.mask]
        # newarr = array[~array.mask]
        #
        # GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
        #                            (xx, yy),
        #                            method='linear')
        #
        # plt.imshow(mouth_mask, interpolation='nearest')
        # plt.show()


       #
       # from scipy import interpolate
       # x = np.arange(-5.01, 5.01, 0.25)
       # y = np.arange(-5.01, 5.01, 0.25)
       # xx, yy = np.meshgrid(x, y)
       # z = np.sin(xx ** 2 + yy ** 2)
       # f = interpolate.interp2d(x, y, z, kind='cubic')
       #


    #
    #
    #
    # # create masks setting 1 values at mouth coordinates
    # for b_idx, image_batch in enumerate(images_tensor):
    #     image = image_batch.permute([1,2,0])
    #     plt.imshow(image)
    #     for p_idx, prediction in enumerate(frame_id_to_landmarks[b_idx][0]):
    #         if p_idx > 48:
    #             plt.text(prediction[0], prediction[1], str(p_idx),  fontsize=5)
    #             print(prediction)
    #     plt.show()


# generic function for processing data at a specified batch size
# If an out of memory error occurs, batch size is cut in half and retried
def process_in_batches(data, batch_size, batch_fn):

    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(data), batch_size)):
                idx_start = i
                idx_end = i + batch_size
                data_batch = data[idx_start:idx_end]
                results = batch_fn(data_batch)
                predictions.extend(results)
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'process_in_batches: OOM with batch_size 1')
            batch_size //= 2
            print('process_in_batches: Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    return predictions


# multiple faces can be detected per image, filter by min_confidence score and return idx -> facebb
def face_detect_filter(face_detections, min_confidence_score=0.90):
    detected_faces_highest_scores = {}
    for b_idx, batch in enumerate(face_detections):
        highest = None
        highest_score = min_confidence_score
        for face in iter(batch):
            if face[-1] > highest_score:
                highest = face
        if highest is not None:
            detected_faces_highest_scores[b_idx] = (np.asarray([highest]))
    return detected_faces_highest_scores


# process a batch of landmarks using the filtered detected faces
def process_landmark_batch(data, detector):
    images_batch = [d['image_tensor'] for d in data]
    face_detected_batch = [d['face_detected'] for d in data]
    results = detector.get_landmarks_from_batch(images_batch, face_detected_batch)
    return results


# returns updated coordinates in frame_id -> [x1,y1,x2,y2] taking into account padding
def process_bounding_boxes(images_cv, frame_id_to_face_detections, pads):

    frame_id_to_box = {}
    pady1, pady2, padx1, padx2 = pads
    for frame_id, face_detections in frame_id_to_face_detections.items():
        image = images_cv[frame_id]

        rect = face_detections[0].astype(np.int)  # there's only one face
        y1 = math.floor(max(0, rect[1] - pady1))
        y2 = math.floor(min(image.shape[0], rect[3] + pady2))
        x1 = math.floor(max(0, rect[0] - padx1))
        x2 = math.floor(min(image.shape[1], rect[2] + padx2))
        box = BoundingBox(x1, y1, x2, y2)
        frame_id_to_box[frame_id] = box

    return frame_id_to_box


def process_faces_and_landmarks(face_alignment_detector, images_cv, args):

    # convert cv2 BGR images to a Pytorch Tensor ready for batch processing (b_c_h_w RGB)
    images_tensor = torch.from_numpy(np.asarray(images_cv.copy())).permute(0, 3, 1, 2)[:, [2, 1, 0], :, :]

    # process face detection on all images
    face_detections = process_in_batches(images_tensor, args.face_det_batch_size,
                                         lambda x: face_alignment_detector.face_detector.detect_from_batch(x))

    # only keep the highest confidence score for bounding boxes
    # (multiple faces could be detected per image, but this keeps things simple)
    # returns a dict with {frame_id: facebb)
    frame_id_to_face_detections = face_detect_filter(face_detections, args.face_det_confidence_score_min)

    # # process the faces for landmarks
    # # returns a dict with {frame_id: landmarks)
    # faces_detected_filtered_keys, faces_detected_filtered_values = zip(*frame_id_to_face_detections.items())
    # images_tensor_filtered = torch.index_select(images_tensor, 0, torch.tensor(np.asarray(faces_detected_filtered_keys)))
    # data = [{"image_tensor": images_tensor_filtered[idx]
    #             , "face_detected": faces_detected_filtered_values[idx]} for idx in
    #         range(0, len(images_tensor_filtered))]
    # detected_landmarks = process_in_batches(data, args.face_landmark_batch_size,
    #                    lambda x: process_landmark_batch(x, face_alignment_detector))
    # frame_id_to_landmarks = dict(zip(faces_detected_filtered_keys, detected_landmarks))

    # smooth coordinates and crop out the face from the full frame images
    frame_id_to_box = process_bounding_boxes(images_cv, frame_id_to_face_detections, args.pads)

    # crop and resize faces for processing
    frame_id_to_face_cv = {}
    for idx in frame_id_to_face_detections.keys():
        box = frame_id_to_box[idx]
        face = images_cv[idx][box.y1:box.y2, box.x1:box.x2]
        face_resized = cv2.resize(face,  (args.img_size, args.img_size))
        frame_id_to_face_cv[idx] = face_resized

    return frame_id_to_face_detections, frame_id_to_box, frame_id_to_face_cv


# convert pydub into float 32 [-1,1] values, create mel spectogram, and split into chunks based on the video fps
def extract_mels(audio_pydub, video_fps):
    audio_16000 = audio_pydub.set_frame_rate(16000)
    audio_np = np.array(audio_16000.get_array_of_samples(), dtype="float32")
    audio_f32 = audio_np / MAX_WAV_VALUE
    mel = audio.melspectrogram(audio_f32)

    # sanity check
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mels = []
    mel_idx_multiplier = N_MEL_CHANNELS / video_fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + MEL_STEP_SIZE > len(mel[0]):
            mels.append(mel[:, len(mel[0]) - MEL_STEP_SIZE:])
            break
        mels.append(mel[:, start_idx: start_idx + MEL_STEP_SIZE])
        i += 1

    return mels



# process all raw data
def preprocess_data(face_alignment_detector, images_cv, audio_pydub, args):

    # detect faces, and crop and resize all faces
    frame_id_to_face_detections, frame_id_to_box, frame_id_to_face_cv = process_faces_and_landmarks(face_alignment_detector, images_cv, args)

    # process audio into mel_chunks (indexed by frames)
    mels = extract_mels(audio_pydub, args.fps)

    # prep the data for processing
    data = []
    face_idxs = frame_id_to_face_detections.keys()
    for idx in face_idxs:
        if idx < len(mels):
            data.append({'frame_id': idx,
                         'img_cv': images_cv[idx],
                         'mel': mels[idx],
                         'face_box': frame_id_to_box[idx],
                         'face_cv': frame_id_to_face_cv[idx],
                         })

    return data


# produces predictions of a single batch
def process_wav2lip_batch(data, wav2lip_model, args, device="cuda"):

    # unpack the data for the batch
    mel_batch = [d['mel'] for d in data]
    face_cv_batch = [d['face_cv'] for d in data]

    # convert into numpy arrays
    face_cv_batch, mel_batch = np.asarray(face_cv_batch), np.asarray(mel_batch)

    # create an image mask covering the bottom of the face
    face_cv_masked = face_cv_batch.copy()
    face_cv_masked[:, args.img_size // 2:] = 0

    # concatenate and reshape face and mel batches
    face_cv_batch = np.concatenate((face_cv_masked, face_cv_batch), axis=3) / 255.
    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

    # convert into tensors and load into memory
    face_cv_batch = torch.FloatTensor(np.transpose(face_cv_batch, (0, 3, 1, 2))).to(device)
    mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

    # run prediction
    with torch.no_grad():
        predictions = wav2lip_model(mel_batch, face_cv_batch)

    # convert back into a cv2 image format
    predictions = predictions.cpu().numpy().transpose(0, 2, 3, 1) * 255.

    return predictions

# For frames that were processed by Wav2Lip:
# - resizes the predicted face to match the original face crop size (96x96x3 -> ?x?x3)
# - generates a mouth alpha mask using facial landmarks and blends the generated face back into the orignal
# - merges together original frames with the predicted frames for a final set
def postprocess_wav2lip(images_cv, predictions, data, mouth_mask_path):

    # load a pre-saved mouth_mask (for alpha blending)
    mouth_mask = np.load(mouth_mask_path)

    # get a dictionary for frame id to predictions, face_boxes, faces, and facial landmarks
    frame_id_to_data = {}
    for d_idx, d in enumerate(data):
        frame_id_to_data[d['frame_id']] = {'pred': predictions[d_idx],
                                           'box': d['face_box']}


    # Some images didn't have faces and don't need to be processed (just accumulated)
    final_frames = []
    for frame_id, orig_image in enumerate(images_cv):

        # if the frame id was processed complete post processing and add it to the final frames
        if frame_id in frame_id_to_data:

            # deconstruct necessary data
            pred = frame_id_to_data[frame_id]['pred']
            box = frame_id_to_data[frame_id]['box']

            # crop the original image to get the original face
            orig_face = orig_image[box.y1:box.y2, box.x1:box.x2]

            # resize the prediction to be it's original size
            pred_resized = cv2.resize(pred.astype(np.uint8), (box.width, box.height))

            # resize the mouth_mask to fit the prediction
            mouth_mask_resized = cv2.resize(mouth_mask, (box.width, box.height))

            # blend using an alpha mask
            a = mouth_mask_resized[:, :, np.newaxis]
            blended = cv2.convertScaleAbs(pred_resized * a + orig_face * (1 - a))

            # write the blended image back into a copy of the original image
            new_img = orig_image.copy()
            new_img[box.y1:box.y2, box.x1:box.x2] = blended

            # add the modified image
            final_frames.append(new_img)

        # no face detected, no processing necessary just add the original image back
        else:
            final_frames.append(orig_image)

    return final_frames



    # blend the predicted image with the original using a generated face mask using landmarks
    frame_id_to_final_cv = {}
    for frame_id, pred, orig_img, box, landmarks in zip(frame_id_batch, predictions, img_cv_batch, face_box_batch, face_landmarks_batch):

        # resize the prediction to be it's original size
        pred_resized = cv2.resize(pred.astype(np.uint8), (box.x2 - box.x1, box.y2 - box.y1))

        # get the original face
        orig_face = orig_img[box.y1:box.y2, box.x1:box.x2]

        # create an alpha transparency mask around the mouth
        mouth_mask = draw_mouth_elipsis(box, landmarks[0])

        # blend using an alpha mask
        a = mouth_mask[:,:, np.newaxis]
        blended = cv2.convertScaleAbs(pred_resized * a + orig_face * (1 - a))

        # write the blended image back into a copy of the original image
        new_img = orig_img.copy()
        new_img[box.y1:box.y2, box.x1:box.x2] = blended

        # save the new image
        frame_id_to_final_cv[frame_id] = new_img

    return frame_id_to_final_cv



def repl_test():
    wav2lip_checkpoint_path = './wav2lip_gan.pth'
    device = "cuda"

    # init
    # face_alignment_detector, wav2lip_model = init(wav2lip_checkpoint_path, device)

    # load tmp data
    argv = ["--wav2lip_checkpoint_path", "wav2lip_gan.pth", "--face",
            "./sample_data/input_vid_portrait_orig.m4v", "--audio", "./sample_data/input_audio.wav"]
    args = gen_args(argv)

    video_path = './sample_data/input_vid_portrait_orig.m4v'
    images_cv, orig_video_fps = extract_video_frames(video_path)
    args.fps = orig_video_fps
    audio_path = "./sample_data/input_audio.wav"
    audio_pydub = AudioSegment.from_wav(audio_path)

    # predictor
    # data = preprocess_data(face_alignment_detector, images_cv, audio_pydub, args)
    #
    # predictions = process_in_batches(data, args.wav2lip_batch_size,
    #                                         lambda x: process_wav2lip_batch(x, wav2lip_model, args, device="cuda"))


def main_(args):
    if not os.path.isfile(args.face):
        fnames = list(glob(os.path.join(args.face, '*.png')))
        sorted_fnames = sorted(fnames, key=lambda f: int(os.path.basename(f).split('.')[0]))
        full_frames = [cv2.imread(f) for f in sorted_fnames]
        fps = args.fps

    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    print("Number of frames available for inference: " + str(len(full_frames)))

    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

        subprocess.call(command, shell=True)
        args.audio = 'temp/temp.wav'

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + MEL_STEP_SIZE > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - MEL_STEP_SIZE:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + MEL_STEP_SIZE])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]

    return full_frames, mel_chunks

    batch_size = args.wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks)

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                    total=int(
                                                                        np.ceil(float(len(mel_chunks)) / batch_size)))):
        if i == 0:
            model = load_model(args.checkpoint_path)
            print("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi',
                                  cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()

    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
    subprocess.call(command, shell=True)
    return True


def gen_args(argv):

    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

    parser.add_argument('--wav2lip_checkpoint_path', type=str,
                        help='Name of saved checkpoint to load weights from', required=True)

    parser.add_argument('--face', type=str,
                        help='Filepath of video/image that contains faces to use', required=True)
    parser.add_argument('--audio', type=str,
                        help='Filepath of video/audio file to use as raw audio source', required=True)
    parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
                        default='results/result_voice.mp4')

    parser.add_argument('--static', type=bool,
                        help='If True, then use only first video frame for inference', default=False)
    parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                        default=25., required=False)

    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                        help='Padding (top, bottom, left, right). Please adjust to include chin at least')

    parser.add_argument('--face_det_batch_size', type=int,
                        help='Batch size for face detection', default=16)
    parser.add_argument('--face_det_confidence_score_min', type=int,
                        help='Minimum score to count as a face', default=0.90)
    parser.add_argument('--face_landmark_batch_size', type=int,
                        help='Batch size for face landmarks', default=16)
    parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

    parser.add_argument('--resize_factor', default=1, type=int,
                        help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                        help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                             'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                        help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                             'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

    parser.add_argument('--rotate', default=False, action='store_true',
                        help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                             'Use if you get a flipped result, despite feeding a normal looking video')

    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')


    args = parser.parse_args(argv)
    args.img_size = 96

    if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        args.static = True

    return args

# create a bounding box based on facial landmarks
# box = BoundingBox(landmarks[0])
#   b = frame_id_to_face_detections[frame_id][0].astype(np.int)
#  box = BoundingBox(b[0], b[1], b[2], b[3])

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# fig, ax = plt.subplots(1)
# ax.imshow(image)
# rect = patches.Rectangle((box.x1, box.y1), box.width, box.height, linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)
# test = frame_id_to_face_detections[frame_id][0]
# test_rect = patches.Rectangle((test[0],test[1]),test[2] - test[0], test[3] - test[1],  linewidth=1, edgecolor='b', facecolor='none')
# ax.add_patch(test_rect)
# plt.show()

# # expand the bounding box boundaries so the bottom half of the box contains the nose down
# center_nose_to_chin_dist = box.y2 - landmarks[0][30][1]
# if landmarks[0][30][1] - box.y1
#
#