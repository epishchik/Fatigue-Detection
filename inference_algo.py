import cv2
import numpy as np
import copy
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import mediapipe as mp
import time
import yaml
from argparse import ArgumentParser

from retinaface_detect import detect as retinaface_detect_faces
from retinaface_detect import create_net as retinaface_model

from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from spiga.demo.visualize.plotter import Plotter

TOTAL_TIME = 0.0
RETINAFACE_TIME = 0.0
SPIGA_TIME = 0.0
ALGO_TIME = 0.0


def extract_bboxes(faces, threshold):
    bboxes = []
    for face in faces:
        if face[4] < threshold:
            continue

        bbox = face[:4]
        new_bbox = xyxy_to_xywh(bbox)

        bboxes.append(new_bbox)
    return bboxes


def xyxy_to_xywh(bbox):
    new_bbox = [0.0] * len(bbox)

    new_bbox[0] = bbox[0]
    new_bbox[1] = bbox[1]
    new_bbox[2] = bbox[2] - bbox[0]
    new_bbox[3] = bbox[3] - bbox[1]

    return new_bbox


def eye_aspect_ratio(eye):
    p2_minus_p8 = dist.euclidean(eye[1], eye[7])
    p3_minus_p7 = dist.euclidean(eye[2], eye[6])
    p4_minus_p6 = dist.euclidean(eye[3], eye[5])
    p1_minus_p5 = dist.euclidean(eye[0], eye[4])
    ear = (p2_minus_p8 + p3_minus_p7 + p4_minus_p6) / (3.0 * p1_minus_p5)
    return ear


def process_image(idx,
                  image,
                  plotter,
                  model='mp_face',
                  plot=False,
                  print_ear=True):
    global TOTAL_TIME
    global RETINAFACE_TIME
    global SPIGA_TIME
    global ALGO_TIME

    normal_features = True
    if model[0] == 'spiga':
        # spiga
        processor = model[1]
        retinaface_net, retinaface_cfg = model[2]

        start = time.time()
        faces = retinaface_detect_faces(image, retinaface_net, retinaface_cfg)
        end = time.time()

        RETINAFACE_TIME += end - start
        TOTAL_TIME += end - start

        try:
            bbox = extract_bboxes(faces, 0.99)[0]

            start = time.time()
            features = processor.inference(image, [bbox])
            end = time.time()

            left_landmarks = np.array(features['landmarks'][0])[60:68]
            right_landmarks = np.array(features['landmarks'][0])[68:76]

            SPIGA_TIME += end - start
            TOTAL_TIME += end - start
        except Exception as _:
            normal_features = False
            features = []
    elif model[0] == 'mp_face':
        # mediapipe face mesh
        mp_model = model[1]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        try:
            features = mp_model.process(image).multi_face_landmarks[0].landmark
        except Exception as _:
            features = []
            normal_features = False
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape
        landmarks = [[f.x * w, f.y * h] for f in features]

        # inverted view, mediapipe uses view from eyes
        mp_face_mesh = mp.solutions.face_mesh
        left_indices = mp_face_mesh.FACEMESH_RIGHT_EYE
        right_indices = mp_face_mesh.FACEMESH_LEFT_EYE

        if normal_features:
            raw_left_landmarks = [landmarks[idx[0]] for idx in left_indices]
            raw_right_landmarks = [landmarks[idx[0]] for idx in right_indices]

            # idx - pos
            # 1 - 0, 3 - 2, 4 - 7, 6 - 3, 8 - 5, 13 - 1, 14 - 6, 15 - 4
            lidx = [1, 13, 3, 6, 15, 8, 14, 4]
            left_landmarks = []
            for idx in lidx:
                left_landmarks.append(raw_left_landmarks[idx])

            # idx - pos
            # 0 - 6, 3 - 1, 4 - 5, 5 - 4, 6 - 3, 10 - 2, 13 - 7, 14 - 0
            ridx = [14, 3, 10, 6, 5, 4, 0, 13]
            right_landmarks = []
            for idx in ridx:
                right_landmarks.append(raw_right_landmarks[idx])

            left_landmarks = np.array(left_landmarks)
            right_landmarks = np.array(right_landmarks)

    canvas = None
    if plot:
        canvas = copy.deepcopy(image)
        if normal_features:
            canvas = plotter.landmarks.draw_landmarks(canvas, left_landmarks)
            canvas = plotter.landmarks.draw_landmarks(canvas, right_landmarks)

        (h, w) = canvas.shape[:2]
        canvas = cv2.resize(canvas, (512, int(h*512/w)))

    start = time.time()

    if normal_features:
        lear = eye_aspect_ratio(left_landmarks)
        rear = eye_aspect_ratio(right_landmarks)
    else:
        lear = 0.5
        rear = 0.5

    end = time.time()

    ALGO_TIME += end - start
    TOTAL_TIME += end - start

    if print_ear:
        str = f'frame: {idx + 1} left EAR: {lear:.5f} right EAR: {rear:.5f}'
        print(str)

    return lear, rear, canvas


def calculate_aes(ears, cnt, show=None):
    top = sorted(ears, reverse=True)[:cnt]

    aes = sum(top) / cnt
    max_threshold = 2 / 3 * aes + 0.0467
    min_threshold = max_threshold - 0.05

    if show is not None:
        str = f'{show} aes: {aes:.5f} '
        str += f'max_thld: {max_threshold:.5f} '
        str += f'min_thld: {min_threshold:.5f}'
        print(str)

    return aes, max_threshold, min_threshold


def calc_fatigue(idx,
                 full_time,
                 start_time,
                 ear,
                 max_thld,
                 min_thld,
                 max_ear,
                 min_ear,
                 speeds,
                 frames,
                 cnts,
                 opening):
    if start_time != 0:
        start_time += 1

    if ear > max_thld:
        if ear > max_ear:
            max_ear = ear
            start_time = 0
        else:
            if start_time == 0:
                start_time += 1

    if ear < min_thld:
        if ear < min_ear:
            min_ear = ear
        else:
            if start_time != 0 and not opening:
                opening = True
                diff = max_ear - min_ear
                speed = diff / start_time
                speeds.append(speed)
    elif ear >= min_thld and opening:
        opening = False
        full_time.append((start_time, idx))

        frames += start_time
        cnts += 1

        start_time = 0
        max_ear = 0.0
        min_ear = 1.0

    out_tuple = (full_time,
                 start_time,
                 max_ear,
                 min_ear,
                 speeds,
                 frames,
                 cnts,
                 opening)
    return out_tuple


def video_end_fatigue(idx,
                      full_time,
                      start_time,
                      max_ear,
                      min_ear,
                      speeds,
                      frames,
                      cnts):
    if start_time > 0:
        full_time.append((start_time, idx))

        speed = (max_ear - min_ear) / start_time
        speeds.append(speed)

        frames += start_time
        cnts += 1

        start_time = 0
        max_ear = 0.0
        min_ear = 1.0

    return full_time, start_time, max_ear, min_ear, speeds, frames, cnts


def display_values(image,
                   ear,
                   freq,
                   dur,
                   aecd,
                   font_size=0.4,
                   thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX

    str_ear = f'EAR: {ear[0]:.3f}, {ear[1]:.3f}'
    str_freq = f'FREQ: {freq[0]:.3f}, {freq[1]:.3f} b/s'
    str_dur = f'DUR: {dur[0]:.3f}, {dur[1]:.3f} s'
    str_aecd = f'AECD: {aecd[0]:.3f}, {aecd[1]:.3f} s / b'

    cv2.putText(image,
                str_ear,
                (10, 20),
                font,
                font_size,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA)

    cv2.putText(image,
                str_freq,
                (10, 40),
                font,
                font_size,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA)

    cv2.putText(image,
                str_dur,
                (10, 60),
                font,
                font_size,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA)

    cv2.putText(image,
                str_aecd,
                (10, 80),
                font,
                font_size,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA)

    return image


def process_video(video_path,
                  plotter,
                  model_name='mp_face',
                  retinaface_backbone='resnet50',
                  dataset='wflw',
                  init_fps=60,
                  max_ears_cnt=3,
                  aes_cnt=400,
                  cnt=None,
                  plot_landmarks=False,
                  print_ear=True,
                  print_aes=('left', 'right'),
                  use_cpu=False):
    global TOTAL_TIME
    global RETINAFACE_TIME
    global SPIGA_TIME
    global ALGO_TIME

    # model
    if model_name == 'mp_face':
        # mediapipe face mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_model = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        model = (model_name, face_model)
    elif model_name == 'spiga':
        # spiga
        processor = SPIGAFramework(ModelConfig(dataset), use_cpu=use_cpu)

        retinaface_weights = './retinaface_pytorch/weights/'
        if retinaface_backbone == 'resnet50':
            retinaface_weights += 'Resnet50_Final.pth'
        elif retinaface_backbone == 'mobile0.25':
            retinaface_weights += 'mobilenet0.25_Final.pth'

        retinaface_net = retinaface_model(network=retinaface_backbone,
                                          weights=retinaface_weights)
        model = (model_name, processor, retinaface_net)

    # ear
    lears, rears = [], []

    # list with blink speeds
    lspd, rspd = [], []

    # max ear
    lmax, rmax = 0.0, 0.0

    # min ear
    lmin, rmin = 1.0, 1.0

    # list with blink times
    lt, rt = [], []

    # start time
    lst, rst = 0, 0

    # number of intervals when blinking
    lcnt, rcnt = 0, 0

    # number of frames when blinking
    lf, rf = 0, 0

    # stage after closing
    lopen, ropen = False, False

    # frequency array
    lfrs, rfrs = [0.0] * (aes_cnt + 1), [0.0] * (aes_cnt + 1)

    # duration array
    ldurs, rdurs = [0.0] * (aes_cnt + 1), [0.0] * (aes_cnt + 1)

    # aecds array
    laecds, raecds = [0.0] * (aes_cnt + 1), [0.0] * (aes_cnt + 1)

    # all facial landmarks
    canvases = []

    nums = 0
    mean_fps = 0.0

    video_cap = cv2.VideoCapture(video_path)
    while (video_cap.isOpened()):
        ret, frame = video_cap.read()
        if ret:
            # calculate ear for each eye
            prev_frame_time = time.time()

            lear, rear, canvas = process_image(nums,
                                               frame,
                                               plotter,
                                               model=model,
                                               plot=plot_landmarks,
                                               print_ear=print_ear)

            # ears for aes calculating stage
            lears.append(lear)
            rears.append(rear)

            start = time.time()

            # aes calculating stage
            if nums == aes_cnt:
                lprep = calculate_aes(lears, max_ears_cnt, show=print_aes[0])
                rprep = calculate_aes(rears, max_ears_cnt, show=print_aes[1])

                laes, lmax_thld, lmin_thld = lprep
                raes, rmax_thld, rmin_thld = rprep

            # fatigue detection stage
            if nums > aes_cnt:
                # tuple with different calculated parameters for left eye
                lfat = calc_fatigue(nums,
                                    lt,
                                    lst,
                                    lear,
                                    lmax_thld,
                                    lmin_thld,
                                    lmax,
                                    lmin,
                                    lspd,
                                    lf,
                                    lcnt,
                                    lopen)
                lt, lst, lmax, lmin, lspd, lf, lcnt, lopen = lfat

                # tuple with different calculated parameters for right eye
                rfat = calc_fatigue(nums,
                                    rt,
                                    rst,
                                    rear,
                                    rmax_thld,
                                    rmin_thld,
                                    rmax,
                                    rmin,
                                    rspd,
                                    rf,
                                    rcnt,
                                    ropen)
                rt, rst, rmax, rmin, rspd, rf, rcnt, ropen = rfat

                # calculate frequencies without aes stage
                lfr = lcnt / (nums - aes_cnt) * init_fps
                rfr = rcnt / (nums - aes_cnt) * init_fps

                lfrs.append(lfr)
                rfrs.append(rfr)

                # calculate durations
                ldur, rdur = lf / init_fps, rf / init_fps
                ldurs.append(ldur)
                rdurs.append(rdur)

                # calculate aecds
                laecd = ldur / lcnt if lcnt != 0 else 0.0
                raecd = rdur / rcnt if rcnt != 0 else 0.0
                laecds.append(laecd)
                raecds.append(raecd)

                # put parameters in canvas
                if canvas is not None:
                    canvas = display_values(canvas,
                                            (lear, rear),
                                            (lfr, rfr),
                                            (ldur, rdur),
                                            (laecd, raecd))

            end = time.time()

            ALGO_TIME += end - start
            TOTAL_TIME += end - start

            new_frame_time = time.time()
            fps = 1.0 / (new_frame_time - prev_frame_time)
            mean_fps += fps

            nums += 1
            print(f'processed {nums} frames')

            if canvas is not None:
                canvases.append(canvas)
        else:
            break

    print(f'fps {(mean_fps / nums):.2f}')

    start = time.time()

    # video ending processing
    lt, lst, lmax, lmin, lspd, lf, lcnt = video_end_fatigue(nums,
                                                            lt,
                                                            lst,
                                                            lmax,
                                                            lmin,
                                                            lspd,
                                                            lf,
                                                            lcnt)

    rt, rst, rmax, rmin, rspd, rf, rcnt = video_end_fatigue(nums,
                                                            rt,
                                                            rst,
                                                            rmax,
                                                            rmin,
                                                            rspd,
                                                            rf,
                                                            rcnt)

    end = time.time()

    ALGO_TIME += end - start
    TOTAL_TIME += end - start

    # concatenate left and right metrics
    speeds = (lspd, rspd)
    frames = (lf, rf)
    cnts = (lcnt, rcnt)
    times = (lt, rt)
    ears = (lears, rears)
    frequencies = (lfrs, rfrs)
    durations = (ldurs, rdurs)
    aecds = (laecds, raecds)

    # output tuple
    out = (speeds,
           frames,
           cnts,
           times,
           canvases,
           frequencies,
           durations,
           ears,
           aecds)

    return out


# from [(last_idx, duration), (last_idx, duration), ..., (last_idx, duration)]
# to [idx, idx, ..., idx]
def blink_frames(times):
    blinks_idx = []

    for dur, idx in times:
        if dur > 0:
            blinks_idx.extend([i for i in range(idx, idx - dur, -1)])

    return sorted(blinks_idx)


# save landmarks with detected blinks
def procces_frames_into_video(times,
                              canvases,
                              name='blinking.mp4',
                              init_fps=60.0,
                              alpha=0.2,
                              lcolor=(0, 200, 0),
                              rcolor=(200, 0, 0),
                              bcolor=(200, 200, 0)):
    assert len(canvases) > 0
    assert init_fps > 0.0
    assert alpha > 0.0 and alpha < 1.0

    h, w, _ = canvases[0].shape

    lidx = blink_frames(times[0])
    ridx = blink_frames(times[1])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(name, fourcc, init_fps, (w, h))

    for i, canvas in enumerate(canvases):
        overlay = canvas.copy()
        h, w, _ = canvas.shape

        start_point = (int(0), int(0))
        end_point = (int(w), int(h))

        if i in lidx and i not in ridx:
            cv2.rectangle(overlay,
                          pt1=start_point,
                          pt2=end_point,
                          color=lcolor,
                          thickness=-1)
            canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
        elif i not in lidx and i in ridx:
            cv2.rectangle(overlay,
                          pt1=start_point,
                          pt2=end_point,
                          color=rcolor,
                          thickness=-1)
            canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
        elif i in lidx and i in ridx:
            cv2.rectangle(overlay,
                          pt1=start_point,
                          pt2=end_point,
                          color=bcolor,
                          thickness=-1)
            canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

        out.write(canvas)

    out.release()


def plot_graphics(name,
                  frequencies,
                  durations,
                  ears,
                  aecds,
                  init_fps=60.0,
                  prefix=('левый', 'правый'),
                  show=False,
                  grid=True,
                  num_xticks=10,
                  num_yticks=10):
    assert len(frequencies[0]) == len(durations[0])
    assert len(frequencies[0]) == len(ears[0])
    assert len(durations[0]) == len(ears[0])
    assert len(frequencies[1]) == len(durations[1])
    assert len(frequencies[1]) == len(ears[1])
    assert len(durations[1]) == len(ears[1])

    num = len(ears[0])
    sec = num / init_fps

    x = np.linspace(0.0, sec, num)
    x_max = x[-1]

    left_ear_max = np.max(ears[0])
    right_ear_max = np.max(ears[1])

    left_freq_max = np.max(frequencies[0])
    right_freq_max = np.max(frequencies[1])

    left_dur_max = np.max(durations[0])
    right_dur_max = np.max(durations[1])

    left_aecd_max = np.max(aecds[0])
    right_aecd_max = np.max(aecds[1])

    x_ticks_val = np.linspace(0.0, x_max, num=num_xticks)
    x_ticks_label = [f'{el:.2f}' for el in x_ticks_val]

    y_ticks_ear_left_val = np.linspace(0.0, left_ear_max, num=num_yticks)
    y_ticks_ear_left_label = [f'{el:.2f}' for el in y_ticks_ear_left_val]

    y_ticks_ear_right_val = np.linspace(0.0, right_ear_max, num=num_yticks)
    y_ticks_ear_right_label = [f'{el:.2f}' for el in y_ticks_ear_right_val]

    y_ticks_freq_left_val = np.linspace(0.0, left_freq_max, num=num_yticks)
    y_ticks_freq_left_label = [f'{el:.2f}' for el in y_ticks_freq_left_val]

    y_ticks_freq_right_val = np.linspace(0.0, right_freq_max, num=num_yticks)
    y_ticks_freq_right_label = [f'{el:.2f}' for el in y_ticks_freq_right_val]

    y_ticks_dur_left_val = np.linspace(0.0, left_dur_max, num=num_yticks)
    y_ticks_dur_left_label = [f'{el:.2f}' for el in y_ticks_dur_left_val]

    y_ticks_dur_right_val = np.linspace(0.0, right_dur_max, num=num_yticks)
    y_ticks_dur_right_label = [f'{el:.2f}' for el in y_ticks_dur_right_val]

    y_ticks_aecd_left_val = np.linspace(0.0, left_aecd_max, num=num_yticks)
    y_ticks_aecd_left_label = [f'{el:.2f}' for el in y_ticks_aecd_left_val]

    y_ticks_aecd_right_val = np.linspace(0.0, right_aecd_max, num=num_yticks)
    y_ticks_aecd_right_label = [f'{el:.2f}' for el in y_ticks_aecd_right_val]

    # create cunvas with 6 subplots
    _, ((ax11, ax12, ax13, ax14),
        (ax21, ax22, ax23, ax24)) = plt.subplots(2,
                                                 4,
                                                 figsize=(10, 5),
                                                 layout='constrained')

    # plot ears
    ax11.set_title(f'{prefix[0]} EAR')
    ax11.set_xlim(0, x_max)
    ax11.set_ylim(0, left_ear_max)
    ax11.grid(grid)
    ax11.plot(x, ears[0])
    ax11.set_xticks(x_ticks_val, x_ticks_label, rotation=90)
    ax11.set_yticks(y_ticks_ear_left_val, y_ticks_ear_left_label)

    ax21.set_title(f'{prefix[1]} EAR')
    ax21.set_xlim(0, x_max)
    ax21.set_ylim(0, right_ear_max)
    ax21.grid(grid)
    ax21.plot(x, ears[1])
    ax21.set_xticks(x_ticks_val, x_ticks_label, rotation=90)
    ax21.set_yticks(y_ticks_ear_right_val, y_ticks_ear_right_label)

    # plot frs
    ax12.set_title(f'{prefix[0]} FREQ')
    ax12.set_xlim(0, x_max)
    ax12.set_ylim(0, left_freq_max)
    ax12.grid(grid)
    ax12.plot(x, frequencies[0])
    ax12.set_xticks(x_ticks_val, x_ticks_label, rotation=90)
    ax12.set_yticks(y_ticks_freq_left_val, y_ticks_freq_left_label)

    ax22.set_title(f'{prefix[1]} FREQ')
    ax22.set_xlim(0, x_max)
    ax22.set_ylim(0, right_freq_max)
    ax22.grid(grid)
    ax22.plot(x, frequencies[1])
    ax22.set_xticks(x_ticks_val, x_ticks_label, rotation=90)
    ax22.set_yticks(y_ticks_freq_right_val, y_ticks_freq_right_label)

    # plot durs
    ax13.set_title(f'{prefix[0]} DUR')
    ax13.set_xlim(0, x_max)
    ax13.set_ylim(0, left_dur_max)
    ax13.grid(grid)
    ax13.plot(x, durations[0])
    ax13.set_xticks(x_ticks_val, x_ticks_label, rotation=90)
    ax13.set_yticks(y_ticks_dur_left_val, y_ticks_dur_left_label)

    ax23.set_title(f'{prefix[1]} DUR')
    ax23.set_xlim(0, x_max)
    ax23.set_ylim(0, right_dur_max)
    ax23.grid(grid)
    ax23.plot(x, durations[1])
    ax23.set_xticks(x_ticks_val, x_ticks_label, rotation=90)
    ax23.set_yticks(y_ticks_dur_right_val, y_ticks_dur_right_label)

    # plot aecds
    ax14.set_title(f'{prefix[0]} AECD')
    ax14.set_xlim(0, x_max)
    ax14.set_ylim(0, left_aecd_max)
    ax14.grid(grid)
    ax14.plot(x, aecds[0])
    ax14.set_xticks(x_ticks_val, x_ticks_label, rotation=90)
    ax14.set_yticks(y_ticks_aecd_left_val, y_ticks_aecd_left_label)

    ax24.set_title(f'{prefix[1]} AECD')
    ax24.set_xlim(0, x_max)
    ax24.set_ylim(0, right_aecd_max)
    ax24.grid(grid)
    ax24.plot(x, aecds[1])
    ax24.set_xticks(x_ticks_val, x_ticks_label, rotation=90)
    ax24.set_yticks(y_ticks_aecd_right_val, y_ticks_aecd_right_label)

    # save graphics
    plt.savefig(name)

    # show graphics
    if show:
        plt.show()


def parse():
    parser = ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='./config_algo.yaml')
    args = parser.parse_args()
    return args


def parse_yaml(file):
    with open(file) as f:
        my_dict = yaml.safe_load(f)
    return my_dict


def main():
    args = parse()
    config = parse_yaml(args.config)

    prefix = config['prefix']
    model_name = config['model_name']
    backbone = config['retinaface_backbone']
    input_path = prefix + config['input_video']
    output_path = prefix + f'/{model_name}_{config["output_video"]}'
    graphics_path = prefix + f'/{model_name}_{config["graphics"]}'
    use_cpu = config['use_cpu']

    plotter = Plotter()

    # iterate over frames
    metrics = process_video(input_path,
                            plotter,
                            model_name=model_name,
                            retinaface_backbone=backbone,
                            max_ears_cnt=4,
                            aes_cnt=config['fps']*5,
                            init_fps=config['fps'],
                            cnt=None,
                            plot_landmarks=True,
                            print_ear=False,
                            print_aes=(None, None),
                            use_cpu=use_cpu)
    _, _, _, times, canvases, frequencies, durations, ears, aecds = metrics

    print(f'left frequency {frequencies[0][-1]:.3f}')
    print(f'left duration {durations[0][-1]:.3f}')
    print(f'left aecd {aecds[0][-1]:.3f}')

    print(f'right frequency {frequencies[1][-1]:.3f}')
    print(f'right duration {durations[1][-1]:.3f}')
    print(f'right aecd {aecds[1][-1]:.3f}')

    if model_name == 'spiga':
        ptime = RETINAFACE_TIME
        pperc = ptime / TOTAL_TIME * 100
        pstr = f'retinaface time: {ptime:.3f}, retinaface percent: {pperc:.2f}'
        print(pstr)

        ptime = SPIGA_TIME
        pperc = ptime / TOTAL_TIME * 100
        pstr = f'spiga time: {ptime:.3f}, spiga percent: {pperc:.2f}'
        print(pstr)

        ptime = ALGO_TIME
        pperc = ptime / TOTAL_TIME * 100
        pstr = f'algo time: {ptime:.3f}, algo percent: {pperc:.2f}'
        print(pstr)

    # glue it into a video
    if len(canvases) > 0:
        procces_frames_into_video(times,
                                  canvases,
                                  name=output_path,
                                  init_fps=config['fps'])

    # plot graphics
    plot_graphics(graphics_path,
                  frequencies,
                  durations,
                  ears,
                  aecds,
                  init_fps=config['fps'])


if __name__ == '__main__':
    main()
