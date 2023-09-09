from inference_algo import process_video
import yaml
from argparse import ArgumentParser
from spiga.demo.visualize.plotter import Plotter


def parse():
    parser = ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='./config_algo_fatigue.yaml')
    args = parser.parse_args()
    return args


def parse_yaml(file):
    with open(file) as f:
        my_dict = yaml.safe_load(f)
    return my_dict


def find_fatigue(frequencies, aecds, freq_threshold, aecd_threshold):
    fatigue = []
    zip_vals = zip(frequencies[0], frequencies[1], aecds[0], aecds[1])
    for left_freq, right_freq, left_aecd, right_aecd in zip_vals:
        left_cond = left_freq <= freq_threshold \
            or left_aecd >= aecd_threshold
        right_cond = right_freq <= freq_threshold \
            or right_aecd >= aecd_threshold
        if left_cond or right_cond:
            fatigue += [1]
        else:
            fatigue += [0]
    return fatigue


def main():
    args = parse()
    config = parse_yaml(args.config)

    prefix = config['prefix']
    model_name = config['model_name']
    backbone = config['retinaface_backbone']
    input_path = prefix + config['input_video']
    use_cpu = config['use_cpu']
    freq_threshold = config['frequency_fatigue_threshold']
    aecd_threshold = config['aecd_fatigue_threshold']

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
                            plot_landmarks=False,
                            print_ear=False,
                            print_aes=(None, None),
                            use_cpu=use_cpu)
    _, _, _, _, _, frequencies, _, _, aecds = metrics

    fatigue_per_timestamp = find_fatigue(frequencies,
                                         aecds,
                                         freq_threshold,
                                         aecd_threshold)
    mean_fatigue = sum(fatigue_per_timestamp) / len(fatigue_per_timestamp)

    return mean_fatigue, fatigue_per_timestamp


if __name__ == '__main__':
    mean_fatigue, fatigue_per_timestamp = main()
    print(f'fatigue per timestamp: {fatigue_per_timestamp}')
    print(f'mean fatigue: {mean_fatigue:.3f}')
