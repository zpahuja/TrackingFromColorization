"""
Download and preprocess videos for Kinetics dataset from Youtube.
Requires youtube-dl and ffmpeg.
"""
import os
import sys
import time
import argparse
import logging
import ujson as json

sys.path.append('.')
sys.path.append('/home/nfs/zpahuja2/tracking_from_colorization')  # vision clus
from colorizer.config import Config


def main(args):
    logging.info('args: %s', args)

    # set default kinetics dir path from config.yaml
    if args.dir is None:
        cfg = Config()
        args.dir = cfg['data_dir']['kinetics']

    kinetics_filename = os.path.join(args.dir, 'kinetics_train.json')
    if not os.path.exists(kinetics_filename):
        raise Exception('File does not exist: "%s\"' % kinetics_filename)

    # create video folders if not exists
    for foldername in ['original', 'processed']:
        if not os.path.exists(os.path.join(args.dir, foldername)):
            os.mkdir(os.path.join(args.dir, foldername))

    kinetics = json.load(open(kinetics_filename))
    keys = sorted(kinetics.keys())

    # download and/or process videos
    if not args.process:
        for i, key in enumerate(keys):
            value = kinetics[key]
            original_path = os.path.join(args.dir, 'original', key + '.mp4')
            if os.path.exists(original_path):
                logging.info('[%04d/%04d] file already exists for "%s"',
                             i, len(kinetics), key)
                continue
            try:
                logging.info('[%04d/%04d] downloading video "%s"',
                             i, len(kinetics), key)

                # download YouTube video
                command = [
                    'youtube-dl', '--quiet', '--no-warnings', '-f', 'mp4',
                    '-o', '"%s"' % original_path, '"%s"' % value['url'], '&',
                ]
                logging.info(' '.join(command))
                os.system(' '.join(command))
                time.sleep(0.5)
            except Exception as e:
                logging.error('[%04d/%04d] download failed for video "%s"',
                              i, len(kinetics), key)
                logging.error('%s: %s', type(e), str(e))

    else:
        for i, key in enumerate(keys):
            value = kinetics[key]
            original_path = os.path.join(args.dir, 'original', key + '.mp4')
            processed_path = os.path.join(args.dir, 'processed', key + '.mp4')
            if not os.path.exists(original_path):
                logging.info('[%04d/%04d] original file does not exist "%s"',
                             i, len(kinetics), key)
                continue
            if os.path.exists(processed_path):
                logging.info('[%04d/%04d] processed file already exists "%s"',
                             i, len(kinetics), key)
                continue
            try:
                logging.info('[%04d/%04d] processing video "%s"',
                             i, len(kinetics), key)

                # process video
                command = [
                    'ffmpeg', '-loglevel panic',
                    '-i', '"%s"' % original_path,
                    '-t', '%f' % value['duration'],
                    '-ss', '%f' % value['annotations']['segment'][0],
                    '-strict', '-2',
                    '"%s"' % processed_path,
                    '&'
                ]
                logging.info(' '.join(command))
                os.system(' '.join(command))
                time.sleep(1.5)
            except Exception as e:
                logging.error('[%04d/%04d] processing failed for video "%s"',
                              i, len(kinetics), key)
                logging.error('%s: %s', type(e), str(e))


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--dir', type=str, default=None)
    parser.add_argument('--process', action='store_true')
    parser.add_argument('-l', '--log', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # configure logger
    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    level = logging.DEBUG if args.debug else logging.INFO
    if not args.log:
        logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=level, format=log_format, filename=args.log)

    main(args)
