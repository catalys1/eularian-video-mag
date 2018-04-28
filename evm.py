import imageio
import cv2 as cv
import numpy as np
from scipy.signal import butter, lfilter
import time


def make_gaussian_pyramid(image, levels):
    '''Construct a gaussian pyramid from the input image.

    Parameters
    ----------
    volume : 2D or 3D array
        Either a grayscale or RGB image
    levels : int
        Number of levels for the output pyramid

    Returns
    -------
    guass_pyramid : list of arrays
        A list of length `levels` of 2D or 3D arrays. The array at level l has
        height N / 2^l and width M / 2^l for l in [0, `levels`-1]
    '''
    pdown = cv.pyrDown

    pyr = [image]
    for l in range(levels):
        d = pdown(image)
        pyr.append(d)
    return pyr


def make_laplacian_pyramid(image, levels):
    '''Construct a laplacian pyramid from the input image.
    TODO: this function changed, redo docs

    Parameters
    ----------
    volume : 3D array
        A 3 channel image
    levels : int
        Number of levels for the output pyramid

    Returns
    -------
    laplace_pyramid : list of arrays
        A list of length `levels` of 2D or 3D arrays. The array at level l has
        height N / 2^l and width M / 2^l for l in [0, `levels`-1]
    '''
    pdown = cv.pyrDown
    pup = cv.pyrUp

    h, w = image.shape[:2]
    inds = np.cumsum([0] + [w * h // (2**(2 * i)) for i in range(levels)])
    pyr = np.empty((inds[-1], 3), dtype='float32')
    img = image
    for l in range(levels - 1):
        d = pdown(img)
        bp = img - pup(d)
        pyr[inds[l]:inds[l + 1]] = bp.reshape(-1, 3)
        img = d
    pyr[inds[-2]:inds[-1]] = img.reshape(-1, 3)
    return pyr, inds
    # pyr = []
    # img = image
    # for l in range(levels - 1):
    #     d = pdown(img)
    #     bp = img - pup(d)
    #     pyr.append(bp)
    #     img = d
    # pyr.append(img)


def collapse_laplacian_pyramid(pyr, inds, width, height):
    '''Collapse a laplacian pyramid back into a single image.
    TODO: this function changed, redo docs

    Parameters
    ----------
    pyr : list of arrays
        A list of either 2D or 3D arrays, each representing a level in the
        laplacian pyramid. `pyr`[0] is bottom level of the pyramid (highest
        frequency content and largest image)

    Returns
    -------
    image : 2D or 3D array
        A single image, reconstructed from the laplacian pyramid
    '''
    pup = cv.pyrUp

    reduction = 2**(len(inds) - 2)
    w, h = width // reduction, height // reduction
    result = pup(pyr[inds[-2]:inds[-1]].reshape(h, w, 3))
    for i in range(len(inds) - 2, 1, -1):
        w = w * 2
        h = h * 2
        result = pup(result + pyr[inds[i - 1]:inds[i]].reshape(h, w, 3))
    result += pyr[0:inds[1]].reshape(h * 2, w * 2, 3)
    return result
    # result = pup(pyr[-1])
    # for i in pyr[-2:0:-1]:
    #     result = pup(result + i)
    # result += pyr[0]


def iir_temporal_filter(Fs, w1, w2, filter_type='butter'):
    '''TODO: Documentation
    '''
    if filter_type == 'butter':
        b, a = butter(2, [2 * w1 / Fs, 2 * w2 / Fs], 'bandpass')
        b = b.astype('float32')
        a = a.astype('float32')

        def filt(x, zi=None):
            if zi is None:
                zi = np.zeros(
                    (max(len(a), len(b)) - 1, *x.shape), dtype='float32')
                # zi = np.empty(
                #     (max(len(a), len(b)) - 1, *x.shape), dtype='float32')
                # zi[:] = x

            if x.ndim == 2:
                x = [x]
            y, zi = lfilter(b, a, x, axis=0, zi=zi)
            return y, zi

        return filt

    elif filter_type == 'cascade':
        temp = 2 - np.cos(w1 * 2 * np.pi)
        a_low = temp - np.sqrt(temp**2 - 1)
        temp = 2 - np.cos(w2 * 2 * np.pi)
        a_high = temp - np.sqrt(temp**2 - 1)

        def filt(x, zi=None):
            if zi is None:
                zi = np.zeros((2, *x.shape))
            low = (1 - a_low) * zi[0] + a_low * x
            high = (1 - a_high) * zi[1] + a_high * x
            zi[0] = low
            zi[1] = high
            y = high - low
            return y, zi

        return filt


def convert_color_space(image):
    # image = cv.cvtColor(image, cv.COLOR_RGB2YUV, image)
    # image = cv.cvtColor(image, cv.COLOR_RGB2YCrCb, image)
    # image = cv.cvtColor(image, cv.COLOR_RGB2LAB, image)
    return image


def revert_color_space(image):
    # image = cv.cvtColor(image, cv.COLOR_YUV2RGB, image)
    # image = cv.cvtColor(image, cv.COLOR_LAB2YCrCb, image)
    # image = cv.cvtColor(image, cv.COLOR_LAB2RGB, image)
    return image


def process_video(source, dest):

    vid_in = imageio.get_reader(source)
    metadata = vid_in.get_meta_data()
    vid_out = imageio.get_writer(dest, fps=metadata['fps'])

    width, height = metadata['size']

    print(f'{source}: width={width}, height={height}')

    Fs = metadata['fps']  # Frame rate / sampling frequency
    w1 = 1.5  # lower cutoff frequency in Hz
    w2 = 2.5  # upper cutoff frequency in Hz

    temporal_filter = iir_temporal_filter(Fs, w1, w2, 'butter')
    zi = None

    pyr_size = 5
    alpha = 40
    attenuate = 0.1

    # amplifier = np.array([alpha] * 3) * [1, attenuate, attenuate]

    times = []
    i = 0
    for frame in vid_in:
        start_time = time.time()
        frame = frame * np.float32(1 / 255)
        frame = convert_color_space(frame)
        pyramid, index = make_laplacian_pyramid(frame, pyr_size)
        y = np.zeros_like(pyramid)
        # rng = slice(index[1], index[-2])
        rng = slice(index[1], index[-2])
        y[rng], zi = temporal_filter(pyramid[rng], zi)
        y = y.squeeze()
        # for i in range(1, pyr_size):
        y[rng] *= alpha
        filtered = collapse_laplacian_pyramid(y, index, width, height)
        # filtered[..., 1:] *= attenuate
        result = frame + filtered
        result = result.clip(0.0, 1.0)
        result = revert_color_space(result)
        result = (result * 255).astype('uint8')
        end_time = time.time()
        times.append(end_time - start_time)
        vid_out.append_data(result, {'fps': Fs})
        i += 1
        print(f'\r{i:3d}: {times[-1]:.4f}', end='')
    print('')
    print(f'Average time: {np.mean(times)}')
    print(f'Max time: {np.max(times)}')
    print(f'Min time: {np.min(times)}')
    print(f'Total time: {np.sum(times)}')

    np.save('filter.npy', zi)


def process_freq():

    vid = 'me.mov'
    V = imageio.get_reader(vid)
    metadata = V.get_meta_data()
    vid_out = imageio.get_writer('mag_freq.mp4', fps=metadata['fps'])

    width, height = metadata['size']

    print(f'{vid}: width={width}, height={height}')

    Fs = metadata['fps']  # Frame rate / sampling frequency
    w1 = 0.7  # lower cutoff frequency in Hz
    w2 = 1.1  # upper cutoff frequency in Hz

    alpha = 60

    # total_frames = V.get_length()
    window_size = 128
    # overlap = window_size / 2

    bin_resolution = Fs / window_size
    lower_bin = int(np.ceil(w1 / bin_resolution))
    upper_bin = int(np.rint(w2 / bin_resolution))

    volume = np.empty((height, width, 3, window_size), 'float32')

    times = []
    for i in range(window_size):
        volume[..., i] = V.get_next_data()
        print('Frame ', i)

    while True:
        start_time = time.time()
        # short-time fourier transform, extract signal, amplify and add in
        freq = np.fft.fft(volume, axis=-1)
        freq[..., :lower_bin] = 0
        freq[..., upper_bin + 1:] = 0
        band_passed = np.fft.ifft(freq, axis=-1)
        volume += (alpha * band_passed)

        end_time = time.time()
        times.append(end_time - start_time)
        print(f'\r{i:3d}: {times[-1]:.4f}', end='')
        # write out frames
        for i in range(volume.shape[-1]):
            vid_out.append_data(volume[..., i])
        # get next block of frames
        for i in range(window_size):
            try:
                volume[..., i] = V.get_next_data()
            except IndexError:
                break
        if i == 0:
            break

    print('')
    print(f'Average time: {np.mean(times)}')
    print(f'Max time: {np.max(times)}')
    print(f'Min time: {np.min(times)}')
    print(f'Total time: {np.sum(times)}')


# TODO
# X Add temporal filtering and amplification to all pyramid levels, not just
#   the top level
#
# - Switch to YIQ color space?
#
# - Add support for different types of filters:
# x butterworth
# x cascaded first order iir?
# * ideal? (this would require going into the frequency domain)
#
# - Two modes: live and pre-recorded
# * In live mode, the video is processed frame by frame as they are received,
#   showing both the incoming and filtered frames
# * In pre-recorded mode, the video is processed (in frame batches?) from disk.
#
# - UI: display and parameter choices
#
# - Modularize the code
#
# - Implement a version that can be run on the GPU? PyTorch?


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='Source video file')
    parser.add_argument('dest', help='Path to destination video file')
    args = parser.parse_args()
    process_video(args.source, args.dest)
    # process_freq()


if __name__ == '__main__':
    main()
