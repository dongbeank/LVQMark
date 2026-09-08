import scipy
import numpy as np

from Models.ts2vec.ts2vec import TS2Vec
from Utils.device_utils import DeviceLike, resolve_device


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def Context_FID(ori_data, generated_data, device: DeviceLike = None):
    """Context-FID between two sets of windows, via a TS2Vec encoder trained on ori_data.

    ori_data / generated_data: (N, T, D) float arrays. device: where TS2Vec runs -- a CUDA
    ordinal, a torch.device, or None (default) for the process's current device, which the
    entry point selected with --gpu. Returns a float; lower is better.

    TS2Vec is retrained on every call, so the score depends on the device: see
    evaluate.py --deterministic for what is needed to make repeated calls agree on GPU.
    """
    model = TS2Vec(
        input_dims=ori_data.shape[-1],
        device=resolve_device(device),
        batch_size=8,
        lr=0.001,
        output_dims=320,
        max_train_length=3000,
    )
    model.fit(ori_data, verbose=False)
    ori_represenation = model.encode(ori_data, encoding_window="full_series")
    gen_represenation = model.encode(generated_data, encoding_window="full_series")
    idx = np.random.permutation(ori_data.shape[0])
    ori_represenation = ori_represenation[idx]
    gen_represenation = gen_represenation[idx]
    results = calculate_fid(ori_represenation, gen_represenation)
    return results
