from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_patch_score(patch):
    """
    Calcule la somme des différences max-min en retirant
    itérativement les pixels min et max.

    Args:
        patch (np.ndarray): patch 2D ou 3D

    Returns:
        float: somme des différences
    """
    # Aplatir le patch (2D ou 3D)
    values = np.ravel(patch).astype(float)

    total_diff = 0.0

    # Trier une seule fois (beaucoup plus efficace)
    values = np.sort(values)

    left = 0
    right = len(values) - 1

    while left < right:
        diff = values[right] - values[left]
        total_diff += diff
        left += 1
        right -= 1

    return total_diff

def get_mask(image_path, patch_size, stride, threshold, output_folder):
    """
    Génère un masque de segmentation à partir des scores de patches.

    Args:
        image_path (str): chemin de l'image
        patch_size (tuple): taille des patches (h, w)
        stride (tuple): stride (h, w)
        threshold (float): seuil de décision

    Returns:
        np.ndarray: masque de segmentation binaire (H, W)
    """

    # Charger image
    img = np.array(Image.open(image_path).convert("L"))  # grayscale
    H, W = img.shape

    ph, pw = patch_size
    sh, sw = stride

    mask = np.zeros((H, W), dtype=np.uint8)
    score_to_count = {}

    for y in range(0, H - ph + 1, sh):
        for x in range(0, W - pw + 1, sw):
            patch = img[y:y+ph, x:x+pw]
            
            score = int(compute_patch_score(patch))
            if score in score_to_count:
                score_to_count[score] +=1
            else:
                score_to_count[score] = 1

            if score > threshold:
                mask[y:y+ph, x:x+pw] = 1

    bins = 450
    plot_scores(score_to_count, bins, output_folder)

    return mask


def mask_to_image(mask, save_path=None):
    """
    Convertit un masque binaire en image noir/blanc.

    Args:
        mask (np.ndarray): masque binaire (H, W) avec valeurs 0/1
        save_path (str, optional): chemin de sauvegarde

    Returns:
        PIL.Image
    """
    img = (mask * 255).astype(np.uint8)
    img = Image.fromarray(img, mode="L")

    if save_path is not None:
        img.save(save_path)



def plot_scores(score_count_dict, bin_size, output_folder):
    """
    Barplot à partir d'un dict {score: count} avec regroupement par bins.

    Args:
        score_count_dict (dict): {score: count}
        bin_size (int or float): largeur d'un bin
    """
    # Extraire et trier
    scores = np.array(list(score_count_dict.keys()))
    counts = np.array(list(score_count_dict.values()))

    order = np.argsort(scores)
    scores = scores[order]
    counts = counts[order]

    # Définir les bins
    min_score = scores.min()
    max_score = scores.max()
    bins = np.arange(min_score, max_score + bin_size, bin_size)

    # Agrégation par bin
    bin_counts = np.zeros(len(bins) - 1)

    for s, c in zip(scores, counts):
        idx = np.searchsorted(bins, s, side="right") - 1
        if 0 <= idx < len(bin_counts):
            bin_counts[idx] += c

    # Labels des bins
    bin_labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins) - 1)]

    # Plot
    plt.figure()
    plt.bar(bin_labels, bin_counts)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Score range")
    plt.ylabel("Count")
    plt.title("Binned score distribution")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/scores.png")
    plt.close()


def run(input_image, output_folder, score_treshold, p_size, s_size):
    """ """

    # create output folder if needed
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # compute mask
    mask = get_mask(input_image, (p_size, p_size), (s_size, s_size), score_treshold, output_folder)

    # save mask
    mask_to_image(mask, f"{output_folder}/mask.png")


if __name__ == "__main__":

    # params
    image_name = "data/Exp3.jpg"
    output_folder = "/tmp/machin"
    score_treshold = 500
    p_size = 8
    s_size = 8

    # run
    run(image_name, output_folder, score_treshold, p_size, s_size)
    

