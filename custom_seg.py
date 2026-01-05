from PIL import Image
import numpy as np

def image_to_patches(
    image_path,
    patch_size=(128, 128),
    stride=(128, 128),
    as_array=True
):
    """
    Charge une image et la découpe en patches.

    Args:
        image_path (str): chemin vers l'image
        patch_size (tuple): (hauteur, largeur) du patch
        stride (tuple): pas de déplacement (hauteur, largeur)
        as_array (bool): retourne les patches en numpy array

    Returns:
        patches: liste ou array de patches
    """
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)

    H, W, C = img.shape
    ph, pw = patch_size
    sh, sw = stride

    patches = []
    coords = []

    for y in range(0, H - ph + 1, sh):
        for x in range(0, W - pw + 1, sw):
            patch = img[y:y+ph, x:x+pw]
            patches.append(patch)
            coords.append((y, x))


    if as_array:
        patches = np.stack(patches)

    return patches, coords



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

def get_mask(
    image_path,
    patch_size=(128, 128),
    stride=(128, 128),
    threshold=0.0
):
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

    for y in range(0, H - ph + 1, sh):
        for x in range(0, W - pw + 1, sw):
            patch = img[y:y+ph, x:x+pw]
            
            score = compute_patch_score(patch)

            if score > threshold:
                mask[y:y+ph, x:x+pw] = 1

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


def run():
    """ """

    print("Tardis")

if __name__ == "__main__":


    mask = get_mask(
        "data/Exp3.jpg",
        patch_size=(16, 16),
        stride=(16, 16),
        threshold=0.5
    )

    mask_to_image(mask, "data/mask_test.png")
