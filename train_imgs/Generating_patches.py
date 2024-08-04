
from scipy.misc import imread, imsave

# Modify the window size and stride to control the amount of final patches.
PATCH_SIZE = 128;
PATCH_STRIDE = 128;

# Original image pairs
num_image = 200;

# Put the original images in the "IVIF_source/IR" & "IVIF_source/VIS_gray" directories.
# The augmented data will be put in the "./IR" and "./VIS_gray" directories.
patchesIR = [];
patchesVIS = [];
picidx = 0;
for idx in range(0 + 1, num_image + 1):
    print("Decomposing " + str(idx) + "-th images...");
    imageIR = imread('LLVIP_source/IR/' + str(idx) + '.png', mode='L');
    imageVIS = imread('LLVIP_source/VIS/' + str(idx) + '.png', mode='L');
    print("The shape of images are:", imageIR.shape);
    h = imageIR.shape[0];
    w = imageIR.shape[1];
    for i in range(0, h - PATCH_SIZE + 1, PATCH_STRIDE):
        for j in range(0, w - PATCH_SIZE + 1, PATCH_STRIDE):
            picidx += 1;
            patchImageIR = imageIR[i:i + PATCH_SIZE, j:j + PATCH_SIZE];
            patchImageVIS = imageVIS[i:i + PATCH_SIZE, j:j + PATCH_SIZE];
            imsave('LLVIP_patches/IR/' + str(picidx) + '.png', patchImageIR);
            imsave('LLVIP_patches/VIS/' + str(picidx) + '.png', patchImageVIS);
    print(picidx);

