from tifffile import imread, imwrite

from noisy2way import correct_2way_alignment

stack = imread(
    "/mnt/internal_hdd/aymanns/181220_Rpr_R57C10_GC6s_tdTom/Fly5/002_coronal/2p/green.tif"
)

aligned = correct_2way_alignment(
    stack,
    basedir=".",
    model_name="example_model",
    train_epochs=10,
    save_denoised_image="denoised.tif",
)

imwrite("output.tif", aligned)
