from tifffile import imread, imwrite

import noisy2way

channel = "red"
stack = imread(
    f"/mnt/internal_hdd/aymanns/181220_Rpr_R57C10_GC6s_tdTom/Fly5/concatenated/2p/{channel}.tif"
)

model = noisy2way.train(
    stack,
    "/mnt/internal_hdd/aymanns/181220_Rpr_R57C10_GC6s_tdTom/Fly5/concatenated/2p",
    f"model_{channel}",
    train_epochs=1,
)

denoised = noisy2way.predict(model, stack)

imwrite(
    f"/mnt/internal_hdd/aymanns/181220_Rpr_R57C10_GC6s_tdTom/Fly5/concatenated/2p/{channel}_denoised.tif",
    denoised,
)
