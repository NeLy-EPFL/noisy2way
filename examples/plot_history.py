from matplotlib import pyplot as plt

import noisy2way

history = noisy2way.load_history("/mnt/internal_hdd/aymanns/181220_Rpr_R57C10_GC6s_tdTom/Fly5/concatenated/2p/model_red/history.pkl")
fig = noisy2way.plot_loss(history)
fig.savefig('/mnt/internal_hdd/aymanns/181220_Rpr_R57C10_GC6s_tdTom/Fly5/concatenated/2p/model_red/history.pdf')
