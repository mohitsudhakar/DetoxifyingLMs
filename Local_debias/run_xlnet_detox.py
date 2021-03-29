import numpy as np

from Local_debias.model import LayerXlnet
from Local_debias.debiaser import Debiaser
from Local_debias.utils.data_utils import DataUtils
from model_utils import getPretrained, plotPC

model_name = 'xlnet'

dataClass = DataUtils(model_name)

df, toxic_df, nontox_df = dataClass.readToxFile()
wsentAll, wsentTox, wsentNT = dataClass.readWordToSentFiles()
sAll, sTox, sNT = dataClass.readWordScores()
ht = dataClass.process(sAll, sTox, sNT)

tokenizer, base_model = getPretrained(model_name)

import nvsmi
orig_list = []
dup = 20
for gpu in list(nvsmi.get_gpus()):
    if gpu.mem_util < 10:   # use gpu if utilization < 10%
        orig_list.append(int(gpu.id))

gpu_list = orig_list*dup
num_gpus = len(orig_list)
layerModels = []
for g in gpu_list[:num_gpus]:
    lb = LayerXlnet(base_model, g)
    lb.eval()
    layerModels.append(lb)
layerModels *= dup

debiaser = Debiaser(ht, wsentTox, model_name, tokenizer, gpu_list, num_gpus)

ev_percent, ev = debiaser.run_group(layerModels, model_name, debias=True)
np.save(model_name+'_debias_evp.npy', ev_percent)
np.save(model_name+'_debias_ev.npy', ev)

ev_percent_normal, ev_normal = debiaser.run_group(layerModels, model_name, debias=False)
np.save(model_name+'_normal_evp.npy', ev_percent_normal)
np.save(model_name+'_normal_ev.npy', ev_normal)

# Plotting
num_pcs = 3

ev, ev_normal = np.load(model_name+'_debias_ev.npy'), np.load(model_name+'_normal_ev.npy')
title = 'Debiased '+model_name.capitalize()+' EV'
plotPC(ev, title, num_pcs)
title = 'Normal '+model_name.capitalize()+' EV'
plotPC(ev_normal, title, num_pcs)
title = 'Debiased '+model_name.capitalize()+' EV Percent'
plotPC(ev_percent, title, num_pcs)
title = 'Normal '+model_name.capitalize()+' EV Percent'
plotPC(ev_percent_normal, title, num_pcs)
