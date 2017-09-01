import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from trash_cache import TrashCache

stim_type = "natural_scenes"
image = 50
if stim_type == 'spont':
    manifest_path = '/media/charlie/Brain2017/data/dynamic-brain-workshop/trash_cache/spontPCA/'
elif stim_type == 'natural_scenes':
    manifest_path = os.path.join('/media/charlie/Brain2017/data/dynamic-brain-workshop/trash_cache/nsPCA/', str(image))

tc = TrashCache(os.path.join(manifest_path, 'trash_cache_manifest.json'))
exps = os.listdir(os.path.join(manifest_path, 'exps'))
# create plot of variance explained across brain regions
fr_pm = []
fr_p = []
fr_rl = []
fr_al = []
fr_am = []
fr_l = []
p_corr_VISpm = []
p_corr_VISp = []
p_corr_VISrl = []
p_corr_VISal = []
p_corr_VISam = []
p_corr_VISl = []
r_corr_VISpm = []
r_corr_VISp = []
r_corr_VISrl = []
r_corr_VISal = []
r_corr_VISam = []
r_corr_VISl = []
plt.figure()
for exp in exps:
    spontPCA_data = tc.load_experiments([exp])
    #print(spontPCA_data[0].keys())
    target_area = spontPCA_data[0]['meta_data']['targeted_structure']
    print(spontPCA_data[0]['corr_mat'].T.keys())
    if  target_area == 'VISpm':
        fr_pm.append(spontPCA_data[0]['Fraction of PCs'])
        p_corr_VISpm.append(max(spontPCA_data[0]['corr_mat'].T['pupil smooth']))
        r_corr_VISpm.append(max(spontPCA_data[0]['corr_mat'].T['running speed smooth']))
        x = np.linspace(0,1, len(spontPCA_data[0]['var_explained']))
        plt.plot(x, spontPCA_data[0]['var_explained'], 'r')
    elif target_area == 'VISp':
        fr_p.append(spontPCA_data[0]['Fraction of PCs'])
        p_corr_VISp.append(max(spontPCA_data[0]['corr_mat'].T['pupil smooth']))
        r_corr_VISp.append(max(spontPCA_data[0]['corr_mat'].T['running speed smooth']))
        x = np.linspace(0,1, len(spontPCA_data[0]['var_explained']))
        plt.plot(x, spontPCA_data[0]['var_explained'], 'y')
    elif target_area == 'VISrl':
        fr_rl.append(spontPCA_data[0]['Fraction of PCs'])
        p_corr_VISrl.append(max(spontPCA_data[0]['corr_mat'].T['pupil smooth']))
        r_corr_VISrl.append(max(spontPCA_data[0]['corr_mat'].T['running speed smooth']))
        x = np.linspace(0,1, len(spontPCA_data[0]['var_explained']))
        plt.plot(x, spontPCA_data[0]['var_explained'], 'g')
    elif target_area == 'VISal':
        fr_al.append(spontPCA_data[0]['Fraction of PCs'])
        p_corr_VISal.append(max(spontPCA_data[0]['corr_mat'].T['pupil smooth']))
        r_corr_VISal.append(max(spontPCA_data[0]['corr_mat'].T['running speed smooth']))
        x = np.linspace(0,1, len(spontPCA_data[0]['var_explained']))
        plt.plot(x, spontPCA_data[0]['var_explained'], 'p')
    elif target_area == 'VISam':
        fr_am.append(spontPCA_data[0]['Fraction of PCs'])
        p_corr_VISam.append(max(spontPCA_data[0]['corr_mat'].T['pupil smooth']))
        r_corr_VISam.append(max(spontPCA_data[0]['corr_mat'].T['running speed smooth']))
        x = np.linspace(0,1, len(spontPCA_data[0]['var_explained']))
        plt.plot(x, spontPCA_data[0]['var_explained'], 'k')
    elif target_area == 'VISl':
        fr_l.append(spontPCA_data[0]['Fraction of PCs'])
        p_corr_VISl.append(max(spontPCA_data[0]['corr_mat'].T['pupil smooth']))
        r_corr_VISl.append(max(spontPCA_data[0]['corr_mat'].T['running speed smooth']))
        x = np.linspace(0,1, len(spontPCA_data[0]['var_explained']))
        plt.plot(x, spontPCA_data[0]['var_explained'], 'b')

plt.xlabel('Fraction of Dimensions')
plt.ylabel('Variance explained')


data = [fr_p, fr_l, fr_al, fr_pm, fr_am, fr_rl]
plt.figure()
plt.boxplot(data)
plt.title('Fraction of dimensions need to explain 50 percent of variance')
plt.ylabel('Fraction of dimensions')
plt.xticks([1,2,3,4,5,6], ['VISp','VISl','VISal','VISpm','VISam','VISrl'])

data = [p_corr_VISp, p_corr_VISl, p_corr_VISal, p_corr_VISpm, p_corr_VISam, p_corr_VISrl]
plt.figure()
plt.boxplot(data)
plt.title('PC correlation with pupil area')
plt.ylabel('Pearson correlation coefficient')
plt.xticks([1,2,3,4,5,6], ['VISp','VISl','VISal','VISpm','VISam','VISrl'])


data = [r_corr_VISp, r_corr_VISl, r_corr_VISal, r_corr_VISpm, r_corr_VISam, r_corr_VISrl]
plt.figure()
plt.boxplot(data)
plt.title('PC correlation with running speed')
plt.ylabel('Pearson correlation coefficient')
plt.xticks([1,2,3,4,5,6], ['VISp','VISl','VISal','VISpm','VISam','VISrl'])
plt.show()
