#!/usr/bin/python
from mars.pipeline import load_data
from os.path import expanduser
from sklearn.cross_decomposition import PLSRegression
from sklearn.cross_validation import cross_val_score

import numpy as np


def match_chan_line(chans,lines,buffsize):
    lines_ind = []
    for line  in lines:
        peak = np.abs(chans-line).argmin()
        mini = max(peak-buffsize,0)
        maxi = min(peak+buffsize,len(chans)-1)
        lines_ind.append([mini,maxi])
    return lines_ind


def run_ratio_cv(model,numer,denom,features,labels,n_bins=5):
    X = features[:,numer].sum(1)/features[:,denom].sum(1)
    order = np.argsort(X)
    srt_X = features[order,1:] # use the whole spectrum, except the wonky first channel
    srt_labels = labels[order]
    X_bins = np.array_split(srt_X, n_bins)
    label_bins = np.array_split(srt_labels, n_bins)
    scores = []
    for X,label in zip(X_bins,label_bins):
        s = np.mean(cross_val_score(model,X,label,scoring='mean_squared_error'))
        scores.append(s)
    return np.mean(scores), np.std(scores)


if __name__ == '__main__':
    # (lazy) SETTINGS
    element = 'SiO2'
    buffsize = 5 # no. channels on each side of peak
    chans = np.genfromtxt('libs/data/prepro_wavelengths.csv')
    lines = np.load(expanduser('~/Data/si_lines.npz'))
    model = PLSRegression(n_components=10,scale=False)
    features,labels = load_data(masked=False,norm=3)

    labels = labels[element] # one element at a time for now

    si1 = lines['si1']
    si2 = lines['si2']
    si1_ind = match_chan_line(chans,si1,buffsize)
    si2_ind = match_chan_line(chans,si2,buffsize)

    log = []
    scores = []
    stds = []
    for i,l1_ind in enumerate(si1_ind):
        for j,l2_ind in enumerate(si2_ind):
            log.append("%.4f / %.4f (ind: %d/%d)" % (si1[i],si2[j],i,j))
            scr,std = run_ratio_cv(model,l1_ind,l2_ind,features,labels)
            scores.append(scr)
            stds.append(std)

            # do the inverse too
            log.append("%.4f / %.4f (ind: %d/%d)" % (si2[j],si1[i],j,i))
            scr,std = run_ratio_cv(model,l2_ind,l1_ind,features,labels)
            scores.append(scr)
            stds.append(std)
        np.savez(element,scores=np.array(scores),stds=np.array(stds),log=np.array(log))
