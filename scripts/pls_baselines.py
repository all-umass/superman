#!/usr/bin/python
import csv
import matplotlib.pyplot as plt
import numpy as np
import os

from mars.data import norm3
from sklearn.cross_decomposition import PLSRegression
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import mean_squared_error

from superman.baseline import (
    als_baseline, airpls_baseline, fabc_baseline, mario_baseline,
    kajfosz_kwiatek_baseline)


def load_data(elem):
    wavelengths = np.loadtxt(os.path.expanduser(
        '~/Mars/Data/mhc/raw/wavelengths.csv'))
    spectra_file = os.path.expanduser('~/Mars/Data/mhc/raw/raw_mhc.csv')
    snames = ['M1A']
    spectra = []
    locations = []
    with open(spectra_file, 'rb') as csvfile:
        csvreader = csv.reader(csvfile,delimiter=',')
        for row in csvreader:
            if row[0] != snames[-1]:
                spectra.append(np.mean(locations,0))
                snames.append(row[0])
                locations = []
            locations.append([float(i) for i in row[1:]])
    spectra.append(np.mean(locations,0))
    spectra = np.array(spectra)

    file = os.path.expanduser('~/Mars/Data/mhc/processed/mastersheet.csv')
    comps = np.genfromtxt(file,delimiter=',',names=True,dtype=None)

    mask = [name not in comps['PELLET'] for name in snames]
    mask_matrix = (np.ones_like(spectra).T*mask).T
    snames = np.ma.array(snames,mask=mask).compressed()
    spectra = np.ma.array(spectra,mask=mask_matrix).compressed().reshape(
        (mask.count(False),spectra.shape[1]))

    mask = [name not in snames for name in comps['PELLET']]
    comps = np.ma.array(comps[elem],mask=mask).compressed()

    return np.array(snames),comps,spectra,wavelengths


def plot_avp(actual, predicted):
    plt.scatter(predicted,actual)
    plt.xlabel('Predicted (wt %)')
    plt.ylabel('Actual (wt %)')
    xlims = plt.xlim()
    ylims = plt.ylim()
    bottom = min(min(xlims),min(ylims))
    top = max(max(xlims),max(ylims))
    plt.plot([bottom,top],[bottom,top],ls="--",c=".3")
    plt.show()


if __name__ == '__main__':
    names,comps,data,feats=load_data('MgO')

    val_inds = np.argwhere(np.isfinite(comps)).ravel()
    names = names[val_inds]
    comps = comps[val_inds]
    data = norm3(data[val_inds])

    #data = als_baseline(data)
    #mario = mario_baseline(waves,spectra)
    #airpls = airpls_baseline(spectra)
    #fabc = fabc_baseline(spectra)
    data = kajfosz_kwiatek_baseline(feats,data)

    model = PLSRegression(n_components=10,scale=False)
    #model = Ridge(alpha=100)#LinearRegression()
    scores = []
    rmsep = []
    for train,test in KFold(len(data),9,shuffle=True):
        model.fit(data[train],comps[train])
        scores.append(model.score(data[test],comps[test]))
        rmsep.append(np.sqrt(mean_squared_error(comps[test],
                                                model.predict(data[test]))))
        #plot_avp(comps[train],model.predict(data[train]))
        #plot_avp(comps[test],model.predict(data[test]))

    print 'Score: %.3f +- %.3f' % (np.mean(scores),np.std(scores))
    print 'RMSEP: %.3f +- %.3f' % (np.mean(rmsep),np.std(rmsep))
