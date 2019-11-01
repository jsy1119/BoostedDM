import subprocess
import re
import pandas
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ja')

if __name__ == '__main__':
    npts = 20
    logroots = np.linspace(0.5, 5, npts)
    roots = np.power(10, logroots)
    totalsigma = np.array([])
    elasticsigma = np.array([])
    inelasticsigma = np.array([])
    completed = 1

    for energy in roots:
        process = subprocess.Popen(['cd /home/k1893416/crmcinstall/; crmc -x -o lhe -n1 -m0 -S{} -f /home/k1893416/allMyStuff/BoostedDM/crmcruns/out.lhe; rm /home/k1893416/allMyStuff/BoostedDM/crmcruns/out.lhe'.format(energy)], shell=True, stdout=subprocess.PIPE)
        stdout = process.communicate()[0]
        print('Completed {} out of {} collisions'.format(completed, npts), end='\r')
        completed += 1

        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', stdout.decode('utf-8'))

        totalsigma = np.append(totalsigma, float(numbers[-3]))
        elasticsigma = np.append(elasticsigma, float(numbers[-2]))
        inelasticsigma = np.append(inelasticsigma, float(numbers[-1]))

    xsections = pandas.DataFrame({'roots [GeV]': roots, 'totalsigma [mb]': totalsigma, 'elasticsigma [mb]': elasticsigma, 'inelasticsigma [mb]': inelasticsigma})
    print(xsections.head())
    xsections.to_csv('/mnt/james/xsections.csv', index=False)
