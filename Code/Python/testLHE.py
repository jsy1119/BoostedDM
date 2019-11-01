import pylhe
import sys

# Read an example .lhe from a crmc run
# Had to modify the pylhe/__init__.py file to ignore the geometry line in the
# event listing

if __name__ == '__main__':
if sys.argv[1] == 'kcl':
    data_dir = '/home/k1893416/allMyStuff/BoostedDM/crmcruns/'
else:
    data_dir = '/Users/james/allMyStuff/BoostedDM/crmcruns/'
events = []

for event in pylhe.readLHE(data_dir + '14TeVpp.lhe'):
    events.append(event)

# As an example of the output, each event is a dictionary such as:

'''
event = {
'eventinfo':
 {'nparticles': 176.0,
  'pid': -1.0,
  'weight': 1.0,
  'scale': -1.0,
  'aqed': -1.0,
  'aqcd': -1.0},

'particles': [
  {'id': 22.0,
   'status': 1.0,
   'mother1': 182.0,
   'mother2': 0.0,
   'color1': 0.0,
   'color2': 0.0,
   'px': 0.08603484183549881,
   'py': -0.18108680844306946,
   'pz': -0.03536580130457878,
   'e': 0.20358085888332225,
   'm': 0.0,
   'lifetime': 3.738733902931923e+19,
   'spin': 9.0}, ...
  ]
}
'''

# As an example we could print out the total number of particles in the first
# event

print('Number of particles:', int(events[1]['eventinfo']['nparticles']))

# Or we could look through an event and look for pions with greater than 1GeV
# energy

for particle in events[1]['particles']:
    if particle['id'] == 111.0 and particle['e'] > 1.0:
        print('Found neutral pion with energy', '{0:.2f}'.format(particle['e']), 'GeV')
