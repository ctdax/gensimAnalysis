from matplotlib import pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.patches import Circle, Wedge
import numpy as np
import pandas as pd
import sys

# Define the detector geometries
detectorVertices = {'Tracker': [[[-300,-100,-100], [-300,100,-100], [-300,100,100], [-300,-100,100]],
                                [[300,-100,-100], [300,100,-100], [300,100,100], [300,-100,100]],
                                [[-300,-100,-100], [300,-100,-100], [300,-100,100], [-300,-100,100]],
                                [[-300,100,-100], [300,100,-100], [300,100,100], [-300,100,100]],
                                [[-300,-100,-100], [300,-100,-100], [300,100,-100], [-300,100,-100]],
                                [[-300,-100,100], [300,-100,100], [300,100,100], [-300,100,100]]],
                    'ECAL'   : [[[-400,-175,-175], [-400,175,-175], [-400,175,175], [-400,-175,175]],
                                [[400,-175,-175], [400,175,-175], [400,175,175], [400,-175,175]],
                                [[-400,-175,-175], [400,-175,-175], [400,-175,175], [-400,-175,175]],
                                [[-400,175,-175], [400,175,-175], [400,175,175], [-400,175,175]],
                                [[-400,-175,-175], [400,-175,-175], [400,175,-175], [-400,175,-175]],
                                [[-400,-175,175], [400,-175,175], [400,175,175], [-400,175,175]]],
                    'HCAL'   : [[[-560, -290, -290], [-560, 290, -290], [-560, 290, 290], [-560, -290, 290]],
                                [[560, -290, -290], [560, 290, -290], [560, 290, 290], [560, -290, 290]],
                                [[-560, -290, -290], [560, -290, -290], [560, -290, 290], [-560, -290, 290]],
                                [[-560, 290, -290], [560, 290, -290], [560, 290, 290], [-560, 290, 290]],
                                [[-560, -290, -290], [560, -290, -290], [560, 290, -290], [-560, 290, -290]],
                                [[-560, -290, 290], [560, -290, 290], [560, 290, 290], [-560, 290, 290]]],
                    'Muon'   : [[[-1100,-750,-750], [-1100,750,-750], [-1100,750,750], [-1100,-750,750]],
                                [[1100,-750,-750], [1100,750,-750], [1100,750,750], [1100,-750,750]],
                                [[-1100,-750,-750], [1100,-750,-750], [1100,-750,750], [-1100,-750,750]],
                                [[-1100,750,-750], [1100,750,-750], [1100,750,750], [-1100,750,750]],
                                [[-1100,-750,-750], [1100,-750,-750], [1100,750,-750], [-1100,750,-750]],
                                [[-1100,-750,750], [1100,-750,750], [1100,750,750], [-1100,750,750]]]
                    }

# Set in place markers for different PDGIDs
markers = ['s', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'x', 'd', '|', '_']

# Dictionary for PDGIDs
PDGDictionary = {11:      r'$e^-$, Q = -1',
                -11:      r'$e^+$, Q = 1',
                 22:      r'$\gamma$, Q = 0',
                 13:      r'$\mu^-$, Q = -1',
                -13:      r'$\mu^+$, Q = 1',
                 211:     r'$\pi^+$, Q = 1',
                -211:     r'$\pi^-$, Q = -1',
                 210:     r'$\pi^0$, Q = 0',
                 321:     r'$K^+$, Q = 1',
                -321:     r'$K^-$, Q = -1',
                 130:     r'$K^0_L$, Q = 0',
                 310:     r'$K^0_S$, Q = 0',
                 2212:    r'$p$, Q = 1',
                 2112:    r'$n$, Q = 0',
                 1000993: r'$\tilde{g}$g, Q = 0',
                 1009113: r'$\tilde{g}$d$\overline{d}$, Q = 0',
                 1009213: r'$\tilde{g}$u$\overline{d}$, Q = 1',
                 1091114: r'$\tilde{g}$ddd, Q = -1',
                 1092214: r'$\tilde{g}$uud, Q = 1',
                 1009313: r'$\tilde{g}$d$\overline{s}$, Q = 0',
                 1009323: r'$\tilde{g}$u$\overline{s}$, Q = 1',
                 1009223: r'$\tilde{g}$u$\overline{u}$, Q = 0',
                 1009333: r'$\tilde{g}$s$\overline{s}$, Q = 0',
                 1092114: r'$\tilde{g}$udd, Q = 0',
                 1092224: r'$\tilde{g}$uuu, Q = 2',
                 1093114: r'$\tilde{g}$sdd, Q = -1',
                 1093214: r'$\tilde{g}$sud, Q = 0',
                 1093224: r'$\tilde{g}$suu, Q = 1',
                 1093314: r'$\tilde{g}$ssd, Q = -1',
                 1093324: r'$\tilde{g}$ssu, Q = 0',
                 1093334: r'$\tilde{g}$sss, Q = -1',
                -1009213: r'$\tilde{g}$d$\overline{u}$, Q = -1',
                -1091114: r'$\tilde{g}$$\overline{d}$$\overline{d}$$\overline{d}$, Q = 1',
                -1092214: r'$\tilde{g}$$\overline{u}$$\overline{u}$$\overline{d}$, Q = -1',
                -1009313: r'$\tilde{g}$s$\overline{d}$, Q = 0',
                -1009323: r'$\tilde{g}$s$\overline{u}$, Q = -1',
                -1092114: r'$\tilde{g}$$\overline{u}$$\overline{d}$$\overline{d}$, Q = 0',
                -1092224: r'$\tilde{g}$$\overline{u}$$\overline{u}$$\overline{u}$, Q = -2',
                -1093114: r'$\tilde{g}$$\overline{s}$$\overline{d}$$\overline{d}$, Q = 1',
                -1093214: r'$\tilde{g}$$\overline{s}$$\overline{u}$$\overline{d}$, Q = 0',
                -1093224: r'$\tilde{g}$$\overline{s}$$\overline{u}$$\overline{u}$, Q = -1',
                -1093314: r'$\tilde{g}$$\overline{s}$$\overline{s}$$\overline{d}$, Q = 1',
                -1093324: r'$\tilde{g}$$\overline{s}$$\overline{s}$$\overline{u}$, Q = 0',
                -1093334: r'$\tilde{g}$$\overline{s}$$\overline{s}$$\overline{s}$, Q = 1',
                 1000022: r'$\tilde{\chi}^0_1$, Q = 0',
                 1000023: r'$\tilde{\chi}^0_2$, Q = 0',
                 1000025: r'$\tilde{\chi}^0_3$, Q = 0',
                 1000035: r'$\tilde{\chi}^0_4$, Q = 0',}


def energyDepositedHistogram(models, particles, events, listdf, filepath=None): # Create a histogram of energy deposited
    colors = ["#BF2229", "#00A88F"]
    if events[0] == -1:
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharey=False, sharex=True)
        axes[1].set_position([0.1, 0.1, 0.8, 0.2]) # Adjust size of axes
        axes[0].set_position([0.1, 0.3, 0.8, 0.6]) # Adjust size of axes
        bins=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        listcounts = [[], []]
        i = 0
        for df in listdf:
            counts, _, _ = axes[0].hist(df['Energy Deposit'], bins=bins, range=(0, 10), color=colors[i], edgecolor='black', alpha=0.5, histtype='stepfilled', label='{}, n={}'.format(models[i], len(df)))
            axes[0].set_xscale('log')
            axes[0].set_xticks(bins, ['KeV', '10 KeV', '100 KeV', 'MeV', '10 MeV', '100 MeV', 'GeV', '10 GeV'])
            axes[0].set_ylabel('Counts')
            axes[0].set_title('1800GeV gluino R-hadron energy deposits in 1000 events'.format(particles, len(df['Event'].unique())))
            axes[0].legend()
            listcounts[i] = counts
            i+=1

        # Create ratio plot
        ratioPositions = [3 * bins[i] for i in range(len(bins)-1)]
        xerr = [[], []]
        for i in range(len(ratioPositions)):
            xerr[0].append(ratioPositions[i]-bins[i])
            xerr[1].append(bins[i+1]-ratioPositions[i])
        ratio = listcounts[1] / listcounts[0]
        uncertainty = ratio * np.sqrt(1/listcounts[0] + 1/listcounts[1])
        axes[1].errorbar(ratioPositions, ratio, xerr=xerr, yerr=uncertainty, fmt='o', color='black', label='Ratio')
        axes[1].set_ylim(0.75, 1.25)
        axes[1].axhline(y=1, color='black', linestyle='--')
        axes[1].set_ylabel('Ratio')
        axes[1].set_xlabel('Energy Deposited')

        if filepath is not None:
            plt.savefig(filepath + '/{}_event{}_energydeposited.png'.format(particles, event))
        else:
            plt.show()
        
    else:
        for event in events:
            i = 0
            for df in listdf:
                plt.hist(df[df['Event'] == event]['Energy Deposit'], bins=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10], range=(0, 10), color=colors[i], edgecolor='black', alpha=0.5, histtype='stepfilled', label='{}, n={}'.format(models[i], len(df[df['Event'] == event])))
                plt.xscale('log')
                plt.xticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10], ['KeV', '10 KeV', '100 KeV', 'MeV', '10 MeV', '100 MeV', 'GeV', '10 GeV'])
                plt.xlabel('Energy Deposited')
                plt.ylabel('Counts')
                plt.title('{} energy deposits in event {}'.format(particles, event))
                i+=1
            plt.legend()
            if filepath is not None:
                plt.savefig(filepath + '/{}_event{}_energydeposited.png'.format(particles, event))
            else:
                plt.show()


def handleHCAL(df):
    hcal_df = df[df['isHCAL'] == 1].copy()

    # Calculate the magnitude of the vector formed by px and py for each row
    pxpy_magnitude = np.sqrt(hcal_df['px']**2 + hcal_df['py']**2)
    prpz_magnitude = np.sqrt(pxpy_magnitude**2 + hcal_df['pz']**2)

    # Normalize the px and py columns across the row
    hcal_df['px_norm'] = hcal_df['px'] / pxpy_magnitude
    hcal_df['py_norm'] = hcal_df['py'] / pxpy_magnitude
    hcal_df['pz_norm'] = hcal_df['pz'] / prpz_magnitude
    hcal_df['pr_norm'] = pxpy_magnitude / prpz_magnitude

    # Set x [cm] and y [cm] based on the normalized values
    hcal_df['x [cm]'] = hcal_df['px_norm'] * 230
    hcal_df['y [cm]'] = hcal_df['py_norm'] * 230
    hcal_df['z [cm]'] = hcal_df['pz_norm'] * 490
    hcal_df['r [cm]'] = hcal_df['pr_norm'] * 500

    # Map the r and z values to a set of allowed coordinates in the HCAL region
    hcal_df.loc[hcal_df['z [cm]'] < -390, 'z [cm]'] = -490 # left region
    hcal_df.loc[hcal_df['z [cm]'] >  390, 'z [cm]'] = 490  # right region
    hcal_df.loc[np.abs(hcal_df['z [cm]']) <  390, 'r [cm]'] = 210  # middle region
    hcal_df.loc[hcal_df['r [cm]'] >  210, 'r [cm]'] = 210  # middle region        

    # Update the original dataframe with the new values
    df.loc[hcal_df.index, 'x [cm]'] = hcal_df['x [cm]']
    df.loc[hcal_df.index, 'y [cm]'] = hcal_df['y [cm]']
    df.loc[hcal_df.index, 'z [cm]'] = hcal_df['z [cm]']
    df.loc[hcal_df.index, 'r [cm]'] = hcal_df['r [cm]']

    return df
    

def twoDPlot(particles, events, df, model, decayVerticesDf=None, filepath=None): # Create an RZ and XY plot for each event
    '''
    particles: 'All', 'SM', or 'SUSY'
    events: list of events to plot, or [-1] for all events
    df: pandas dataframe
    model: List of the model names
    decay: Boolean to show SM decay products
    filepath: path to save the plots, or None to show the plots
    '''

    # Insert boolean column for stopped particles if beta < 0.05
    #df['stopped'] = df['Track Energy'].apply(lambda E: 1 if np.sqrt(1 - (1800/E)**2) < 0.05 else 0)

    if events[0] == -1:
        events = df['Event'].unique()

    # Iterate over each event
    for event in events:
        eventdf = df[df['Event'] == event]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharey=False)

        # Grab information
        uniqueIDs = eventdf['PDG'].unique() # Get unique PDGIDs in the event
        marker_dict = {} # Assign markers to each unique PDGID, with a default of 'o' for non R-Hadron or muon
        j = 0
        for i in range(len(uniqueIDs)):
            if uniqueIDs[i] in PDGDictionary:
                marker_dict[uniqueIDs[i]] = markers[j % len(markers)]
                j += 1
            else:
                marker_dict[uniqueIDs[i]] = 'o'

        # Correlate marker size to track energy
        minEnergyForMarkerSize = 0
        legendSizes = [20, 120, 220, (eventdf['Track Energy'].min() - minEnergyForMarkerSize) / (3400 - minEnergyForMarkerSize) * 200 + 20, (eventdf['Track Energy'].max() - minEnergyForMarkerSize) / (3400 - minEnergyForMarkerSize) * 200 + 20] # Marker sizes for the legend
        legendSizeLabels = [minEnergyForMarkerSize, 1800, 3400, eventdf['Track Energy'].min(), eventdf['Track Energy'].max()] # Labels for the legend
        ax1.scatter([], [], color='w', label="Track Energies:", s=50, alpha=0)
        for i in range(len(legendSizes)):
            if i == len(legendSizes) - 1:
                ax1.scatter([], [], color='w', edgecolors='black', label="Max = {}".format(round(legendSizeLabels[i])), s=legendSizes[i]) 
            elif i == len(legendSizes) - 2:
                ax1.scatter([], [], color='w', edgecolors='black', label="Min = {}".format(round(legendSizeLabels[i])), s=legendSizes[i])
        ax1.scatter([], [], color='w', label=' ', s=50, alpha=0)
        ax1.scatter([], [], color='w', label='Particles:', s=50, alpha=0)

        # Put black circle at (0, 0) in the XY plot with size 10
        ax2.scatter(0, 0, color='w', edgecolor='black', s=30, zorder=10)

        # Add stop sign for decayed Rhadrons
        if decayVerticesDf is not None:
            eventVertices = decayVerticesDf[decayVerticesDf['Event'] == event]
            eventVertices['primaryVertexR'] = np.sqrt(eventVertices['primaryVertexX']**2 + eventVertices['primaryVertexY']**2)
            print(eventVertices)
            ax1.scatter(eventVertices['primaryVertexZ'], eventVertices['primaryVertexR'], edgecolor='red', linewidths=1, c='lightgray', marker='8', s=100, label='Decay', zorder=10)
            ax2.scatter(eventVertices['primaryVertexX'], eventVertices['primaryVertexY'], edgecolor='red', linewidths=1, c='lightgray', marker='8', s=100, zorder=10)

        # Plot the data
        if 'o' in marker_dict.values(): # If there are other SM particles, plot them first
            ax1.scatter([], [], marker='o', color='lightgreen', edgecolors='black', label='Other SM', s=50)
        for uniqueID in uniqueIDs:
            subset = eventdf[eventdf['PDG'] == uniqueID]
            sizes = (subset['Track Energy'] - minEnergyForMarkerSize) / (3400 - minEnergyForMarkerSize) * 200 + 20 # Scale the marker size based on the track energy
            cmap = plt.cm.nipy_spectral

            try:
                ax1.scatter([], [], marker=marker_dict[uniqueID], color='lightgreen', edgecolors='black', label=PDGDictionary[uniqueID], s=50)  # Add normal size legend symbol
            except KeyError:
                pass
            alpha = 0.5 if marker_dict[uniqueID] == 'o' else 0.9 # Set alpha to 0.3 for SM particles, 0.9 for R-Hadrons
            ax1.scatter(subset['z [cm]'], subset['r [cm]'], cmap=cmap, alpha=alpha, c=subset['Energy Deposit'], s=sizes, marker=marker_dict[uniqueID], norm=colors.LogNorm(vmin=1e-9, vmax=1000), linewidths=0, edgecolor='orange')
            ax2.scatter(subset['x [cm]'], subset['y [cm]'], cmap=cmap, alpha=alpha, c=subset['Energy Deposit'], s=sizes, marker=marker_dict[uniqueID], norm=colors.LogNorm(vmin=1e-9, vmax=1000), linewidths=0, edgecolor='orange')
        
        # Add boxes to represent the detectors in the RZ plot
        ax1.fill_betweenx([0, 112], -270, 270, color='yellow', alpha=0.1, edgecolor='w') # Tracker
        ax1.text(-250, 80, 'Tracker', fontsize=12, fontweight='bold')
        ax1.fill_betweenx([112, 175], -270, 270, color='green', alpha=0.1, edgecolor='w') # ECAL
        ax1.fill_betweenx([0, 175], -390, -270, color='green', alpha=0.1, edgecolor='w') # ECAL
        ax1.fill_betweenx([0, 175], 270, 390, color='green', alpha=0.1, edgecolor='w') # ECAL
        ax1.text(-370, 145, 'ECAL', fontsize=12, fontweight='bold')
        ax1.fill_betweenx([175, 290], -390, 390, color='blue', alpha=0.1, edgecolor='w') # HCAL
        ax1.fill_betweenx([0, 290], -560, -390, color='blue', alpha=0.1, edgecolor='w') # HCAL
        ax1.fill_betweenx([0, 290], 390, 560, color='blue', alpha=0.1, edgecolor='w') # HCAL
        ax1.text(-540, 260, 'HCAL', fontsize=12, fontweight='bold')
        ax1.fill_betweenx([290, 750], -560, 560, color='red', alpha=0.1, edgecolor='w') # Muon
        ax1.fill_betweenx([0, 750], -1100, -560, color='red', alpha=0.1, edgecolor='w') # Muon
        ax1.fill_betweenx([0, 750], 560, 1100, color='red', alpha=0.1, edgecolor='w') # Muon
        ax1.text(-1000, 700, 'Muon Chamber', fontsize=12, fontweight='bold')

        # Add red dashed lines to represent eta = -2.4, -1, 1, 2.4:
        ax1.plot([0, -1100], [0, 201.23], linestyle='--', color='red', alpha=0.2) # eta = -2.4
        ax1.plot([0, -894], [0, 750], linestyle='--', color='red', alpha=0.2) # eta = -1
        ax1.plot([0, 894], [0, 750], linestyle='--', color='red', alpha=0.2) # eta = 1
        ax1.plot([0, 1100], [0, 201.23], linestyle='--', color='red', alpha=0.2) # eta = 2.4

        # Add circles to represent the detectors in the XY plot
        ax2.add_patch(plt.Circle((0, 0), 112, color='yellow', alpha=0.1, edgecolor='w'))
        ax2.add_patch(Wedge((0, 0), 175, 0, 360, width=63, color='green', alpha=0.1, edgecolor='w'))
        ax2.add_patch(Wedge((0, 0), 290, 0, 360, width=115, color='blue', alpha=0.1, edgecolor='w'))
        ax2.add_patch(Wedge((0, 0), 750, 0, 360, width=460, color='red', alpha=0.1, edgecolor='w'))

        # Add labels to the RZ and XY plots
        ax1.axis(xmin=-1100, xmax=1100, ymin=0, ymax=750)
        ax1.set_xlabel('z (cm)')
        ax1.set_ylabel('r (cm)')
        ax1.autoscale(enable=False, axis='y')        
        ax2.axis(xmin=-750, xmax=750, ymin=-750, ymax=750)
        ax2.set_xlabel('x (cm)')
        ax2.set_ylabel('y (cm)')
        ax2.autoscale(enable=False, axis='y')

        # Add the legend and colorbar
        plt.suptitle('{} {} hits in event {}'.format(model, particles, event))
        cbar = fig.colorbar(ax2.collections[-1])
        cbar.solids.set_alpha(1) # Force alpha=1
        cbar.set_ticks([1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])
        cbar.set_ticklabels(['eV', 'KeV', 'MeV', '10 MeV', '100 MeV', 'GeV', '10 GeV', '100 GeV', 'TeV'])
        cbar.set_label('Energy Deposited')
        ax1.legend(bbox_to_anchor=(-0.15, 1.15), loc='upper left', borderaxespad=0.)

        if filepath is not None:
            plt.savefig(filepath + '/{}_event{}_rz.png'.format(particles, event))
        else:
            plt.show()


def twoDPlotComparison(particles, events, dfs, models, filepath=None): # Create an RZ and XY plot for each event. Compares two models
    '''
    particles: 'All', 'SM', or 'SUSY'
    events: list of events to plot, or [-1] for all events
    df: pandas dataframe
    filepath: path to save the plots, or None to show the plots
    '''

    if events[0] == -1:
        df = dfs[0]
        events = df['Event'].unique()

    # Iterate over each event
    for event in events:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey=False)
        ax_pairs = [(axes[0, 0], axes[0, 1]), (axes[1, 0], axes[1, 1])]
        axisIterator = 0
        smPlotted = False
        for df in dfs:
            eventdf = df[df['Event'] == event]
            susyeventdf = eventdf[(abs(eventdf['PDG']) > 999999) & (abs(eventdf['PDG']) < 3999999)]
            #susyeventdf['stopped'] = susyeventdf['Track Energy'].apply(lambda E: 1 if np.sqrt(1 - (1800/E)**2) < 0.05 else 0)   # Insert boolean column for stopped R-Hadrons if beta < 0.05
            ax1, ax2 = ax_pairs[axisIterator]

            # Grab information
            uniqueIDs = eventdf['PDG'].unique() # Get unique PDGIDs in the event
            marker_dict = {} # Assign markers to each unique PDGID, with a default of 'o' for non R-Hadron or muon
            for i in range(len(uniqueIDs)):
                if uniqueIDs[i] in PDGDictionary:
                    marker_dict[uniqueIDs[i]] = markers[i % len(markers)]

            # Correlate marker size to track energy
            legendSizes = [(susyeventdf['Track Energy'].min() - 1800) / (3400 - 1800) * 200 + 20, (susyeventdf['Track Energy'].max() - 1800) / (3400 - 1800) * 200 + 20] # Marker sizes for the legend
            legendSizeLabels = [susyeventdf['Track Energy'].min(), susyeventdf['Track Energy'].max()] # Labels for the legend
            ax1.scatter([], [], color='w', label="Track Energies:", s=50, alpha=0)
            for i in range(len(legendSizes)):
                if i == len(legendSizes) - 1:
                    ax1.scatter([], [], color='w', edgecolors='black', label="Max = {}".format(round(legendSizeLabels[i])), s=legendSizes[i]) 
                elif i == len(legendSizes) - 2:
                    ax1.scatter([], [], color='w', edgecolors='black', label="Min = {}".format(round(legendSizeLabels[i])), s=legendSizes[i])
            ax1.scatter([], [], color='w', label=' ', s=50, alpha=0)
            ax1.scatter([], [], color='w', label='Particles:', s=50, alpha=0)

            # Add stop sign for stopped particles
            #stopped = susyeventdf[susyeventdf['stopped'] == 1]
            #if len(stopped) > 0:
            #    ax1.scatter(stopped['z [cm]'], stopped['r [cm]'], edgecolor='red', linewidths=1, c='lightgray', marker='8', s=50, label='Stopped Particles')
            #    ax2.scatter(stopped['x [cm]'], stopped['y [cm]'], edgecolor='red', linewidths=1, c='lightgray', marker='8', s=50, label='Stopped Particles')

            # Plot the data
            for uniqueID in uniqueIDs:
                subset = eventdf[eventdf['PDG'] == uniqueID]
                if (abs(uniqueID) > 999999) & (abs(uniqueID) < 3999999):
                    sizes = (subset['Track Energy'] - 1800) / (3400 - 1800) * 200 + 20 # Scale the marker size based on the track energy for SUSY particles
                else:
                    sizes = np.ones(len(subset)) * 5 # Default size = 5 for SM particles

                cmap = colors.LinearSegmentedColormap.from_list('mycmap', plt.cm.nipy_spectral(np.linspace(0.2, 1, 256)))
                if uniqueID in PDGDictionary: # Add normal size legend symbol
                    ax1.scatter([], [], marker=marker_dict[uniqueID], color='lightgreen', edgecolors='black', label=PDGDictionary[uniqueID], s=50)
                    ax1.scatter(subset['z [cm]'], subset['r [cm]'], cmap=cmap, alpha=0.9, c=subset['Energy Deposit'], s=sizes, marker=marker_dict[uniqueID], norm=colors.LogNorm(vmin=1e-9, vmax=1000), linewidths=subset['isHCAL'], edgecolor='orange')
                    ax2.scatter(subset['x [cm]'], subset['y [cm]'], cmap=cmap, alpha=0.9, c=subset['Energy Deposit'], s=sizes, marker=marker_dict[uniqueID], norm=colors.LogNorm(vmin=1e-9, vmax=1000), linewidths=subset['isHCAL'], edgecolor='orange')
                else:
                    if smPlotted == False:
                        ax1.scatter([], [], marker='o', color='lightgreen', edgecolors='black', label='SM', s=50)
                        smPlotted = True
                    ax1.scatter(subset['z [cm]'], subset['r [cm]'], cmap=cmap, alpha=0.9, c=subset['Energy Deposit'], s=sizes, marker='o', norm=colors.LogNorm(vmin=1e-9, vmax=1000), linewidths=subset['isHCAL'], edgecolor='orange')
                    ax2.scatter(subset['x [cm]'], subset['y [cm]'], cmap=cmap, alpha=0.9, c=subset['Energy Deposit'], s=sizes, marker='o', norm=colors.LogNorm(vmin=1e-9, vmax=1000), linewidths=subset['isHCAL'], edgecolor='orange')

            # Add boxes to represent the detectors in the RZ plot
            ax1.fill_betweenx([0, 112], -270, 270, color='yellow', alpha=0.1, edgecolor='w') # Tracker
            ax1.text(-250, 80, 'Tracker', fontsize=12, fontweight='bold')
            ax1.fill_betweenx([112, 175], -270, 270, color='green', alpha=0.1, edgecolor='w') # ECAL
            ax1.fill_betweenx([0, 175], -390, -270, color='green', alpha=0.1, edgecolor='w') # ECAL
            ax1.fill_betweenx([0, 175], 270, 390, color='green', alpha=0.1, edgecolor='w') # ECAL
            ax1.text(-370, 145, 'ECAL', fontsize=12, fontweight='bold')
            ax1.fill_betweenx([175, 290], -390, 390, color='blue', alpha=0.1, edgecolor='w') # HCAL
            ax1.fill_betweenx([0, 290], -560, -390, color='blue', alpha=0.1, edgecolor='w') # HCAL
            ax1.fill_betweenx([0, 290], 390, 560, color='blue', alpha=0.1, edgecolor='w') # HCAL
            ax1.text(-540, 260, 'HCAL', fontsize=12, fontweight='bold')
            ax1.fill_betweenx([290, 750], -560, 560, color='red', alpha=0.1, edgecolor='w') # Muon
            ax1.fill_betweenx([0, 750], -1100, -560, color='red', alpha=0.1, edgecolor='w') # Muon
            ax1.fill_betweenx([0, 750], 560, 1100, color='red', alpha=0.1, edgecolor='w') # Muon
            ax1.text(-1000, 700, 'Muon Chamber', fontsize=12, fontweight='bold')

            # Add red dashed lines to represent eta = -2.4, -1, 1, 2.4:
            ax1.plot([0, -1100], [0, 201.23], linestyle='--', color='red', alpha=0.2) # eta = -2.4
            ax1.plot([0, -894], [0, 750], linestyle='--', color='red', alpha=0.2) # eta = -1
            ax1.plot([0, 894], [0, 750], linestyle='--', color='red', alpha=0.2) # eta = 1
            ax1.plot([0, 1100], [0, 201.23], linestyle='--', color='red', alpha=0.2) # eta = 2.4

            # Add circles to represent the detectors in the XY plot
            ax2.add_patch(plt.Circle((0, 0), 112, color='yellow', alpha=0.1, edgecolor='w'))
            ax2.add_patch(Wedge((0, 0), 175, 0, 360, width=63, color='green', alpha=0.1, edgecolor='w'))
            ax2.add_patch(Wedge((0, 0), 290, 0, 360, width=115, color='blue', alpha=0.1, edgecolor='w'))
            ax2.add_patch(Wedge((0, 0), 750, 0, 360, width=460, color='red', alpha=0.1, edgecolor='w'))

            # Add labels to the RZ and XY plots
            ax1.axis(xmin=-1100, xmax=1100, ymin=0, ymax=750)
            ax1.set_xlabel('z (cm)')
            ax1.set_ylabel('r (cm)')
            ax1.autoscale(enable=False, axis='y')        
            ax2.axis(xmin=-750, xmax=750, ymin=-750, ymax=750)
            ax2.set_xlabel('x (cm)')
            ax2.set_ylabel('y (cm)')
            ax2.autoscale(enable=False, axis='y')

            # Add the legend and colorbar
            cbar = fig.colorbar(ax2.collections[-1])
            cbar.set_ticks([1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])
            cbar.set_ticklabels(['eV', 'KeV', 'MeV', '10 MeV', '100 MeV', 'GeV', '10 GeV', '100 GeV', 'TeV'])
            cbar.set_label('Energy Deposited')
            ax1.legend(bbox_to_anchor=(-0.3, 1.3), loc='upper left', borderaxespad=0.)
            ax2.set_title('{} {} hits in event {}'.format(models[axisIterator], particles, event))

            axisIterator += 1

        if filepath is not None:
            plt.savefig(filepath + '/{}_event{}_rz.png'.format(particles, event))
        else:
            plt.show()


def XYZPlot_EnergyDeposited(detectorVertices, particles, events, df, filepath=None): # Create x y z plot for each event
    if events[0] == -1:
        events = df['Event'].unique()
    for event in events:
        eventdf = df[df['Event'] == event]
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        #Split data for proper marker plotting
        smdf = eventdf[abs(eventdf['PDG']) < 999999]
        susydf = eventdf[abs(eventdf['PDG']) > 999999]

        if particles == "SM":
            sc = ax.scatter(smdf['z [cm]'], smdf['x [cm]'], smdf['y [cm]'], marker='o', c=smdf['Energy Deposit'], cmap='viridis', alpha=1, norm=colors.LogNorm(vmin=0.0001, vmax=1))
        elif particles == "SUSY":
            sc = ax.scatter(susydf['z [cm]'], susydf['x [cm]'], susydf['y [cm]'], marker='o', c=susydf['Energy Deposit'], cmap='viridis', alpha=1, norm=colors.LogNorm(vmin=0.00000001, vmax=1))
        else:
            sc = ax.scatter(smdf['z [cm]'], smdf['x [cm]'], smdf['y [cm]'], marker='o', c=smdf['Energy Deposit'], cmap='viridis', alpha=1, norm=colors.LogNorm(vmin=0.0001, vmax=1), label='SM')
            ax.scatter(susydf['z [cm]'], susydf['x [cm]'], susydf['y [cm]'], marker='x',  c=susydf['Energy Deposit'], cmap='viridis', alpha=1, norm=colors.LogNorm(vmin=0.0001, vmax=1), label='SUSY')   
            plt.legend(loc='upper left')         

        ax.set_xlabel('z (cm)')
        ax.set_ylabel('x (cm)')
        ax.set_zlabel('y (cm)')
        ax.set_xlim(-1200, 1200)
        ax.set_ylim(-800, 800)
        ax.set_zlim(-800, 800)
        plt.suptitle('{} energy deposits in event {}'.format(particles, event))
        cbar = plt.colorbar(sc)
        cbar.set_label('Energy Deposited [GeV]')

        # Add boxes to represent the detectors
        ax.add_collection3d(Poly3DCollection(detectorVertices['Tracker'], facecolor='yellow', linewidths=1, edgecolors='black', alpha=.05))
        ax.add_collection3d(Poly3DCollection(detectorVertices['ECAL'], facecolor='cyan', linewidths=1, edgecolors='black', alpha=.05))
        ax.add_collection3d(Poly3DCollection(detectorVertices['HCAL'], facecolor='orange', linewidths=1, edgecolors='black', alpha=.05))
        ax.add_collection3d(Poly3DCollection(detectorVertices['Muon'], facecolor='red', linewidths=1, edgecolors='black', alpha=.05))



        if filepath is not None:
            plt.savefig(filepath + '/{}_event{}_xy.png'.format(particles, event))
        else:
            plt.show()


def XYZPlot_TrackEnergy(detectorVertices, particles, events, df, filepath=None): # Create x y z plot for each event
    if events[0] == -1:
        events = df['Event'].unique()
    for event in events:
        eventdf = df[df['Event'] == event]
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        #Split data for proper marker plotting
        smdf = eventdf[abs(eventdf['PDG']) < 999999]
        susydf = eventdf[abs(eventdf['PDG']) > 999999]

        if particles == "SM":
            sc = ax.scatter(smdf['z [cm]'], smdf['x [cm]'], smdf['y [cm]'], marker='o', c=smdf['Track Energy'], cmap='viridis', alpha=1, norm=colors.LogNorm(vmin=0.0001, vmax=1))
        elif particles == "SUSY":
            sc = ax.scatter(susydf['z [cm]'], susydf['x [cm]'], susydf['y [cm]'], marker='o', c=susydf['Track Energy'], cmap='viridis', alpha=1)
        else:
            sc = ax.scatter(smdf['z [cm]'], smdf['x [cm]'], smdf['y [cm]'], marker='o', c=smdf['Track Energy'], cmap='viridis', alpha=1, norm=colors.LogNorm(vmin=0.0001, vmax=1), label='SM')
            ax.scatter(susydf['z [cm]'], susydf['x [cm]'], susydf['y [cm]'], marker='x',  c=susydf['Track Energy'], cmap='viridis', alpha=1, norm=colors.LogNorm(vmin=0.0001, vmax=1), label='SUSY')   
            plt.legend(loc='upper left')         

        ax.set_xlabel('z (cm)')
        ax.set_ylabel('x (cm)')
        ax.set_zlabel('y (cm)')
        ax.set_xlim(-1200, 1200)
        ax.set_ylim(-800, 800)
        ax.set_zlim(-800, 800)
        plt.suptitle('{} track energies in event {}'.format(particles, event))
        cbar = plt.colorbar(sc)
        cbar.set_label('Track Energy [GeV]')

        # Add boxes to represent the detectors
        ax.add_collection3d(Poly3DCollection(detectorVertices['Tracker'], facecolor='yellow', linewidths=1, edgecolors='black', alpha=.05))
        ax.add_collection3d(Poly3DCollection(detectorVertices['ECAL'], facecolor='cyan', linewidths=1, edgecolors='black', alpha=.05))
        ax.add_collection3d(Poly3DCollection(detectorVertices['HCAL'], facecolor='orange', linewidths=1, edgecolors='black', alpha=.05))
        ax.add_collection3d(Poly3DCollection(detectorVertices['Muon'], facecolor='red', linewidths=1, edgecolors='black', alpha=.05))



        if filepath is not None:
            plt.savefig(filepath + '/{}_event{}_xy.png'.format(particles, event))
        else:
            plt.show()


def totalMuonHits(df):
    return len(df[(df['r [cm]'] >= 290) | (np.abs(df['z [cm]']) >= 560)])


def medianTrackEnergy(df):
    return df['Track Energy'].median()


def pTvsR_ratio(listdf): # Create a pT vs r line plot in bins of 10cm where each point is the median pT in that bin
    '''
    listdf: list of dataframes
    '''

    # Set up the figure
    fig = plt.figure()
    plt.xlabel('r [cm]')
    plt.ylabel(r'Median $\frac{p_T^{v3.06}}{p_T^{v3}}$')
    plt.title(r'Median $p_T$ ratio for 2k 1800GeV gluino R-Hadrons')
    plt.xlim(0, 700)
    plt.ylim(0.8, 1.2)
    plt.xticks(np.arange(0, 700, 50))
    df_iter = 0

    # Initialize variables for plotting
    r, median_pT_ratio, xerr, yerr, counts = [], [], [], [], []

    # Iterate over each dataframe
    for df in listdf:
        df['r [cm]'] = np.sqrt(df['x [cm]']**2 + df['y [cm]']**2)
        df['pT'] = np.sqrt(df['px']**2 + df['py']**2)

        # Obtain the median pT ratio in each bin and plot
        bins = [0, 112, 175, 290, 700]
        for i in range(0, len(bins)-1):
            min_r = bins[i]
            max_r = bins[i+1]
            subdf = df[(df['r [cm]'] >= min_r) & (df['r [cm]'] < max_r)]

            if df_iter == 0:
                r.append(min_r + (max_r - min_r) / 2)
                xerr.append((max_r - min_r) / 2)
                yerr.append(subdf['pT'].std() / np.sqrt(subdf['pT'].count()))
                median_pT_ratio.append(subdf['pT'].median())
                counts.append(subdf['pT'].count())

            else:
                median_pT_of1 = median_pT_ratio[i]
                median_pT_ratio[i] = median_pT_ratio[i] / subdf['pT'].median()
                yerr[i] = median_pT_ratio[i] * np.sqrt((yerr[i]/median_pT_of1)**2 + ((subdf['pT'].std() / np.sqrt(subdf['pT'].count())) / subdf['pT'].median())**2)
            print(yerr[i], subdf['pT'].count())
        df_iter += 1

    # Plot the ratio
    plt.errorbar(r, median_pT_ratio, xerr=xerr, yerr=yerr, fmt='none', color='black')
    plt.axhline(y=1, color='black', linestyle='--')

    # Add boxes to represent the detectors
    plt.fill_betweenx([plt.ylim()[0], plt.ylim()[1]], 0, 112, color='yellow', alpha=0.1, edgecolor='w') # Tracker
    plt.text(10, 0.81, 'Tracker', fontsize=12, fontweight='bold')
    plt.fill_betweenx([plt.ylim()[0], plt.ylim()[1]], 112, 175, color='green', alpha=0.1, edgecolor='w') # ECAL
    plt.text(120, 0.81, 'ECAL', fontsize=12, fontweight='bold')
    plt.fill_betweenx([plt.ylim()[0], plt.ylim()[1]], 175, 290, color='blue', alpha=0.1, edgecolor='w') # HCAL
    plt.text(185, 0.81, 'HCAL', fontsize=12, fontweight='bold')
    plt.fill_betweenx([plt.ylim()[0], plt.ylim()[1]], 290, 700, color='red', alpha=0.1, edgecolor='w') # Muon
    plt.text(300, 0.81, 'Muon Chamber', fontsize=12, fontweight='bold')

    # Add legend and show plot
    plt.show()


def move_decimal_left(s):
    # Find the position of the decimal point
    decimal_pos = s.find('.')
    
    # If no decimal point found, return the original string
    if decimal_pos == -1:
        return s
    
    # If decimal is already at the beginning, add a zero
    if decimal_pos == 0:
        return '0' + s
    
    # Remove the decimal point and insert it one position to the left
    without_decimal = s.replace('.', '')
    new_decimal_pos = decimal_pos - 1
    
    # Insert decimal at new position
    result = without_decimal[:new_decimal_pos] + '.' + without_decimal[new_decimal_pos:]
    
    return result


def decayLogReader(filepath):
    '''
    Reads the decay log file and returns a dictionary with the event number and secondary vertex location.
    '''
    vertices = {'Event': [], 'primaryVertexX': [], 'primaryVertexY': [], 'primaryVertexZ': []}
    event = 0
    with open(filepath, 'r') as f:
        for line in f:
            # If line has Run 1, Event #, grab the number
            if 'Run 1, Event ' in line:
                event = int(line.split('Run 1, Event ')[1].split(',')[0])
            if 'RHadronPythiaDecayer: Location is ' in line:
                # Grab the secondary vertex location. Format is "RHadronPythiaDecayer: Location is (x, y, z)"
                coords = line.split('RHadronPythiaDecayer: Location is ')[1].strip('()').split(',')
                
                # Move the decimal point to the left by 1 in the string
                coords[0] = move_decimal_left(coords[0])
                coords[1] = move_decimal_left(coords[1])
                coords[2] = coords[2].split(')')[0]
                coords[2] = move_decimal_left(coords[2])

                # Append the coordinates to the vertices dictionary
                vertices['Event'].append(event)
                vertices['primaryVertexX'].append(float(coords[0]))
                vertices['primaryVertexY'].append(float(coords[1]))
                vertices['primaryVertexZ'].append(float(coords[2]))

    return vertices


def dataSelector(df, selection='SUSY', events=[-1], decayLog=None):
    '''
    Selects data from the dataframe based on the selection criteria.
    df: pandas dataframe containing the data
    selection: 'SUSY' for SUSY particles, 'All' for all particles, or 'SUSYDecay' for SUSY + decay products
    events: -1 for all events, or a list of event numbers to select
    decayLog: If provided, will include the decay products of Rhadrons
    '''
    verticesDict = None
    if selection == 'All':
        return df
    elif selection == 'SUSY':
        df = df[(abs(df['PDG']) > 999999) & (abs(df['PDG']) < 3999999)]
    elif selection == 'SUSYDecay':
        if decayLog is not None:
            verticesDict = decayLogReader(decayLog)

            # Return rows where the PDG is SUSY or the primary vertex matches the decay log
            df = df[(abs(df['PDG']) > 999999) & (abs(df['PDG']) < 3999999) | 
                    (df['Event'].isin(verticesDict['Event']) & 
                     df['primaryVertexX'].isin(verticesDict['primaryVertexX']) & 
                     df['primaryVertexY'].isin(verticesDict['primaryVertexY']) & 
                     df['primaryVertexZ'].isin(verticesDict['primaryVertexZ']))]
        else:
            print("Error: Decay log is None, but selection is 'SUSYDecay'. Exiting now.")
            sys.exit(1)
    
    if events != [-1]:
        df = df[df['Event'].isin(events)]

    return df, verticesDict

# Set parameters
models = ['~R->qq_10GeVchi10_10ns']
particles = "SUSYDecay" # All or SUSY
events = [-1] # -1 for all events, or a list of event
filepath = None # None for no save, or a path to save the plots

# Read in data
listdf = [pd.DataFrame() for i in range(len(models))]
listVerticesDf = [pd.DataFrame() for i in range(len(models))]
for i in range(len(models)):
    listdf[i] = pd.read_csv('data/RhadronDecay/eventdisplays/{}-eventdisplay.csv'.format(models[i]))
    decayLog = 'data/RhadronDecay/logs/{}.log'.format(models[i]) if particles == 'SUSYDecay' else None # Set decay log file if needed
    listdf[i], verticesDict = dataSelector(listdf[i], selection=particles, events=events, decayLog=decayLog) # Select data based on the parameters
    if verticesDict is not None:
        listVerticesDf[i] = pd.DataFrame(verticesDict)
    else:
        listVerticesDf[i] = None

# Handle HCAL
for df in listdf:
    df = handleHCAL(df)


# Print total number of muon hits
#for i in range(len(listdf)):
#    print('Total number of muon hits in {} = {}'.format(models[i], totalMuonHits(listdf[i])))

# Print median track energy
#for i in range(len(listdf)):
#    print('Median track energy in {} = {}'.format(models[i], medianTrackEnergy(listdf[i])))

# Create plots
#energyDepositedHistogram(models, particles, events, listdf, filepath)
#pTvsR_ratio(listdf)

if len(models) == 1:
    twoDPlot(particles, events, listdf[0], models[0], listVerticesDf[0], filepath)
elif len(models) == 2:
    twoDPlotComparison(particles, events, listdf, models, filepath)
else:
    for i in range(len(models)):
        twoDPlot(particles, events, listdf[i], models[i], filepath)

#XYZPlot_EnergyDeposited(detectorVertices, particles, events, listdf[i], filepath)
#XYZPlot_TrackEnergy(detectorVertices, particles, events, df, filepath)