## Repository that simplifies the Gen-Sim step in CMSSW and outputs a CSV designed for energy deposit plotting in matplotlib if desired.

### Step 1:
- Clone the repository in the /src directory. Do not change the names of the directories or files, it uses a custom class that expects particular names.

### Step 2:
- Modify `Demo/SpikedRHadronAnalyzer/simulate_rhadron_decays.py` to your desire. This is the python config file that does the Gen-Sim step.

### Step 3: 
- Modify `runSimulations.sh` to match what you need. You should only need to modify the 6 parameters at the top control center.

### Step 4:
- Run `cmsenv` followed by `scram b -j 8`. You are now ready to run the simulations. Run `./runSimulations.sh` and make sure you have edit access. If not, run `chmod +x runSimulations.sh` and try again.

### Step 5:
- If you set `eventdisplay` to `true` in `runSimulations.sh`, you should now have a CSV file in your respective /data directory and can now pass that into the `models` list in `simhitsAnalysis.py` if you would like to plot the energy deposits. The file expects the csv to exist in a particular directory that has been catered for me, but feel free to change it. The event display is also very much designed for Rhadron sims so other sims may run into issues.
