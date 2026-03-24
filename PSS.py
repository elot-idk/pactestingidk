
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime, timedelta
import scipy
from scipy import stats
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, filtfilt, hilbert #Fourier Transform
from scipy.optimize import curve_fit
import mne
import random
import serial
import csv
import json
from PSS.constants import * #Import the constants
from PSS.mock_data import * #Import the mock_data


"""
Behavioral data (Block 2 and Block 5 <- open loop)

Subject_id, block, trial #, soa_ms, response <-Light first, Sound first, Equal, reaction_time, timestamp

EEG
"""

#Mock function



"""
subject_name: ______, subject_id: _______
sub_1,
sub_2
sub_3
"""


def load_subject_registry():
    """Loads a saved subject's information"""
    os.makedirs("subject.csv", exist_ok = True)
    if os.path.exists(SUBJECTS_REGISTRY):
        with open(SUBJECTS_REGISTRY, 'r') as f:
            return json.load(f)
    return {"subjects": [], "next_id": 1}

def save_subject_registry(registry):
    """Saves a subject's information """
    with open(SUBJECTS_REGISTRY, "w") as f:
        json.dump(registry, f, indent = 2)


def get_device_specs(board_id):
    """Attempts to get sampling rate and eeg channels, and if it works, attempts to get eeg names."""
    try:
        sampling_rate = BoardShim.get_sampling_rate(board_id)
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        try:
            eeg_names = BoardShim.get_eeg_names(board_id)
        except:
            print("No channel names for device")
        return sampling_rate, eeg_channels, eeg_names
    except:
        print("Error")
        return None, None

def connect_eeg():
    """Attempts to connect EEG"""
    try:
        params = BrainFlowInputParams()
        board = BoardShim(EEG_BOARD_ID, params)
        board.prepare_session()
        print("EEG successfully connected")  
    except:
         print("Connection Failed")


#start EEG <- board.start_stream()
#stop <- board.stop_stream()


def get_eeg_data(board, num_samples = 250):
    """Gets the board data using the num_samples variable and eeg channels using board id before returning."""
    data = board.get_current_board_data(num_samples)
    eeg_channels = board.get_eeg_channels(EEG_BOARD_ID)
    return data[eeg_channels[EEG_CHANNEL], :]

#Channel 1: [4,4.0,2.0, 1.0, 20.0]
 
#disconnect <- board.release_session()

"""Pre-Processing"""

#Gamma
def bandpass_filter(data,fs,low_freq,high_freq,order = 4):
    """Filters brain data into a range of frequencies. """
    nyquist = 0.5 *fs #Maximum frequency that you can accurately get from an electronic device
    low = low_freq/nyquist
    high = high_freq/nyquist
    b,a = butter(order, [low, high], btype = "band") #Extract from that range, all the gamma values <- 30-100hz
    #100hz<- high gamma, 30hz <- low gamma
    return filtfilt(b,a, data)

def compute_phase(data):
    """Get's the phase at a point in the data"""
    analytic_signal = hilbert(data) #rotate your data <- represent it like a unit circle
    phase_radians = np.angle(analytic_signal) #The degree in radians
    phase_degrees = np.degrees(phase_radians) % 360
    return phase_degrees

    #720 % 360 -> 0<- phase of the beginner <- , 360*4 = 1140
    #1440 % 360 <- 0 <- 
    #720 <- 

def compute_amplitude(data):
    analytic_signal = hilbert(data) # Gives us the circle
    return np.abs(analytic_signal) #Give us the radius

def find_optimal_phase(theta_phase, gamma_amps, bin_width = 20):
    #360 degrees , 360/20, 18, -> 0, 20, 20, 40<-coordinates for the edges
    bin_edges = np.arrange(0, 360 +bin_width, bin_width)    #minimum value, #maximum value + bin_width,step <- bin_width
    # [(0,20), (20,40), (40,60)]
    bin_centers = (bin[:-1] + bin_edges[1:])/2 #give the center index
    # (0,20)->10
    #list comprehension
    gamma_by_bin = [[] for _ in range(len(bin_centers))]  #Creates an empty list for every bin
    #amplitude <- gamma <-5 <- theta phase <- 20 degrees, (5,20)<- first number <- gamma, second number <- theta

    for phase,amp in zip(theta_phase, gamma_amps): #iteratable<- all possible pairs
        # List of phases, list of amplitudes-> [(phase, amplitude), (phase,amplitude)]
        bin_index = int(phase // bin_width) # increments of 20, Divide without remainder<- integer division<- floor division
        #5/4 -> 5/4-1.25,  5 // 4-> 1

        #TODO: Append the amp<- to the list gamma_by_bin , bin_index
        # [[2.5],[6,7],[6,7]]-> bin_index = 0
        gamma_by_bin[bin_index].append(amp) #2.5

    mean_gamma_per_bin = np.mean(gamma_by_bin) # [[2,3]]-> [[2.5], [1]]

    best_bin_index = np.argmax(mean_gamma_per_bin) #-> 0

    #TODO: bin_centers<- list <- best_bin_index<- index
    optimal_phase = bin_centers[best_bin_index] #Got the center of the "best bin"<- select that phase

       #TODO: return best phase is, list of averages, bin centers
    return optimal_phase, mean_gamma_per_bin, bin_centers

        


    
 # for each each bin what are the coordinates
    
    
    
"""ARDUINO"""

def connect_arduino():
    
    """ Attempts to connect arduino to code, prints error if goes wrong."""
    try:
        arduino = serial.Serial(ARDUINO_PORT, 115200, timeout = 1)
        print(f"Arduino connected to port {ARDUINO_PORT}")
    except:
        print("Error connecting Arduino")
    


#Helper Functions

def send_stimulus(serial_number, soa_ms, led_ms = 50, buzzer_ms= 50):
    """Sends light/buzzer at soa"""
    cmd = f"S,{int(soa_ms),{led_ms}, {buzzer_ms}}\n"
    serial_number.write(cmd.encode('utf-8'))
    time.sleep(0.3)
    

def save_data(data_list, file_name): #Sub_1_behavioral_table.csv
    pass


def sigmoid(x, slope, pss):
    return 1/ (1+ np.exp(-slope*(x-pss))) #S curve shape

def calculate_pss(data_table):
    print("Data table")
    print(data_table.head()) #Printing out data table
    """This calculates the pss by getting the percentage of light first responses, then fit a curve line and return pss."""
    #soa<- delay <- x-axis, response<- light first, sound first, equal <- y -axis
    #y -axis <- % of light first responses
    data_table["response_num"] = data_table["response"].map({"L": 1, "S": 0}) # = 
    data_table = data_table[data_table["response_num"].notna()]
    #1, 0,1,0,01
    mean_responses = data_table.groupby("soa")["response_num"].mean() # Soa_value<- Na <- not a number
    """
    150 L, S
        40 20
    """
    x = mean_responses.index.values.astype(float) #creating a list of soa values
    y = mean_responses.values #My averages

    #sort the data first

    order = np.argsort(x) #[6,2,10], <- 0,1,2-> 1,0,2
    x,y = x[order], y[order] #applying the sorted indexes-> [2,6,10] instead of [6,2,10]

    #bounding the curve <- so it has limits
    bounds = (
        [0.001, x.min()],
        [10.0, x.max()]
    )


    initial_guess = [0.05, np.median(x)] #Guess for where the pss is

    params, _ = curve_fit(sigmoid, x, y, p0= initial_guess, bounds = bounds, maxfev = 10000) #Fit our data to a curve <- sigmoid curve

    slope,pss = params #me saving the slope of the curve and the pss, slope <- how fast to pss, pss <- 50% value on the curve

    return pss, slope, mean_responses
"""
Multi-line
"""
#single line
def plot_psychometric_curve(
        data_table, #contains the soa and the responses as columns
        pss,
        slope,
):
    """Gets the mean of % of light first and soa values and plots it."""
    #Setting up our data for plotting
    data_table["response_num"] = data_table["response"].map({"L": 1, "S": 0})
    data_table = data_table[data_table["response_num"].notna()]
    mean_responses = data_table.groupby("soa")["response_num"].mean()
    x = mean_responses.index.values  #soas
    print("X is", x)
    y = mean_responses.values

    #Prepare out plotting ranges
    x_smooth = np.linspace(x.min(), x.max(), 100)
    y_smooth = sigmoid(x_smooth, slope, pss) #To get the y, we plug the x into the sigmoid

    #Matplotlib

    plt.figure(figsize = (16,10)) #Create our graph with dimensions 16px by 10px

    plt.axvline(pss, linestyle = "--", label =f"PSS = {pss:.1f} ms") #Vertical line labeled where the PSS in in ms<- at what soa does the PSS occur

    #plotting
    plt.plot(x_smooth, y_smooth, label = "Psychometric Curve")

    plt.title("Psychometric curve")
    plt.xlabel("SOA(Ms)")
    plt.ylabel("% of light first")
    plt.savefig("Psychometric_curve.png")
    plt.show()


#Test function for the psychometric curve and PSS logic 
def test_psychometric_simulation():
    """Test psychometric curve fitting with simulated data"""
    print("\n" + "="*70)
    print("TESTING PSYCHOMETRIC CURVE FITTING")
    print("="*70)
    
    # Simulate behavioral data with known PSS
    true_pss = 25  # ms
    true_slope = 0.08
    
    print(f"Simulating data with PSS = {true_pss} ms, slope = {true_slope}")
    
    # Generate simulated responses,
    #Normally a participant would give you this data <- simulating it
    simulated_data = []
    for soa in SOA_VALUES:
        for _ in range(TRIALS_PER_SOA):
            # Probability of "light first" response
            p_light = sigmoid(soa, true_slope, true_pss)
            # Simulate response
            response = "L" if np.random.rand() < p_light else "S"
            simulated_data.append({
                "block": "TEST",
                "trial_number": len(simulated_data) + 1,
                "soa": soa,
                "response": response
            })
    
    # Create DataFrame
    df = pd.DataFrame(simulated_data)
    
    # Calculate PSS
    estimated_pss, estimated_slope, _ = calculate_pss(df)
    
    print(f"\nResults:")
    print(f"  True PSS: {true_pss} ms")
    print(f"  Estimated PSS: {estimated_pss:.1f} ms")
    print(f"  Error: {abs(estimated_pss - true_pss):.1f} ms")
    print(f"  True slope: {true_slope}")
    print(f"  Estimated slope: {estimated_slope:.4f}")
    
    # Plot
    plot_psychometric_curve(df, estimated_pss, estimated_slope)
    
    print("\n✓ Psychometric test complete!")
    print("="*70)



   

    







#Block 1: Baseline 
def run_block1(board, subject_id, subject_name):
     """"Records EEG data for 3 minutes on eyes opened and eyes closed before saving."""
     #Eyes closed
     print(f"Block 1 resting eeg of {subject_name}")
     board.start_stream() #Start the EEG recording
     input("Press enter to begin recording eeg. Keep your eyes closed during the recording.")
     block_closed_start= time.time() #start of the block
     time.sleep(180)
     block_closed_end= time.time() #end of the block
     
      #Eyes open
     input("Press enter to begin recording eeg. Keep your eyes open during the recording.")
     block_open_start = time.time()
     time.sleep(180)
     block_open_end= time.time() #end of the block

     #Save things to file
     sub_folder_block1 = f"{DIRECTORY_NAME}/block_1_data"
     os.makedirs(sub_folder_block1, exist_ok = True)
     csv_path = f"{sub_folder_block1}/sub_{subject_id}_resting.csv"
     with open(csv_path, 'w', newline = '') as f:
          writer = csv.writer(f)
          writer.writerow(['segment', 'start_time', 'end_time', 'duration (sec)', 'subject_id']) # row names 
          writer.writerow(['Eyes Closed', block_closed_start, block_closed_end, block_closed_end-block_closed_start, subject_id])
          writer.writerow(['Eyes Opened', block_open_start,block_open_end , block_open_end-block_open_start, subject_id])
     board.stop_stream()
        
     print("Block 1 Closed. ")




#Block 2: Open-Loop Pre-test
def run_block2(arduino, subject_id, subject_name):
    #instructions
    print("This is block 2. This is to estimate your PSS.")
    print("In this trial we will play a light and a buzzer. Press \"L\" if you think light came first, press \"S\" if you think sound came first, and \"=\" if they came at the same time. ")

    trials = []
    # -150 <- 20 times
    for i in range(len(SOA_VALUES)):
        for j in range(TRIALS_PER_SOA):
            trials.append(SOA_VALUES[i])
    
            
    random.shuffle(trials) #random order of SOA values
    block_2_dict = {
        "response": [],
        "soa": [],
        "trial_number": [],
        "block": [],
        "trial_start": [],
        "trial_end": [],
        "subject_id": [],
    }
    for i in range(len(trials)):
        
        print(f"Trial number {i+1} out of {len(trials)}")
        input("Press enter to continue")

        t_start = time.time()
        send_stimulus(ARDUINO_PORT, trials[i])

        
        response = input("If light came first, type L, if sound came first, type S, if they came at the same time, type =")
        while response.upper() not in ["S", "L", "="]:
             print("Invalid, Please always enter L, S, or =")
             response = input("If light came first, type L, if sound came first, type S, if they came at the same time, type =")

        t_end = time.time()

        #Save data
        block_2_dict["trial_start"].append(t_start) 
        block_2_dict["trial_end"].append(t_end)
        #fill in rest of the dict
        block_2_dict["soa"].append(trials[i])
        block_2_dict["trial_number"].append(i+1)
        block_2_dict["block"].append(2)
        block_2_dict["response"].append(response)
        block_2_dict["subject_id"].append(subject_id)

    print("Trial ended.")
    #path <- DIRECTORY_NAME/block_2/subid_block2.csv
    #TODO: Make a directory called block 2 data
    sub_folder_block2 = f"{DIRECTORY_NAME}/block_2_data"
    os.makedirs(sub_folder_block2, exist_ok = True)
    # TODO: Make a variable that stores this path you made
    block_2_dataframe = pd.DataFrame(block_2_dict) #table/excel spreadsheet
    """
    f"{sub_folder_block2}/sub_{subject_id}_block_2.csv"
    """
    file_location = f"{sub_folder_block2}/sub_{subject_id}_block_2.csv"
    block_2_dataframe.to_csv(file_location) #Save it file 
    


    #Calculate the PSS
    pss, slope, _ = calculate_pss(block_2_dataframe)
        
    #pandas.DataFrame(block_2_dict)
    
    plot_psychometric_curve(block_2_dataframe, pss, slope)
    return pss
     

     
#Block 3: Closed Loop Stimulation
#re-recoding neural data after task
def plot_phase_amplitude(phase_bins, gamma_by_phase, optimal_phase):
    plt.figure(figsize = (16,10))
    plt.bar(phase_bins, gamma_by_phase, width = 15, alpha = 0.8) # 0->20, ,20-40, height <- how much the gamma amplitude
    plt.xlabel("Theta phase bins in degrees", fontsize = 12)
    plt.ylabel("Mean Gamma Amplitudes", fontsize = 12)
    #Phase Amplitude Coupling
    # <- synchronizing the theta phase with the mei
    plt.title("Phase Amplitude Coupling. Theta vs Gamma")
    plt.axvline(optimal_phase, color = "green", linestyle = '--', linewidth =2, label = f"Optimal Phase: {optimal_phase}") #axis vertical line
    plt.xlim(0, 360)
    plt.xticks(np.arrange(0,361,20)) #range(0,2), 0,1
    plt.legend(fontsize = 12)
    plt.show()
    plt.savefig("Subject_PAC_Block3.png")
    plt.close()

def run_block3(arduino, board, pss, subject_id, subject_name):
    neural_dict = {
        "block": [],#always going to be 3
        "trial_number": [],
        "pss_ms": [],
        "theta_phase_at_stim": [],
        "gamma_amplitude_mean": [],
        "start_time": [],
        "end_time": [],
        "optimal_phase": [],
        "subject_id": [],
    }
    print("Block 3 is starting, stimuli will be delivered at PSS.")
    input("Brain activity will be recorded. Press enter when ready.")

    board.start_stream() #Start the EEG recording
    print("Trials are starting.")
    
    for i in range(BLOCK3_TRIALS):
        input(f"You are on trial {i+1} out of {BLOCK3_TRIALS}. To continue, press enter.")
        eeg_data = get_eeg_data(board)
        filtered_theta = bandpass_filter(eeg_data, EEG_SAMPLING_RATE, THETA_BAND[0], THETA_BAND[1]) #List of signals

        #Compute Phase
        phases = compute_phase(filtered_theta) #List of phases <- degree <- radians
        theta_phase_at_stimulation = phases[-1] 

        #Manipulated variable <- what phase is ideal to reach PSS

        t_start = time.time()
        send_stimulus(arduino, pss) #Light and sound <- pss <- timestamp <- when the person was 50% likely to view light and sound as the same

        time.sleep(0.15) #Arduino hardware<- time to recoup

        eeg_post = get_eeg_data(board)

        #Process the gamma band data

        # 1) bandpass filter for gamma on the eeg_post
        filtered_gamma_eeg_post = bandpass_filter(eeg_post, EEG_SAMPLING_RATE, GAMMA_BAND[0], GAMMA_BAND[1]) # a list of gamma signals
        #2) Amplitude of the gamma frequencies
        post_eeg_gamma_amplitude = compute_amplitude(eeg_post) #List of amplitudes that correspond to those signals
        #3) Mean amplitude of the gamma activity 
        mean_gamma_activity = 0 
        for i in range(len(post_eeg_gamma_amplitude)):
            mean_gamma_activity += post_eeg_gamma_amplitude[i]
        mean_gamma_activity/len(post_eeg_gamma_amplitude) 

        t_end = time.time() #stop recording
    
        #saving the data to our data dictionary
        neural_dict["block"].append(3)
        neural_dict["trial_number"].append(i+1) #Follow this format
        #pss_ms
        neural_dict["pss_ms"].append(pss)
        #theta_phase_at_stim
        neural_dict["theta_phase_at_stim"].append(theta_phase_at_stimulation)
        #gamma_amplitude_mean
        neural_dict["gamma_amplitude_mean"].append(mean_gamma_activity)
        #start_time 
        neural_dict["start_time"].append(t_start)
        #end_time
        neural_dict["end_time"].append(t_end)
        neural_dict["subject_id"].append(subject_id)

    
    board.stop_stream() # We no longer need to be recording the EEG
    optimal_phase, gamma_by_phase, phase_bins = find_optimal_phase(
        theta_phase_at_stimulation, 
        mean_gamma_activity,
        bin_width=20
    )
    block_3_dataframe = pd.DataFrame(neural_dict) #table/excel spreadsheet
    sub_folder_block3 = f"{DIRECTORY_NAME}/block_3_data"
    os.makedirs(sub_folder_block3, exist_ok = True)
    file_location_block3 = f"{sub_folder_block3}/sub_{subject_id}_block_3.csv"
    block_3_dataframe.to_csv(file_location_block3)
    return optimal_phase

    
    
    

    



        #what is the starting phase
    
#Block 4: Find best Phase Optimization

def phase_error(phase1, phase2):
    """what is the distance between the 2 phases"""
    distance = abs(phase2 - phase1) %360 #distance is always positive, %360
    return distance if distance <= 180 else 360 -distance
def wait_for_target_phase(board, target_phase, tolerance_deg = 15, window_simples = 25, max_wait_sec = 5):
    t_start = time.time() # HH:MM:SS, gives you the current timestamp
    while True: #break 
        eeg_data = get_eeg_data(board,window_simples)
        filtered_eeg_data = bandpass_filter(eeg_data, EEG_SAMPLING_RATE, THETA_BAND[0], THETA_BAND[1])
        current_phase = compute_phase(filtered_eeg_data)[-1]
        error_phase = phase_error(current_phase, target_phase) #how many degrees off am I 
        # 15 or less <- good
        #bad <- keep looping
    
        if error_phase <= tolerance_deg:
            return current_phase
        t_end = time.time()
        if t_end-t_start > max_wait_sec:
            print("Exceeded max wait time, returning current phase.")
            return current_phase
        
        time.sleep(0.005) #waiting time

    
        


def run_block4(arduino, board, pss, optimal_phase, subject_id, subject_name):
    print("This will flash stimuli at different times.")
    print("Press L for light first, S for sound first, or = for if they were equal.")
    trials = []
    # -150 <- 20 times
    for i in range(len(SOA_VALUES)):
        for j in range(TRIALS_PER_SOA):
            trials.append(SOA_VALUES[i])
    block_4_dict = {
        "response": [],
        "soa": [], 
        "trial_number": [],
        "block": [],
        "trial_start": [],
        "trial_end": [],
        "optimal_phase":[],
        "actual_phase_degree":[],
        "subject_id":[],
    }
    board.start_stream
    for i in range(len(trials)): #[150, -150, ] trials[i]<
        
        print(f"Trial number {i+1} out of {len(trials)}")
        input("Press enter to continue")

        t_start = time.time()
        phase_degree = wait_for_target_phase(board, optimal_phase)
        send_stimulus(ARDUINO_PORT, trials[i])
        response = input("If light came first, type L, if sound came first, type S, if they came at the same time, type =")
        while response.upper() not in ["S", "L", "="]:
             print("Invalid, Please always enter L, S, or =")
             response = input("If light came first, type L, if sound came first, type S, if they came at the same time, type =")
 
        t_end = time.time()

        block_4_dict["response"].append(response)
        block_4_dict["soa"].append(trials[i])
        block_4_dict["trial_number"].append(i)
        block_4_dict["block"].append(4)
        block_4_dict["trial_start"].append(t_start)
        block_4_dict["trial_end"].append(t_end)
        block_4_dict["optimal_phase"].append(optimal_phase)
        block_4_dict["actual_phase_degree"].append(phase_degree)
        block_4_dict["subject_id"].append(subject_id)


    print("Trial ended.")
    block_4_dataframe = pd.DataFrame(block_4_dict) #table/excel spreadsheet
    sub_folder_block4 = f"{DIRECTORY_NAME}/block_4_data"
    os.makedirs(sub_folder_block4, exist_ok = True)
    file_location_block4 = f"{sub_folder_block4}/sub_{subject_id}_block_4.csv"
    block_4_dataframe.to_csv(file_location_block4)


    #Calculate the PSS
    pss, slope, _ = calculate_pss(block_4_dataframe)
            
    #pandas.DataFrame(block_4_dict)
        
    plot_psychometric_curve(block_4_dataframe, pss, slope)
            

#Block 5: Open Loop Post test (Did PSS Adapt)

#Block 6: Statistical Analysis (P-Value and t-test) Significant difference between open and closed loop

def main():
    BRAIN_DATA = "BrainData"
    os.makedirs(BRAIN_DATA, exist_ok = True) #making the folder

    #subject information

    subject_registry = load_subject_registry() #{"subjects": [A.S., J.S., ], "next_id": 4}
    subjects_lst =subject_registry["subjects"]
    if subjects_lst:
        print("Subject initials    Subject ID")
        for i in range(len(subjects_lst)):
            print(f"{subjects_lst[i]} {i+1}")
    subject_initials = input("What are your initials? ")
    subjects_lst.append(subject_initials)
    current_subject_id = subject_registry["next_id"]
    subject_registry["next_id"] += 1
    save_subject_registry(subject_registry)
    # go through all blocks for this subject
    #Connnect the EEG
    connect_arduino()
    connect_eeg()

    # #Block 1
    # run_block1(EEG_BOARD_ID, current_subject_id, subject_initials)
    # # Block 2
    #  #returns pss
    # current_pss = run_block2(ARDUINO_PORT, current_subject_id, subject_initials)
    # # Block 3
    
    # optimal_phase = run_block3(ARDUINO_PORT, EEG_BOARD_ID, current_pss, current_subject_id, subject_initials)

    # # Block 4   
    # run_block4(ARDUINO_PORT, EEG_BOARD_ID, current_pss, optimal_phase, current_subject_id, subject_initials)

if __name__ == "__main__":
    main()
    

#python PSS.py <- run this file








