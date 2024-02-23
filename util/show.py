import matplotlib.pyplot as plt

def show_ecg(ecg_data,figsize=(30,15)):
    figure1 = plt.figure(figsize=figsize)
    
    ecg_place = {0:0,1:2,2:4, 3:6, 4:8, 5:10, 6:1, 7:3, 8:5, 9:7, 10:9, 11:11}
    
    max_scale = ecg_data.max()
    min_scale = ecg_data.min()

    for j in range(len(ecg_data)):
        # plot 12 leads on one figure with subplot

        # ax_figure.subplots(6,2,i+1)
        ax1 = figure1.add_subplot(6,2,j+1)
        ax1.set_ylim([min_scale,max_scale])
        ax1.plot(ecg_data[j])
        
        
    plt.show()