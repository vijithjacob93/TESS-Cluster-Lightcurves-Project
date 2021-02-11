### PROGRAM TO PLOT GIF OF LOMB-SCARGLE PERIODOGRAM POWER FOR INDIVIDUAL PIXELS IN DIFFERENT FREQUENCY RANGES ###


## import required packages ##

import glob
import lightkurve as lk
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from astropy.stats import LombScargle
from astropy.table import Table
import imageio


## all required functions ##

def get_LCs(CLUSTERS,phot_type):
    NUMBER_PCA_COMPONENTS = 5 #no of vectors for PCA
    #initialize the masks
    star_mask=np.empty([len(tpfs),cutout_size,cutout_size],dtype='bool')
    if phot_type == 'pixel': # if pixel photometry is requested, define a different variable
        radius_mask=np.empty([len(tpfs),cutout_size,cutout_size],dtype='bool')
    sky_mask=np.empty([len(tpfs),cutout_size,cutout_size],dtype='bool')

    frame = tpfs[0].shape[0]//2
    # obtain the aperture and the background masks with help of cluster radius and percentile (85%)
    if phot_type == 'pixel':
        radius_mask[0],sky_mask[0] = circle_aperture(tpfs[0][frame].flux,tpfs[0][frame].flux,                Complete_Clusters[Complete_Clusters['NAME'] == CLUSTERS[0]]['CORE_RADIUS'].values[0],85)
    elif phot_type == 'ensemble': 
        star_mask[0],sky_mask[0] = circle_aperture(tpfs[0][frame].flux,tpfs[0][frame].flux,            Complete_Clusters[Complete_Clusters['NAME'] == CLUSTERS[0]]['CORE_RADIUS'].values[0],85)

    # if pixel photometry, iterate over each pixel in the radius mask and download LCs
    if phot_type == 'pixel':
        for i in range(len(np.where(radius_mask[0])[0])):
            star_mask=np.zeros([len(tpfs),cutout_size,cutout_size],dtype='bool')
            pixel_loc=np.where(radius_mask[0])[0][i],np.where(radius_mask[0])[1][i]
            star_mask[0][pixel_loc]=True

            lightcurves = [tpfs[i].to_lightcurve(aperture_mask=star_mask[i]) for i in range(len(tpfs))];
            for i in range(len(lightcurves)):
                lightcurves[i]=lightcurves[i][lightcurves[i].flux_err > 0]
            
            #PCA correction
            regressors = [tpfs[i].flux[:, sky_mask[i]] for i in range(len(tpfs))]# The regressor is the inverse of the aperture
            #Design a matrix for linear regression
            dms = [lk.DesignMatrix(r, name='regressors').pca(NUMBER_PCA_COMPONENTS).append_constant() for r in regressors]
            #Remove noise using linear regression against a DesignMatrix.
            correctors = [lk.RegressionCorrector(lc) for lc in lightcurves]
            correctedLCs = [correctors[i].correct(dms[i]) for i in range(len(correctors))]#this is how to subtract the background noise from the star pixels

            #polyomial correction. to remove long period systematic trends that survived the PCA correction
            correctedLCs_poly,poly_terms = polynomial_correction(correctedLCs[0])
            #save the pixel corrected LCs
            correctedLCs_poly.to_csv(CLUSTERS[0]+' pixels/corrected_LCs_CORE_RADIUS_pixel_no.{0}'.format(pixel_loc))

    # is ensemble photometry, download LC for entire aperture
    elif phot_type == 'ensemble':
        lightcurves = [tpfs[i].to_lightcurve(aperture_mask=star_mask[i]) for i in range(len(tpfs))];
        for i in range(len(lightcurves)):
            lightcurves[i]=lightcurves[i][lightcurves[i].flux_err > 0]
        regressors = [tpfs[i].flux[:, sky_mask[i]] for i in range(len(tpfs))]# The regressor is the inverse of the aperture
        #Design a matrix for linear regression
        dms = [lk.DesignMatrix(r, name='regressors').pca(NUMBER_PCA_COMPONENTS).append_constant() for r in regressors]
        #Remove noise using linear regression against a DesignMatrix.
        correctors = [lk.RegressionCorrector(lc) for lc in lightcurves]
        correctedLCs = [correctors[i].correct(dms[i]) for i in range(len(correctors))]
        #polynomial correction
        correctedLCs_poly,poly_terms = polynomial_correction(correctedLCs[0])
        
        #calculate the lomb-scargle periodogram for the ensemble corrected LC for a range frequencies
        t = correctedLCs_poly[correctedLCs_poly.quality == 0].time
        dy = correctedLCs_poly[correctedLCs_poly.quality == 0].flux_err
        y = correctedLCs_poly[correctedLCs_poly.quality == 0].flux
        omega=np.arange(0.05,11,0.01)
        P_LS = LombScargle(t, y, dy=dy).power(omega)# LS power
#         max_power=np.max(P_LS)
#         freq_at_max_power=omega[np.argmax(P_LS)]
        return omega, P_LS #return frequencies and LS power


#Polynomial correction removing trends that span ranges larger than the time windows in each sector (~13 days)
def polynomial_correction(correctedLCs):
    poly_terms = np.zeros([len(start_times),3])
    correctedLCs_poly = correctedLCs.copy()
    for i in range(len(start_times)):
        poly_terms[i] = np.polyfit(correctedLCs[np.logical_and.reduce([correctedLCs.time>start_times[i],            correctedLCs.time<end_times[i],correctedLCs.quality == 0])].time,                   correctedLCs[np.logical_and.reduce([correctedLCs.time>start_times[i],            correctedLCs.time<end_times[i],correctedLCs.quality == 0])].flux,deg=2)
        correctedLCs_poly.flux[np.logical_and(correctedLCs.time>start_times[i],correctedLCs.time<end_times[i])]            =correctedLCs_poly[np.logical_and(correctedLCs.time>start_times[i],                                              correctedLCs.time<end_times[i])].flux            -np.polyval(poly_terms[i],correctedLCs_poly[np.logical_and(correctedLCs.time>start_times[i],                                                                       correctedLCs.time<end_times[i])].time)
    return correctedLCs_poly,poly_terms

def degs_to_pixels(degs):
    return degs*60*60/21 #convert degrees to arcsecs and then divide by the resolution of TESS (21 arcsec per pixel)

def pixels_to_degs(pixels):
    return pixels*21/(60*60) #convert degrees to arcsecs and then divide by the resolution of TESS (21 arcsec per pixel)

UPPER_LIMIT_METHOD = 1

# Global values for method 1
PERCENTILE = 80

# Global values for method 2
BINS = 300

# Global values for method 3


def getUpperLimit(dataDistribution,PERCENTILE):
    if UPPER_LIMIT_METHOD == 1:
        return np.nanpercentile(dataDistribution, PERCENTILE)
    
    elif UPPER_LIMIT_METHOD == 2:
        hist = np.histogram(dataDistribution, bins=BINS, range=(0, 3000))# Bin the data
        return hist[1][np.argmax(hist[0])]# Return the flux corresponding to the most populated bin
    
    elif UPPER_LIMIT_METHOD == 3:
        pass
    
    elif UPPER_LIMIT_METHOD == 4:
        numMaxima = countMaxima(tpfs[i][frame].flux.reshape((cutout_size, cutout_size)))
        numPixels = np.count_nonzero(~np.isnan(tpfs[i][frame].flux))
        return np.nanpercentile(dataDistribution, 100 - numMaxima / numPixels * 100)
        
    else:
        return 150
    
#calculate aperture for cluster given a radius and flux percentile
def circle_aperture(data,bkg,radius,PERCENTILE):
    radius_in_pixels=degs_to_pixels(radius)
    data_mask = np.zeros_like(data)
    x_len=np.shape(data_mask)[1]
    y_len=np.shape(data_mask)[2]
    #centers
    cen_x=x_len//2
    cen_y=y_len//2
    bkg_mask = np.zeros_like(bkg)
    bkg_cutoff = getUpperLimit(bkg,PERCENTILE)
    print(bkg_cutoff)
    for i in range(x_len):
        for j in range(y_len):
            if (i-cen_x)**2+(j-cen_y)**2<(radius_in_pixels)**2:# star mask condition
                data_mask[0,i,j]=1
                
    x_len=np.shape(bkg_mask)[1]
    y_len=np.shape(bkg_mask)[2]
    cen_x=x_len//2
    cen_y=y_len//2
    for i in range(x_len):
        for j in range(y_len):
            if np.logical_and((i-cen_x)**2+(j-cen_y)**2>(radius_in_pixels)**2, bkg[0,i,j]<bkg_cutoff): # sky mask condition
                bkg_mask[0,i,j]=1            

    star_mask = data_mask==1
    sky_mask = bkg_mask==1
#     sky_mask_2 = bkg_mask_2==1
#     field_mask = field==1
    return star_mask,sky_mask# return masks


## START HERE ##

#Read and process the Kharchenko catalog which contains info about clusters
Complete_Clusters=Table.read('../../../Documents/Data_Files/Cluster_Catalog_Kharchenko_updated.fits')
Complete_Clusters=Complete_Clusters.to_pandas()
Complete_Clusters['CLUSTER_RADIUS']=Complete_Clusters['CLUSTER_RADIUS']
for i in range(len(Complete_Clusters)):
    Complete_Clusters['NAME'][i] = Complete_Clusters['NAME'][i].decode("utf-8").strip()


## Download target pixel file ##

# input name of the interested cluster
CLUSTERS = ["NGC 2422"]

# initialize the target pixel file and cutout size
tpfs = [0]
cutout_size = 99
search = lk.search_tesscut(CLUSTERS[0])#search for the cluster in TESS using lightkurve
char = ""
if len(search) != 1: char = "s"
print("{0} has {1} result{2}.".format(CLUSTERS, len(search), char))
tpfs = search.download_all(cutout_size=cutout_size)# download target pixel file for corresponding cluster

sectors = [this_tpfs.sector for this_tpfs in tpfs] #sectors cluster was observed in
orbit_times = pd.read_csv('orbit_times_20201013_1338.csv', comment = '#')# read the file containing epoch info of the sectors
start_times = orbit_times[orbit_times['Sector'].isin(sectors)]['Start TJD'].values # start times for the sectors that this cluster was observed in
end_times = orbit_times[orbit_times['Sector'].isin(sectors)]['End TJD'].values # end times for the sectors

#get the frequency and Lomb-Scargle periodogram power for the ensemble photometry of the cluster
omega, P_LS = get_LCs(CLUSTERS,'ensemble')

plt.plot(omega,P_LS)
plt.xscale('log') # quick plot of the periodogram


## Calculate the periodogram powers for LCs from individual pixels ##

#initialize variables
pixel_loc_write=[]
P_LS_pixel=[]
omega=np.arange(0.05,11,0.01)
radius_mask=np.empty([len(tpfs),cutout_size,cutout_size],dtype='bool')

#check if LCs have already been downloaded, corrected, and saved. if not, do it
if len(glob.glob(CLUSTERS[0]+' pixels/core_pixels'))==0:
    get_LCs(CLUSTERS,'pixel')
    
#define the radius mask based on the cluster radius
for i in range(len(tpfs)):
    frame = tpfs[0].shape[0]//2
    radius_mask[i],_ = circle_aperture(tpfs[i][frame].flux,tpfs[i][frame].flux,Complete_Clusters[Complete_Clusters['NAME'] == CLUSTERS[0]]['CORE_RADIUS'].values[0],85)

for i in range(len(np.where(radius_mask[0])[0])):
	# the pixel locations
    pixel_loc=np.where(radius_mask[0])[0][i],np.where(radius_mask[0])[1][i]

	#read in the corrected LCs of individual pixels
    correctedLCs_poly = pd.read_csv(CLUSTERS[0]+' pixels/core_pixels/'
                                    'corrected_LCs_CENTRAL_RADIUS_pixel_no.{0}'.format(pixel_loc),index_col=0)
    t = correctedLCs_poly['time']
    dy = correctedLCs_poly['flux_err']
    y = correctedLCs_poly['flux']
    P_LS_pixel.append(LombScargle(t, y, dy=dy).power(omega))# calculate and save the LS periodogram power
    pixel_loc_write.append(pixel_loc)#save the pixel location


## Make the individual plots of pixel power for different frequency ranges ##

#define the frequency range to iterate over in log space
freq_bin_centers=np.logspace(-1, 1, num=50)
for freq_index in range(len(freq_bin_centers)-1):#iterate over the frequency range except for the last point
    freq_range = freq_bin_centers[freq_index+1]-freq_bin_centers[freq_index]# select a range of frequencies to search pixel power in
    radius_in_pixels=degs_to_pixels(Complete_Clusters[Complete_Clusters['NAME'] == CLUSTERS[0]]['CORE_RADIUS'].values[0]) # radius of the cluster in pixels

    fig, (ax1, ax2, ax3)=plt.subplots(ncols = 3, figsize=(18,4))#initialize plot parameters
    p = tpfs[0][frame].plot(frame=tpfs[0][0].shape[0] // 2,ax=ax1)# plot of the flux in the cutout
    a = plt.Circle((ax1.get_xlim()[0]+(ax1.get_xlim()[1]-ax1.get_xlim()[0])/2, ax1.get_ylim()[0]+(ax1.get_ylim()[1]-ax1.get_ylim()[0])/2), radius_in_pixels, color='b', fill=False, linestyle='--')# circle of the cluster radius
    ax2.imshow(radius_mask[0],extent=list(ax1.get_xlim())+list(ax1.get_ylim()), origin = 'lower' , cmap = 'Reds', alpha = 0.5, label = 'Star')
    ax2.add_patch(a)# add circle

    max_power=np.zeros([len(tpfs),cutout_size,cutout_size],dtype='float64')
    pixel_loc_x_list=[]
    pixel_loc_y_list=[]
    max_power_list=[]
    for pixel_index in range(len(pixel_loc_write)):# iterate over the pixel locations
        pixel_loc_x_list.append(pixel_loc_write[pixel_index][1])
        pixel_loc_y_list.append(pixel_loc_write[pixel_index][0])
        max_power[0,pixel_loc_write[pixel_index][0],pixel_loc_write[pixel_index][1]] = np.max(P_LS_pixel[pixel_index][np.logical_and(freq_bin_centers[freq_index]-freq_range/2<omega, omega<freq_bin_centers[freq_index]+freq_range/2)]) # maximum of the periodogram power of the LC from this pixel within the frequency range being searched
    ax1.scatter(ax1.get_xlim()[0]+pixel_loc_x_list+1,ax1.get_ylim()[0]+pixel_loc_y_list+1,c='r', cmap='Greys',s = 1,alpha=0.5) # plot the positions of the pixels being looked at   
    im=ax2.imshow(max_power[0,:,:],extent=list(ax1.get_xlim())+list(ax1.get_ylim()), origin = 'lower', cmap='Greys', vmax=1.0) # plot the max pixel power corresponding to each pixel
    # axes limits set depending on the cluster
    ax1.set_xlim(60,95)
    ax1.set_ylim(980,1015)
    ax2.set_xlim(60,95)
    ax2.set_ylim(980,1015)
    # axes adjustments
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_yticks(ax1.get_yticks())
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_title('Aperture (star pixels)')
    
    cbar = fig.colorbar(im,ax=ax2)#add colorbar
    ax1.set_title(CLUSTERS[0]+'; ('+str(tpfs[0].ra)+', '+str(tpfs[0].dec)+')')
    cbar.set_label('L-S periodogram power')
    ax2.set_title('Aperture + L-S power at ~{0} freq. range'.format(round(freq_bin_centers[freq_index],2)))

	# plot the LS periodoram for the ensemble cluster LC
	ax3.plot(omega,P_LS,color='k',linewidth=1)
    ax3.axvline(freq_bin_centers[freq_index]-freq_range/2,color='b',linestyle='dashed')
    ax3.axvline(freq_bin_centers[freq_index]+freq_range/2,color='b',linestyle='dashed')
    ax3.set_xscale('log')
    ax3.set_xlabel('Frequency (1/day)')
    ax3.set_ylabel('Power')
    ax3.set_title('Lomb-Scargle periodogram for {0}'.format(CLUSTERS[0]))

    fig.suptitle('Pixels for {0}: gif plots (freq={1})'.format(CLUSTERS[0],round(freq_bin_centers[freq_index],2)))
#     fig.savefig('pixel gif plots/Pixels_for_{0}:_gif_plots_(freq={1}).png'.format(CLUSTERS[0],round(freq_bin_centers[freq_index],2)))# save those plots
    fig.show()

## Make the gif using the individual plots ##

# set the gif destination path and the path to find the frames in
gif_path = "pixel gif plots/{0}/pixel_power.gif"
frames_path = "pixel gif plots/{0}/Pixels_for_{0}:_gif_plots_(freq={1}).png"
with imageio.get_writer(gif_path.format(CLUSTERS[0]), mode='I',fps=1) as writer:# open gif writer
    for freq_index in range(len(freq_bin_centers)-1):# iterate over the frequencies
        writer.append_data(imageio.imread(frames_path.format(CLUSTERS[0],round(freq_bin_centers[freq_index],2))))#write the gif

