# DKI-TMS Word Learning Study
# TODO: Describe study design and data structure etc.

# IMPORTS
import numpy as np
import pandas as pd
import os
import time
import logging
from sklearn.cross_validation import StratifiedKFold
from nilearn.decoding import SpaceNetClassifier
from sklearn.externals import joblib

# SETUP LOGGING
# Go to project working directory (file paths will be relative)
os.chdir("/projects/MINDLAB2016_TMS-NovelWordKurtosis/scratch/MVPA/")

# Get timestamp as string
timestamp = time.strftime('%Y%m%d-%H%M%S')
logname = './%s_SpaceNet_log.txt' %(timestamp)
logging.basicConfig(filename=logname,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s : %(message)s',
                    datefmt='%d %b %Y %H:%M:%S')

# 0. ANALYSIS PARAMETERS
inputcsv = 'MDinfo.csv' # csv containing participant info and file paths
testday = 1 # select data from day 1 or 2
excludegroup = 3 # analyse groups 1 and 2
num_folds = 10 # number of KFolds
fitpenalty = 'tv-l1' # regularisation using tv-l1 or graph-net

######################################################################

# 1. LOAD THE DATA
# Load csv
inputdata = pd.read_csv(@inputcsv)
# Select a subset of data based on testing day and tms group (1=M1,2=SPL,3=M1Control)
datasubset = inputdata.query('day==@testday & tms!=@excludegroup')
# Get array X of .nii file paths from the "image" column
X = np.array(datasubset['image'])
# Get array Y of subject group indices (1=M1,2=SPL,3=M1Control)
y = np.array(datasubset['tms'])

logging.info('Data loaded.')

# Sort data for better visualization
perm = np.argsort(y)
y = y[perm]
X = X[perm]

# Load the MNI template brain, and the corresponding mask
mnibrain = "./mni152_template/mni_152_t1_brain.nii"
mnimask = "./mni152_template/mni_152_t1_mask.nii"

logging.info('MNI template and mask loaded.')

# 2. CONSTRUCT TRAINING & TEST DATA
# Use StratifiedKFold cross-validation generator: 10 splits
cv = StratifiedKFold(n_folds=num_folds,y=y)

logging.info('Cross validation set up.')

# 3. CREATE AND FIT MODEL
# Define SpaceNetClassifier with tv-l1/graph-net penalty
decoder = SpaceNetClassifier(cv=cv, memory="nilearn_cache", penalty=fitpenalty,
                            memory_level=2, n_jobs=1, mask=mnimask)
logging.info('Classifier set up. Starting model fit.')
decoder.fit(X, y)  # fit
logging.info('Model fit complete. Saving outputs...')

# 4. OUTPUTS

# Save decoder object
decoder_dir = './decoder/' # directory
if not os.path.exists(decoder_dir): os.makedirs(decoder_dir) # create if missing
dec_filename = '%s%s_decoder_SpaceNet_%s_day%d.jbl' %(decoder_dir,timestamp, decoder.penalty, testday)
joblib.dump(decoder, dec_filename)

# Save coefficients to nifti file
coef_dir = './coefs/' # directory
if not os.path.exists(coef_dir): os.makedirs(coef_dir) # create if missing
coef_filename = '%s%s_coefs_SpaceNet_%s_day%d.nii' %(coef_dir,timestamp, decoder.penalty, testday)
coef_img = decoder.coef_img_
coef_img.to_filename(coef_filename)
logging.info('All outputs saved.')




# TODO: VISUALISE COEFICIENTS
'''coef_img = decoder.coef_img_
plot_stat_map(coef_img, background_img,
              title='SpaceNet Coeficients',
              cut_coords=(-52, -5), display_mode='yz')

# for 3d surface plots, see: http://nilearn.github.io/auto_examples/01_plotting/plot_surf_stat_map.html '''
