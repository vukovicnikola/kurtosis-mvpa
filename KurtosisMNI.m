%% Normalise native Diffusion Kurtosis maps into MNI space; produce difference maps between baseline and days 1 and 2

%% Project Parameters;
project= 'MINDLAB2016_TMS-NovelWordKurtosis';


% get the handle to the StormDB database
try
    load dbhandle
catch
    dbhandle=stormdb_connect_db;
end


subjects=stormdb_get_subjects(dbhandle,project);

% Create jobstruct for all subjects. NOTE: Subjects 6 and 39 are missing Session 3.
for i=1:numel(subjects)
    jobstruct(i).subject=subjects(i);
    jobstruct(i).subjectnum=i;
    jobstruct(i).project=project;
    jobstruct(i).modality='MR'; 
    jobstruct(i).branch='mar4';
    jobstruct(i).typetoalign='KURTOSIS_FSL_ADC'; % StormDB filetype to compute mni transform
    jobstruct(i).fittype='KURTOSIS_DKITOOLSFULL_MK'; % StormDB filetype to write to mni, e.g. 'KURTOSIS_DKITOOLSFULL_MD' or 'KURTOSIS_DKITOOLSFULL_MK'
    jobstruct(i).dbhandle=dbhandle;
end

% cluster queue parameters
clusterconfig('long_running',1)
jobid=job2cluster(@normalise_data,jobstruct);

function out=normalise_data(jobstruct)

out=[];

studies = stormdb_get_studies(jobstruct(1).dbhandle,jobstruct(1).project,jobstruct(1).subject); % retrieve study times
study1 = datestr(datenum(studies{1}), 'yyyymmdd_HHMMSS'); % reformat study times to fit StormDB folder convention
study2 = datestr(datenum(studies{2}), 'yyyymmdd_HHMMSS');
study3 = datestr(datenum(studies{3}), 'yyyymmdd_HHMMSS');
outdir = strcat('/projects/MINDLAB2016_TMS-NovelWordKurtosis/scratch/MVPA/INPUT/',jobstruct(1).fittype,'/'); % save output here

% construct input file paths for the images to be normalised
extrapars.isnifti=1; % toggles between info or data (if =1)
extrapars.ispermanent=0; %toggles between scratch and misc (misc=1 default)

sess1refpath = stormdb_std_output_path(jobstruct(1).project,jobstruct(1).branch,jobstruct(1).subject,study1,jobstruct(1).modality,jobstruct(1).typetoalign,extrapars); % directories 
sess2refpath = stormdb_std_output_path(jobstruct(1).project,jobstruct(1).branch,jobstruct(1).subject,study2,jobstruct(1).modality,jobstruct(1).typetoalign,extrapars);
sess3refpath = stormdb_std_output_path(jobstruct(1).project,jobstruct(1).branch,jobstruct(1).subject,study3,jobstruct(1).modality,jobstruct(1).typetoalign,extrapars);

sess1path = stormdb_std_output_path(jobstruct(1).project,jobstruct(1).branch,jobstruct(1).subject,study1,jobstruct(1).modality,jobstruct(1).fittype,extrapars); % directories for images to-be-normalised
sess2path = stormdb_std_output_path(jobstruct(1).project,jobstruct(1).branch,jobstruct(1).subject,study2,jobstruct(1).modality,jobstruct(1).fittype,extrapars);
sess3path = stormdb_std_output_path(jobstruct(1).project,jobstruct(1).branch,jobstruct(1).subject,study3,jobstruct(1).modality,jobstruct(1).fittype,extrapars);

sess1alignimage = strcat(sess1refpath,'/0001.nii'); % files to compute the mni warps from
sess2alignimage = strcat(sess2refpath,'/0001.nii');
sess3alignimage = strcat(sess3refpath,'/0001.nii');

sess1image = strcat(sess1path,'/0001.nii'); % files in native space to apply the warps on
sess2image = strcat(sess2path,'/0001.nii');
sess3image = strcat(sess3path,'/0001.nii');

sess1mniimage = strcat(sess1path,'/mni_0001.nii'); % output files from above, now in MNI space
sess2mniimage = strcat(sess2path,'/mni_0001.nii');
sess3mniimage = strcat(sess3path,'/mni_0001.nii');

outdir = strcat('/projects/MINDLAB2016_TMS-NovelWordKurtosis/scratch/MVPA/INPUT/',jobstruct(1).fittype,'/'); % output directory for day1 and day2 change images
day1changefile = strcat(num2str(jobstruct(1).subject),'_',jobstruct(1).fittype,'_day1change');
day2changefile = strcat(num2str(jobstruct(1).subject),'_',jobstruct(1).fittype,'_day2change');

jobfile = {'/projects/MINDLAB2016_TMS-NovelWordKurtosis/scratch/MVPA/KurtosisMNI_job.m'};
jobs = repmat(jobfile, 1, 1);
inputs = cell(14, 1);
inputs{1, 1} = {sess1alignimage}; % Normalise: Estimate & Write: Align Session 1
inputs{2, 1} = {sess1image}; % Normalise: Estimate & Write: Write Session 1
inputs{3, 1} = {sess2alignimage}; % Normalise: Estimate & Write: Align Session 2
inputs{4, 1} = {sess2image}; % Normalise: Estimate & Write: Write Session 2
inputs{5, 1} = {sess3alignimage}; % Normalise: Estimate & Write: Align Session 3
inputs{6, 1} = {sess3image}; % Normalise: Estimate & Write: Write Session 3
inputs{7, 1} = {sess1mniimage; sess2mniimage}; % Image Calculator: Input Images for Session 2 - Session 1
inputs{8, 1} = day1changefile; % Image Calculator: Output Filename
inputs{9, 1} = {outdir}; % Image Calculator: Output Directory
inputs{10, 1} = 'i2-i1'; % Image Calculator: Expression
inputs{11, 1} = {sess1mniimage; sess3mniimage}; % Image Calculator: Input Images for Session 3 - Session 1
inputs{12, 1} = day2changefile; % Image Calculator: Output Filename
inputs{13, 1} = {outdir}; % Image Calculator: Output Directory
inputs{14, 1} = 'i2-i1'; % Image Calculator: Expression

spm('defaults', 'FMRI');
spm_jobman('run', jobs, inputs{:});

disp (['finished processing subject ' num2str(jobstruct(i).subjectnum)])

end
