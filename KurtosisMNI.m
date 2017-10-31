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
    jobstruct(i).group=stormdb_get_subjectmeta(dbhandle, project, subjects(i), 'group');
    jobstruct(i).project=project;
    jobstruct(i).modality='MR'; 
    jobstruct(i).branch='mar4';
    jobstruct(i).fitref='KURTOSIS_DKITOOLSFULL_MD'; %ref image for computing mni transform
    jobstruct(i).fittype='KURTOSIS_DKITOOLSFULL_MD'; % StormDB filetype to write to mni, e.g. 'KURTOSIS_DKITOOLSFULL_MD' or 'KURTOSIS_DKITOOLSFULL_MK'
    jobstruct(i).dbhandle=dbhandle;
end

% Create meta info struct and csv
r=1; %row index
for m=1:numel(subjects)
    metainfo(r).subject=subjects(m);
    metainfo(r).hand='right';
    metainfo(r).tms=stormdb_get_subjectmeta(dbhandle, project, subjects(m), 'group'); % 1="M1" 2="SPL" 3="M1Control"
    metainfo(r).project=project;
    metainfo(r).branch='mar4';
    metainfo(r).parameter=jobstruct(m).fittype;
    if size(stormdb_get_studies(jobstruct(m).dbhandle,jobstruct(m).project,jobstruct(m).subject),1) == 3
        metainfo(r).image = strcat('INPUT/',jobstruct(m).fittype,'/',num2str(jobstruct(m).subject),'_',jobstruct(m).fittype,'_day1change.nii');
        
        r = r+1; %add new row for new test day
        metainfo(r).subject=subjects(m);
        metainfo(r).hand='right';
        metainfo(r).tms=stormdb_get_subjectmeta(dbhandle, project, subjects(m), 'group'); % 1="M1" 2="SPL" 3="M1Control"
        metainfo(r).project=project;
        metainfo(r).branch='mar4';
        metainfo(r).parameter=jobstruct(m).fittype;
        metainfo(r).image = strcat('INPUT/',jobstruct(m).fittype,'/',num2str(jobstruct(m).subject),'_',jobstruct(m).fittype,'_day2change.nii');
       
    else
        metainfo(r).image = strcat('INPUT/',jobstruct(m).fittype,'/',num2str(jobstruct(m).subject),'_',jobstruct(m).fittype,'_day1change.nii');
    end
    r = r+1; % move to next row
end
writetable(struct2table(metainfo),'metainfo.csv'); % write structure to csv


%% Send job to cluster
clusterconfig('queue','short.q')
jobid=job2cluster(@normalise_data,jobstruct);

%% SPM batch function
function out=normalise_data(jobstruct)

out=[];

studies = stormdb_get_studies(jobstruct(1).dbhandle,jobstruct(1).project,jobstruct(1).subject); % retrieve study times
study1 = datestr(datenum(studies{1}), 'yyyymmdd_HHMMSS'); % reformat study times to fit StormDB folder convention
study2 = datestr(datenum(studies{2}), 'yyyymmdd_HHMMSS');
study3 = datestr(datenum(studies{3}), 'yyyymmdd_HHMMSS');
outdir = strcat('/projects/MINDLAB2016_TMS-NovelWordKurtosis/scratch/MVPA/INPUT/',jobstruct(1).fittype,'/'); % save final output here

% construct input file paths for the images to be normalised
extrapars.isnifti=1; % toggles between info or data (if =1)
extrapars.ispermanent=0; %toggles between scratch and misc (misc=1 default)

sess1refpath = stormdb_std_output_path(jobstruct(1).project,jobstruct(1).branch,jobstruct(1).subject,study1,jobstruct(1).modality,jobstruct(1).fittype,extrapars); % session directories 
sess2refpath = stormdb_std_output_path(jobstruct(1).project,jobstruct(1).branch,jobstruct(1).subject,study2,jobstruct(1).modality,jobstruct(1).fittype,extrapars);
sess3refpath = stormdb_std_output_path(jobstruct(1).project,jobstruct(1).branch,jobstruct(1).subject,study3,jobstruct(1).modality,jobstruct(1).fittype,extrapars);

% Extract grey and white matter from raw DKI images
% session 1
sess1refimage = strcat('/projects/',jobstruct(1).project,'/scratch/datamar4/',sprintf('%04d',jobstruct(1).subject),'/',study1,'/',jobstruct(1).modality,'/',jobstruct(1).fitref,'/NATSPACE/0001.nii'); % Sess1 align reference image
sess1GMprobmask = strcat('/projects/',jobstruct(1).project,'/scratch/masksmar4/',sprintf('%04d',jobstruct(1).subject),'/',study1,'/',jobstruct(1).modality,'/KURTOSIS_FSL_ADC/NATSPACE/segment1.nii'); % GM probability mask
sess1WMprobmask = strcat('/projects/',jobstruct(1).project,'/scratch/masksmar4/',sprintf('%04d',jobstruct(1).subject),'/',study1,'/',jobstruct(1).modality,'/KURTOSIS_FSL_ADC/NATSPACE/segment2.nii'); % WM probability mask
sess1rawimage = strcat(sess1refpath,'/0001.nii'); % file to compute the mni warps from
sess1alignimage = strcat(sess1refpath,'/gmwm0001.nii'); % gm and wm masked image
spm_imcalc({sess1rawimage;sess1GMprobmask;sess1WMprobmask},sess1alignimage,'i1.*(i2+i3)'); % spm image calculator

% session 2
sess2refimage = strcat('/projects/',jobstruct(1).project,'/scratch/datamar4/',sprintf('%04d',jobstruct(1).subject),'/',study2,'/',jobstruct(1).modality,'/',jobstruct(1).fitref,'/NATSPACE/0001.nii'); % Sess2 align reference image
sess2GMprobmask = strcat('/projects/',jobstruct(1).project,'/scratch/masksmar4/',sprintf('%04d',jobstruct(1).subject),'/',study2,'/',jobstruct(1).modality,'/KURTOSIS_FSL_ADC/NATSPACE/segment1.nii'); % GM probability mask
sess2WMprobmask = strcat('/projects/',jobstruct(1).project,'/scratch/masksmar4/',sprintf('%04d',jobstruct(1).subject),'/',study2,'/',jobstruct(1).modality,'/KURTOSIS_FSL_ADC/NATSPACE/segment2.nii'); % WM probability mask
sess2rawimage = strcat(sess2refpath,'/0001.nii'); % file to compute the mni warps from
sess2alignimage = strcat(sess2refpath,'/gmwm0001.nii'); % gm and wm masked image
spm_imcalc({sess2rawimage;sess2GMprobmask;sess2WMprobmask},sess2alignimage,'i1.*(i2+i3)'); % spm image calculator

% session 3
sess3refimage = strcat('/projects/',jobstruct(1).project,'/scratch/datamar4/',sprintf('%04d',jobstruct(1).subject),'/',study3,'/',jobstruct(1).modality,'/',jobstruct(1).fitref,'/NATSPACE/0001.nii'); % Sess3 align reference image
sess3GMprobmask = strcat('/projects/',jobstruct(1).project,'/scratch/masksmar4/',sprintf('%04d',jobstruct(1).subject),'/',study3,'/',jobstruct(1).modality,'/KURTOSIS_FSL_ADC/NATSPACE/segment1.nii'); % GM probability mask
sess3WMprobmask = strcat('/projects/',jobstruct(1).project,'/scratch/masksmar4/',sprintf('%04d',jobstruct(1).subject),'/',study3,'/',jobstruct(1).modality,'/KURTOSIS_FSL_ADC/NATSPACE/segment2.nii'); % WM probability mask
sess3rawimage = strcat(sess3refpath,'/0001.nii'); % file to compute the mni warps from
sess3alignimage = strcat(sess3refpath,'/gmwm0001.nii'); % gm and wm masked image
spm_imcalc({sess3rawimage;sess3GMprobmask;sess3WMprobmask},sess3alignimage,'i1.*(i2+i3)'); % spm image calculator

% directories for images to-be-normalised
sess1path = stormdb_std_output_path(jobstruct(1).project,jobstruct(1).branch,jobstruct(1).subject,study1,jobstruct(1).modality,jobstruct(1).fittype,extrapars); 
sess2path = stormdb_std_output_path(jobstruct(1).project,jobstruct(1).branch,jobstruct(1).subject,study2,jobstruct(1).modality,jobstruct(1).fittype,extrapars);
sess3path = stormdb_std_output_path(jobstruct(1).project,jobstruct(1).branch,jobstruct(1).subject,study3,jobstruct(1).modality,jobstruct(1).fittype,extrapars);

sess1mniimage = strcat(sess1path,'/mni_gmwm0001.nii'); % output files in MNI space
sess2mniimage = strcat(sess2path,'/mni_gmwm0001.nii');
sess3mniimage = strcat(sess3path,'/mni_gmwm0001.nii');

outdir = strcat('/projects/MINDLAB2016_TMS-NovelWordKurtosis/scratch/MVPA/INPUT/',jobstruct(1).fittype,'/'); % output directory for day1 and day2 change images
day1changefile = strcat(num2str(jobstruct(1).subject),'_',jobstruct(1).fittype,'_day1change');
day2changefile = strcat(num2str(jobstruct(1).subject),'_',jobstruct(1).fittype,'_day2change');

jobfile = {'/projects/MINDLAB2016_TMS-NovelWordKurtosis/scratch/MVPA/KurtosisMNI_job.m'};
jobs = repmat(jobfile, 1, 1);
inputs = cell(14, 1);
inputs{1, 1} = {sess1refimage}; % Normalise: Estimate & Write: Align Session 1
inputs{2, 1} = {sess1alignimage}; % Normalise: Estimate & Write: Write Session 1
inputs{3, 1} = {sess2refimage}; % Normalise: Estimate & Write: Align Session 2
inputs{4, 1} = {sess2alignimage}; % Normalise: Estimate & Write: Write Session 2
inputs{5, 1} = {sess3refimage}; % Normalise: Estimate & Write: Align Session 3
inputs{6, 1} = {sess3alignimage}; % Normalise: Estimate & Write: Write Session 3
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

disp (['finished processing subject ' num2str(jobstruct(1).subjectnum)])

end
