# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:16:48 2021

@author: maxen
"""


import os
import warnings
import subprocess
from pathlib import Path
from shutil import rmtree
import json
import shutil

from pydicom import dcmread

def rename(series, filenames, path):
    if 'MPRAGE' in series.upper() or '3DT1' in series.upper():
        if 'CS' in series.upper() or 'ISO1' in series.upper():
            return()
        return ["MPRAGE"]
    if 'FLAIR' in series.upper(): # ORIG ???
        return ["FLAIR"]
    if 'phase' in series.upper() or ("SWI_EPI" in series.upper() and "_ph" in series.upper()):
        return ["acq-phase_T2star"]
    if '3D EPI' in series.upper() or "SWI_EPI" in series:
        return ["acq-mag_T2star"]
    if 'Opt_DTI' in series.upper() or 'DWI' in series.upper():
        return ["DWI"]
    if "T1map" in series.upper():
        return ["T1map"]
    if "DIR" in series.upper():
        return ['DIR']
    if "T2opt" in series.upper():
        return ['T2']
    if "T1W" in series.upper() and "gd" in series.upper():
        return ['T1w_Gd']
    if "MP2RAGE" in series.upper():
        if len(filenames)>1:
            new_filenames = []
            for filename in filenames:
                try:
                    with open(f'{path}/{filename}.json') as json_file:
                        df = json.load(json_file)
                        new_filenames.append(f"inv-{df['EchoNumber']}_part-mag_MP2RAGE")
                except:
                    pass
            return new_filenames
        else:
            return ['UNIT1']
    return []





def convert_all_dicoms(directory, dicom2niix_path="dcm2niix", convert=True):
    """
    Converts all dicom files of a particular patient into multiple compressed
    nifti (.nii.gz) files.

    Parameters
    ----------
    directory : <str>
        Path to patient's DICOM directory.
    dicom2niix_path : <str>
        ONLY FOR WINDOWS USERS. Path to dcm2niix.exe file.

    Returns
    -------
    all_sequences : <list> of <tuple>
        List of tuples (Path to specific dicom series directory, Series description).

    """
    directory = os.path.join(directory)
    print("[INFO] Starting to convert ...")
    # if os.path.isfile(f"{directory}/DICOMDIR"):


        # ds = dcmread(f"{directory}/DICOMDIR")


        # warnings.filterwarnings("ignore")

        # [patient] = ds.patient_records
        # [study] = [ ii for ii in patient.children if ii.DirectoryRecordType == "STUDY" ]
        # all_series = [ii for ii in study.children if ii.DirectoryRecordType == "SERIES"]

        # all_sequences=[]

        # for series in all_series:
        #     # Find all the IMAGE records in the series
        #     images = [
        #         ii for ii in series.children
        #         if ii.DirectoryRecordType == "IMAGE"
        #     ]

        #     descr = getattr(
        #         series, "SeriesDescription", "(no value available)"
        #     )

        #     elems = [ii["ReferencedFileID"] for ii in images]

        #     paths = [[ee.value] if ee.VM == 1 else ee.value for ee in elems]
        #     paths = [Path(*p) for p in paths]

        #     p = paths[0]

        #     path = f"{directory}/{p}"
        #     path = path.replace("\\",'/')
        #     if convert:
        #         subprocess.call([dicom2niix_path, '-f', "\"%f_%p_%t_%s\"", "-p",
        #                          "y", "-z", "y", path])

        #     path = path.replace(path.split("/")[-1],'')[:-1]

        #     all_sequences.append((path, descr))

    # else:
    if True:
        all_sequences = []
        for subdir, dirs, files in os.walk(directory):
            if len(dirs) !=0 or len(files)< 10:
                continue
            print(f"SUBDIR: {subdir}\tDIRS: {dirs}")#\nFILES: {files}\n")
            path = os.path.normpath(subdir)
            if convert:
                # print("\n\n>>> CALLED <<<")
                # print(' '.join(["dicom2niix", '-f', "\"%f_%p_%t_%s\"", "-p",
                #                  "y", "-z", "y", path]))
                subprocess.call([dicom2niix_path, '-f', "\"%f_%p_%t_%s\"", "-p",
                                  "y", "-z", "y", path])
            descr = dcmread(f"{path}/{files[0]}").SeriesDescription
            all_sequences.append((path, descr))
        all_sequences = [(x[0].replace('\\', '/'),x[1]) for x in all_sequences]

    print("[INFO] Converted all dicom files to compressed nifti")
    return all_sequences


def make_directories(bids_dir, pat_id=None, session=None):

    define_pat_id = pat_id is None

    # Assign a database ID to the patient
    if define_pat_id:
        all_directories = [x for x in next(os.walk(bids_dir))[1]]
        all_subj_dir = []
        for d in all_directories:
            if d.find('sub-') == 0:
                all_subj_dir.append(d)

        if all_subj_dir == []:
            pat_id = "001"
        else:
            subjects = [int(x.split('-')[1]) for x in all_subj_dir]
            pat_id = str((max(subjects) + 1)).zfill(3)

    # TODO: Check session, add dynamically or has to be specified

    subj_dir = os.path.join(bids_dir,f"sub-{pat_id}")
    already_in_db = os.path.exists(subj_dir)

    if not already_in_db:
        # Make directories
        os.mkdir(subj_dir)

    if session is None:

        all_directories = [x for x in next(os.walk(subj_dir))[1]]
        all_ses_dir = []
        for d in all_directories:
            if d.find('ses-') == 0:
                all_ses_dir.append(d)
        if define_pat_id:
            session = "01"
        else:
            sessions = [int(x.split('-')[1]) for x in all_ses_dir]
            if len(sessions) == 0:
                session = '01'
            else:
                session = str(max(sessions) + 1).zfill(2)


    d = os.path.join(subj_dir,f'ses-{session}')
    if not os.path.exists(d):
        os.mkdir(d)

    if not os.path.exists(os.path.join(d, 'anat')):
        os.mkdir(os.path.join(d, 'anat'))
    if not os.path.exists(os.path.join(d, 'dwi')):
        os.mkdir(os.path.join(d, 'dwi'))

    deriv = os.path.join(bids_dir, 'derivatives')

    def add_derivatives_dirs(derivative):
        try:
            os.mkdir(os.path.join(deriv, derivative,f'sub-{pat_id}'))
        except FileExistsError:
            pass
        try:
            os.mkdir(os.path.join(deriv, derivative,f'sub-{pat_id}', f'ses-{session}'))
        except FileExistsError:
            pass
    add_derivatives_dirs('transformations')
    add_derivatives_dirs('samseg')
    add_derivatives_dirs('stats')
    add_derivatives_dirs('segmentations')

    return pat_id, session


def delete_subject(bids_dir, pat_id):
    try:
        rmtree(os.path.join(bids_dir, f'sub-{pat_id}'))
    except FileNotFoundError:
        print("[Exception caught] Cannot remove directory that does not exists")
    try:
        rmtree(os.path.join(bids_dir, 'derivatives', 'transformations', f'sub-{pat_id}'))
    except FileNotFoundError:
        print("[Exception caught] Cannot remove directory that does not exists")
    try:
        rmtree(os.path.join(bids_dir, 'derivatives', 'samseg', f'sub-{pat_id}'))
    except FileNotFoundError:
        print("[Exception caught] Cannot remove directory that does not exists")
    try:
        rmtree(os.path.join(bids_dir, 'derivatives', 'stats', f'sub-{pat_id}'))
    except FileNotFoundError:
        print("[Exception caught] Cannot remove directory that does not exists")
    try:
        rmtree(os.path.join(bids_dir, 'derivatives', 'segmentations', f'sub-{pat_id}'))
    except FileNotFoundError:
        print("[Exception caught] Cannot remove directory that does not exists")

def delete_session(bids_dir, pat_id, session):
    try:
        rmtree(os.path.join(bids_dir, 'derivatives', 'transformations', f'sub-{pat_id}', f'ses-{session}'))
    except FileNotFoundError:
        print("[Exception caught] Cannot remove directory that does not exists")
    try:
        rmtree(os.path.join(bids_dir, 'derivatives', 'samseg', f'sub-{pat_id}', f'ses-{session}'))
    except FileNotFoundError:
        print("[Exception caught] Cannot remove directory that does not exists")
    try:
        rmtree(os.path.join(bids_dir, 'derivatives', 'stats', f'sub-{pat_id}', f'ses-{session}'))
    except FileNotFoundError:
        print("[Exception caught] Cannot remove directory that does not exists")
    try:
        rmtree(os.path.join(bids_dir, 'derivatives', 'segmentations', f'sub-{pat_id}', f'ses-{session}'))
    except FileNotFoundError:
        print("[Exception caught] Cannot remove directory that does not exists")


def rename_and_move_nifti(dicom_series, bids_dir, pat_id, session='01'):
    IGNORED_SERIES = ['3Plane_Loc_SSFSE', 'Ax T2 Propeller', 'AX REFORMAT',
                      'Opt_DTI_corr', "COR REFORMAT"]

    if len(dicom_series) == 1:
        path, series = dicom_series[0]
        moved = []
        for file in os.listdir(path):
            if not file.endswith(".nii.gz") or file.replace(".nii.gz", "") in moved:
                continue

            if file in IGNORED_SERIES or 'Survey' in file:
                os.remove(os.path.join(path, file))
                os.remove(os.path.join(path, file.replace('.nii.gz', '.json')))
                moved.append(file.replace(".nii.gz", ""))
                continue

            new_names = rename(file.replace(".nii.gz", ""), [file.replace(".nii.gz", "")], path)
            print(new_names)
            if new_names is None:
                print(f"DICOM series not recognized: {file.replace('.nii.gz','')}\nPath: {path}")
                new_names = [file.replace(".nii.gz", "")]

            for filename, new_name in zip([file.replace(".nii.gz", "")], new_names):
                dos = 'dwi' if new_name == 'DWI' else 'anat'
                if new_name != 'DWI':

                    shutil.move(f"{path}/{filename}.nii.gz",
                              f"{bids_dir}/sub-{pat_id}/ses-{session}/{dos}/sub-{pat_id}_ses-{session}_{new_name}.nii.gz")
                    shutil.move(f"{path}/{filename}.json",
                              f"{bids_dir}/sub-{pat_id}/ses-{session}/{dos}/sub-{pat_id}_ses-{session}_{new_name}.json")
                else:

                    shutil.move(f"{path}/{filename}.nii.gz",
                              f"{bids_dir}/sub-{pat_id}/ses-{session}/{dos}/sub-{pat_id}_ses-{session}_{new_name}.nii.gz")
                    shutil.move(f"{path}/{filename}.json",
                              f"{bids_dir}/sub-{pat_id}/ses-{session}/{dos}/sub-{pat_id}_ses-{session}_{new_name}.json")
                    shutil.move(f"{path}/{filename}.bval",
                              f"{bids_dir}/sub-{pat_id}/ses-{session}/{dos}/sub-{pat_id}_ses-{session}_{new_name}.bval")
                    shutil.move(f"{path}/{filename}.bvec",
                              f"{bids_dir}/sub-{pat_id}/ses-{session}/{dos}/sub-{pat_id}_ses-{session}_{new_name}.bvec")

            moved.append(file.replace(".nii.gz", ""))





    for path, series in dicom_series:
        if series in IGNORED_SERIES or 'ORIG' in series or 'PSIR' in series or 'Survey' in series:
            for file in os.listdir(path):
                if file.endswith(".nii.gz") or file.endswith(".json"):
                    os.remove(os.path.join(path, file))
            continue
        nifti_filenames = []
        for obj in list(os.walk(path))[0][2]:
            if obj.find('.nii.gz') <= 0:
                continue
            nifti_filenames.append( obj.replace('.nii.gz', '') )


        new_names = rename(series, nifti_filenames, path)
        print(new_names)
        if new_names is None:
            print(f"DICOM series not recognized: {series}\nPath: {path}")
            new_names = nifti_filenames

        for filename, new_name in zip(nifti_filenames, new_names):
            dos = 'dwi' if new_name == 'DWI' else 'anat'
            if new_name != 'DWI':

                shutil.move(f"{path}/{filename}.nii.gz",
                          f"{bids_dir}/sub-{pat_id}/ses-{session}/{dos}/sub-{pat_id}_ses-{session}_{new_name}.nii.gz")
                shutil.move(f"{path}/{filename}.json",
                          f"{bids_dir}/sub-{pat_id}/ses-{session}/{dos}/sub-{pat_id}_ses-{session}_{new_name}.json")
            else:

                shutil.move(f"{path}/{filename}.nii.gz",
                          f"{bids_dir}/sub-{pat_id}/ses-{session}/{dos}/sub-{pat_id}_ses-{session}_{new_name}.nii.gz")
                shutil.move(f"{path}/{filename}.json",
                          f"{bids_dir}/sub-{pat_id}/ses-{session}/{dos}/sub-{pat_id}_ses-{session}_{new_name}.json")
                shutil.move(f"{path}/{filename}.bval",
                          f"{bids_dir}/sub-{pat_id}/ses-{session}/{dos}/sub-{pat_id}_ses-{session}_{new_name}.bval")
                shutil.move(f"{path}/{filename}.bvec",
                          f"{bids_dir}/sub-{pat_id}/ses-{session}/{dos}/sub-{pat_id}_ses-{session}_{new_name}.bvec")

        print(f"SERIES: {series}\n   Filenames: {nifti_filenames}\n   RENAME: {rename(series, nifti_filenames, path)}\n\n")

    return

def delete_nii_json_in_dicomdir(dicom_series):
    for path, series in dicom_series:
        for file in os.listdir(path):
            if file.endswith(".nii.gz") or file.endswith(".json"):
                os.remove(os.path.join(path, file))


def rename_subject(bids_dir, old_id, new_id):
    if os.path.exists(os.path.join(bids_dir, f'sub-{new_id}')):
        raise FileExistsError(f"Subject {new_id} already exists in the database. Delete the subject first or choose another subject id.")
    if not os.path.exists(os.path.join(bids_dir, f'sub-{old_id}')):
        raise FileNotFoundError(f"Subject {old_id} is not in the database.")

    subject_dir = os.path.join(bids_dir, f"sub-{old_id}")


    def rename(main_dir):
        for path, subdirs, files in os.walk(main_dir):
            for filename in files:
                 if filename.startswith(f'sub-{old_id}'):
                    shutil.move(os.path.join(path, filename), \
                              os.path.join(path, filename.replace(f"sub-{old_id}", f"sub-{new_id}")))

    rename(subject_dir)
    shutil.move(subject_dir, os.path.join(bids_dir, f"sub-{new_id}"))

    derivatives = os.path.join(bids_dir, "derivatives")
    all_directories = [x for x in next(os.walk(derivatives))[1]]

    for derivative in all_directories:
        der_dir = os.path.join(derivatives, derivative, f'sub-{old_id}')
        rename(der_dir)
        shutil.move(der_dir, os.path.join(derivatives, derivative, f"sub-{new_id}"))





if __name__ == '__main__':
    pass
    # directory = "D:/WSBIM/ROSE"
    dicom2niix_path = "C:/Users/maxen/OneDrive/Bureau/UCLouvain/Q14/WSBIM2243/project/dcm2niix.exe"
    bids_dir = "C:/Users/maxen/OneDrive/Bureau/UCLouvain/Q14/WSBIM2243/project/WSBIM2243/database"



    # dicom_series = convert_all_dicoms(directory, dicom2niix_path)
    # pat_id, session = make_directories(bids_dir)
    # rename_and_move_nifti(dicom_series, bids_dir, pat_id, session)


