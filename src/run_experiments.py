'''

This script automatically runs the nonrigid code and save the results with all desired datasets.
--input: .txt file containing a list of paths, where each path points to the base directory of a dataset.
--output: Output path where file results are going to be saved.

Example:
python run_experiments.py --input /home/user/datasets/run_all_experiments.txt --output /home/user/nonrigid/results

'''

import os
import subprocess
import glob
import argparse
import re

nrigid_bin_folder = os.path.dirname(os.path.realpath(__file__))  +'/build/' #Set the path to the compute_descriptor executable.
source_folder = '/home/guipotje/Sources/2020-ijcv-geobit-extended-code/geobit/' #Set the path to this source code

pyramid_nlevels = 2
kp_scales = [1.0] #Desired scales to test
isocurvesizes = [0.05] #Desired iso sizes to test 0.05 (synthetic), 0.0025-0.004 (real)
experiment_name = 'ijcv_results' #Desired experiment name root folder

def check_dir(f):
	if not os.path.exists(f):
		os.makedirs(f)

def get_dir_list(filename):
	with open(filename,'r') as f:
		dirs = [line.rstrip('\n').rstrip() for line in f if line.rstrip('\n').rstrip()]

	return dirs

CWD = '/homeLocal/guipotje/tmp' ; check_dir(CWD)

def parseArg():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", help="Input file containing a list of experiments folders"
	, required=True, default = 'lists.txt') 
	parser.add_argument("-o", "--output", help="Output path where file results are going to be saved."
	, required=True, default = '')
	parser.add_argument("-f", "--file", help="is a file list?"
	, action = 'store_true')  
	parser.add_argument("-d", "--dir", help="is a dir with several dataset folders?"
	, action = 'store_true') 	
	args = parser.parse_args()
	return args

def standarize_csv_names(csv_list):
	for csv in csv_list:
		csv_path = os.path.dirname(csv)
		csv_name = os.path.basename(csv)
		csv_rename = re.findall('cloud_[0-9a-zA-Z]+',csv_name)[0] + '.pcd.csv'
		if csv_name != csv_rename:
			command = 'mv ' + csv_path + '/' + csv_name + ' ' + csv_path + '/' + csv_rename
			proc = subprocess.Popen(['/bin/bash','-c',command])	
			proc.wait()
	

def main():

	args = parseArg()

	if args.file:
		exp_list = get_dir_list(args.input)
	elif args.dir:
		exp_list = [d for d in glob.glob(args.input+'/*') if os.path.isdir(d)]
	else:
		exp_list = [args.input]

	for exp_dir in exp_list:
		datasettype_flag = ''

		if 'synthetic' in exp_dir.lower() or 'simulation' in exp_dir.lower():
			datasettype_flag = 'synthetic-'
			isocurvesizes = [0.06] # 0.05
		else:
			datasettype_flag = 'realdata-'
			isocurvesizes = [0.0020]

		#if 'smoothed' in exp_dir:
		datasettype_flag += 'smoothed'
		#else:
		#	datasettype_flag += 'standard'

		#dataset_name = os.path.basename(os.path.dirname(exp_dir))
		dataset_name = os.path.basename(exp_dir) #dataset_name+= os.path.basename(exp_dir)
		#dataset_name = os.path.abspath(exp_dir).split('/')[-2] + '_' +  os.path.abspath(exp_dir).split('/')[-1]
		#print dataset_name ; raw_input()
		
		experiment_files_unfiltered = glob.glob(exp_dir + "/*-rgb.png")
		experiment_files = [os.path.basename(e).split('-rgb.png')[0] for e in experiment_files_unfiltered]

		#print experiment_files ; quit()
		#standarize_csv_names(glob.glob(exp_dir + "/*.csv"))
		
		master_f = ''
		target_files = []
		for exp_file in experiment_files:
			if 'master' in exp_file or 'ref' in exp_file:
				master_f = os.path.basename(exp_file)
			else:
				target_files.append(os.path.basename(exp_file))


		for kp_scale in kp_scales:
			for isocurvesize in isocurvesizes:
				
				#building command
				command = nrigid_bin_folder + './nonrigid_descriptor -inputdir '
				command+= exp_dir
				command+= ' -refcloud ' + master_f

				for target_file in target_files:
					command+= ' -clouds ' + target_file

				command+= ' -kpscale ' + str(kp_scale)
				#command+= ' -sourcedir ' + source_folder
				command+= ' -isocurvesize ' + str(isocurvesize*kp_scale)
				command+= ' -detector FAST -distthreshold 512 -desc ORB -desc DAISY -desc FREAK'
				command+= ' -datasettype ' + datasettype_flag
				command+= ' -pyramidlevels ' + str(pyramid_nlevels)

				master_f_name, _ = os.path.splitext(os.path.basename(master_f))
				#print command +'\n\n'
				proc = subprocess.Popen(['/bin/bash','-c','rm ' + '*.png' ], cwd = CWD) #clean old data
				proc.wait()
				
				supercommand = '{ time %s ; } 2> %s_time.txt'%(command,master_f_name)
				print(supercommand) ; input()
				proc = subprocess.Popen(['/bin/bash','-c',supercommand], cwd = CWD)	
				proc.wait()

				result_dir = os.path.join(args.output,experiment_name) + '/' + dataset_name #+ '/' + datasettype_flag
				check_dir(result_dir)


				if len(master_f_name) > 5:
					command = 'mv -f ' + '*' + master_f_name + '* ' + result_dir ; #print command + '\n\n'
					proc = subprocess.Popen(['/bin/bash','-c',command], cwd = CWD)	
					proc.wait()

				command = 'mv -f ' + '*' + 'heatflow' + '* ' + result_dir ; #print command + '\n\n'
				proc = subprocess.Popen(['/bin/bash','-c',command], cwd = CWD)	
				proc.wait()

				command = 'mv -f ' + '*' + '.kp ' + result_dir ; #print command + '\n\n'
				proc = subprocess.Popen(['/bin/bash','-c',command], cwd = CWD)	
				proc.wait()

				command = 'mv -f ' + '*' + '.ply ' + result_dir ; #print command + '\n\n'
				proc = subprocess.Popen(['/bin/bash','-c',command], cwd = CWD)	
				proc.wait()

main()
