import os
import torch
import pandas as pd
import glob

def csvFilesMaker(all_util, Accumulator_TOT, InputA_TOT, InputB_TOT, totalCycles, dim):
		#normalize the results by number of cycles
		all_util = (all_util/totalCycles)*100
		Accumulator_TOT = (Accumulator_TOT/totalCycles)*100
		InputA_TOT = (InputA_TOT/totalCycles)*100
		InputB_TOT = (InputB_TOT/totalCycles)*100

		filenamek_csv = "UtilityFor"+str(dim)+"X"+str(dim)+"Dim.csv"
		Destination_path = "/home/a.mosa/Ameer/llama3/OutputFiles/SP" #"/home/firasramadan/miniconda3/Ameer_Project_Transformers/PTQ4ViT_SA_SP/OutputFiles/SP/" 
		os.makedirs(Destination_path, exist_ok=True)
		Destination_path = Destination_path + "/"
		#os.makedirs('/home/a.mosa/Ameer/PTQ4ViT_Firas/OutputFiles/SP', exist_ok=True)
		all_util_df = pd.DataFrame(all_util[:,:,0].numpy().round(3))
		all_util_df.to_csv(Destination_path + filenamek_csv,header = False, index = False)
    
		for bits in range(1,33): # Making files for each bit separately
			if bits < 9:
				filename_csv_InA = "InputA-SP-Bit"+str(9-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 32 is the signbit
				InA_df = pd.DataFrame(InputA_TOT[:,:,bits-1].numpy().round(3))
				InA_df.to_csv(Destination_path + filename_csv_InA,header = False, index = False)
				filename_csv_InB = "InputB-SP-Bit"+str(9-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 32 is the signbit
				InB_df = pd.DataFrame(InputB_TOT[:,:,bits-1].numpy().round(3))
				InB_df.to_csv(Destination_path + filename_csv_InB,header = False, index = False)
				#filename_csv_InA_toggle = "InputA-TR-Bit"+str(9-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 32 is the signbit
				#InA_toggle_df = pd.DataFrame(ToggleCount_InputA_Bits[:,:,bits-1,0].numpy())
				#InA_toggle_df.to_csv(Destination_path + filename_csv_InA_toggle,header = False, index = False)
				#filename_csv_InB_toggle = "InputB-TR-Bit"+str(9-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 32 is the signbit
				#InB_toggle_df = pd.DataFrame(ToggleCount_InputB_Bits[:,:,bits-1,0].numpy())
				#InB_toggle_df.to_csv(Destination_path + filename_csv_InB_toggle,header = False, index = False)
			if bits < 17:	
				filename_csv_mult = "Multiplier-SP-Bit"+str(17-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 16 is the MSB
				mult_df = pd.DataFrame(all_util[:,:,bits].numpy().round(3))
				mult_df.to_csv(Destination_path + filename_csv_mult,header = False, index = False)
				#filename_csv_mult_toggle = "Multiplier-TR-Bit"+str(17-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 16 is the MSB
				#mult_toggle_df = pd.DataFrame(ToggleCount_MultiplierBits[:,:,bits-1,0].numpy())
				#mult_toggle_df.to_csv(Destination_path + filename_csv_mult_toggle,header = False, index = False)
			#filename_csv_accum_toggle = "Accumulator-TR-Bit"+str(33-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 32 is the signbit
			#accum_toggle_df = pd.DataFrame(ToggleCount_Accumulator_Bits[:,:,bits-1,0].numpy())
			#accum_toggle_df.to_csv(Destination_path + filename_csv_accum_toggle,header = False, index = False)
			filename_csv_accum = "Accumulator-SP-Bit"+str(33-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 32 is the signbit
			accum_df = pd.DataFrame(Accumulator_TOT[:,:,bits-1].numpy().round(3))
			accum_df.to_csv(Destination_path + filename_csv_accum,header = False, index = False)

def load_checkpoint(file_path):
		checkpoint = {}
		with open(file_path, 'r') as f:
			data = f.read()
			checkpoint = eval(data)
		return checkpoint

def load_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    tensor = torch.tensor(df.values)
    return tensor

def main():
	dim = 128 #dim of the SA

	all_util = torch.zeros(dim,dim,17)
	Accumulator_TOT = torch.zeros(dim,dim,32)
	InputA_TOT = torch.zeros(dim,dim,8)
	InputB_TOT = torch.zeros(dim,dim,8)
	totalCycles = torch.zeros(1)

	output_files_path = '/home/a.mosa/Ameer/llama3/OutputFiles/*'
	folders = glob.glob(output_files_path)

	for folder in folders:
		all_util_file = os.path.join(folder, 'all_util.pt')
		Accumulator_TOT_file = os.path.join(folder, 'Accumulator_TOT.pt')
		InputA_TOT_file = os.path.join(folder, 'InputA_TOT.pt')
		InputB_TOT_file = os.path.join(folder, 'InputB_TOT.pt')
		totalCycles_file = os.path.join(folder, 'cycles.pt')

    	# Load the tensors from the checkpoint
		all_util += torch.load(all_util_file)
		Accumulator_TOT += torch.load(Accumulator_TOT_file)
		InputA_TOT += torch.load(InputA_TOT_file)
		InputB_TOT += torch.load(InputB_TOT_file)
		totalCycles += torch.load(totalCycles_file)
	
	print(f'Number of folders: {len(folders)}')
	print(totalCycles)
	csvFilesMaker(all_util, Accumulator_TOT, InputA_TOT, InputB_TOT, totalCycles, dim)

if __name__ == "__main__":
    main()
	