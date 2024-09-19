import torch
import c_smt_sa
import sys
import csv
import pandas as pd
import numpy as ny
import pickle
import argparse
import math
type_bits=33
parser = argparse.ArgumentParser(description='Firas Ramadan, firasramadan@campus.technion.ac.il')
parser.add_argument('--dimension', type=int, required=True, help='choose the dimension of the systolic array')
#parser.add_argument('--GroupID', type=int, required=True, help='choose which group of the generated pickle files to load- all are in size of 10')

def csvFilesMaker(dim,GroupID,all_util,Accumulator_TOT,InputA_TOT,InputB_TOT,ToggleCount_MultiplierBits,ToggleCount_Accumulator_Bits,ToggleCount_InputA_Bits,ToggleCount_InputB_Bits):
		filenamek_csv = "UtilityFor"+str(dim)+"X"+str(dim)+"Dim.csv"
		Destination_path = "/home/a.mosa/AS_SP/OutputFiles/" #"/home/firasramadan/miniconda3/project_quantization_8bit/c_smt_sa/OutputFiles/Group-" + str(GroupID)+ "/"
		all_util_df = pd.DataFrame(all_util[:,:,0].numpy())
		all_util_df.to_csv(Destination_path + filenamek_csv,header = False, index = False)
    
		for bits in range(1,33): # Making files for each bit separately
			if bits < 9:
				filename_csv_InA = "InputA-SP-Bit"+str(9-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 32 is the signbit
				InA_df = pd.DataFrame(InputA_TOT[:,:,bits-1].numpy())
				InA_df.to_csv(Destination_path + filename_csv_InA,header = False, index = False)
				filename_csv_InB = "InputB-SP-Bit"+str(9-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 32 is the signbit
				InB_df = pd.DataFrame(InputB_TOT[:,:,bits-1].numpy())
				InB_df.to_csv(Destination_path + filename_csv_InB,header = False, index = False)
				filename_csv_InA_toggle = "InputA-TR-Bit"+str(9-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 32 is the signbit
				InA_toggle_df = pd.DataFrame(ToggleCount_InputA_Bits[:,:,bits-1,0].numpy())
				InA_toggle_df.to_csv(Destination_path + filename_csv_InA_toggle,header = False, index = False)
				filename_csv_InB_toggle = "InputB-TR-Bit"+str(9-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 32 is the signbit
				InB_toggle_df = pd.DataFrame(ToggleCount_InputB_Bits[:,:,bits-1,0].numpy())
				InB_toggle_df.to_csv(Destination_path + filename_csv_InB_toggle,header = False, index = False)
			if bits < 17:	
				filename_csv_mult = "Multiplier-SP-Bit"+str(17-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 16 is the MSB
				mult_df = pd.DataFrame(all_util[:,:,bits].numpy())
				mult_df.to_csv(Destination_path + filename_csv_mult,header = False, index = False)
				filename_csv_mult_toggle = "Multiplier-TR-Bit"+str(17-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 16 is the MSB
				mult_toggle_df = pd.DataFrame(ToggleCount_MultiplierBits[:,:,bits-1,0].numpy())
				mult_toggle_df.to_csv(Destination_path + filename_csv_mult_toggle,header = False, index = False)
			filename_csv_accum_toggle = "Accumulator-TR-Bit"+str(33-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 32 is the signbit
			accum_toggle_df = pd.DataFrame(ToggleCount_Accumulator_Bits[:,:,bits-1,0].numpy())
			accum_toggle_df.to_csv(Destination_path + filename_csv_accum_toggle,header = False, index = False)
			filename_csv_accum = "Accumulator-SP-Bit"+str(33-bits)+"in"+str(dim)+"X"+str(dim)+"Dim.csv" #Bit 32 is the signbit
			accum_df = pd.DataFrame(Accumulator_TOT[:,:,bits-1].numpy())
			accum_df.to_csv(Destination_path + filename_csv_accum,header = False, index = False)

def aZeroCounter(a,dim,column):
		counter =0
		for i in range(0,a.size(1)):
			if i%dim == column:
				for j in range(0,a.size(2)):
					if a[0,i,j] == 0:
						counter += 1
		
		print(counter)
		return counter
		
def bZeroCounter(b,dim,raw):
		counter =0
		for i in range(0,b.size(1)):
			if i%dim == raw:
					if b[j,i] == 0:
						counter += 1
		return counter

def main():
		original_stdout = sys.stdout
		args = parser.parse_args()
		dim = args.dimension
		#GroupID = args.GroupID
		all_util = torch.zeros(dim,dim,17)
		Accumulator_TOT = torch.zeros(dim,dim,32)
		InputA_TOT = torch.zeros(dim,dim,8)
		InputB_TOT = torch.zeros(dim,dim,8)
		ToggleCount_MultiplierBits = torch.zeros(dim,dim,16,2)
		ToggleCount_InputA_Bits = torch.zeros(dim,dim,8,2)
		ToggleCount_InputB_Bits = torch.zeros(dim,dim,8,2)
		ToggleCount_Accumulator_Bits = torch.zeros(dim,dim,32,2)
		totalCycles=totalCycles_tmp=total_zeros_first_a=total_zeros_first_b=total_tiles=0

		#for i in range(10*(GroupID-1),10*GroupID):	
			
		#filename_pickle = "MatricesForAllLayersOfPhoto-QUANTIZED8bit-"+str(i+1)+".pkl"
		#with open(r'/home/firasramadan/miniconda3/project_quantization_8bit/OUTPUT-Imagenet/PickleFiles/' + filename_pickle, 'rb') as file:
			#CurrentPhoto = pickle.load(file)
		#print(len(CurrentPhoto))
		#for m in range(0,len(CurrentPhoto)):
		#a= CurrentPhoto[m].InputT
		#b= CurrentPhoto[m].WeightT 
		dim1 = dim
		dim2 = dim//2
		a = torch.randint(high=50, size=(dim1, dim2))
		b = torch.randint(high=50, size=(dim2, dim1))
		print(a)
		print(b)
		#a=a[:,:,:].detach().cpu()
		#b=b[:,:].detach().cpu()
		ref = torch.matmul(a, b)
		dut, util, cycles, PUs_access_count,AccumulatorBitsCount,Input_A_BitsCount,Input_B_BitsCount,MultiplierToggle,AccumulatorToggle,InputAToggle,InputBToggle = c_smt_sa.exec(a[None,:,:].detach().cpu(),b[:,:].detach().cpu(), dim, 1, 1024)
		all_util += PUs_access_count
		Accumulator_TOT += AccumulatorBitsCount
		InputA_TOT += Input_A_BitsCount
		InputB_TOT += Input_B_BitsCount
		ToggleCount_MultiplierBits += MultiplierToggle
		ToggleCount_Accumulator_Bits += AccumulatorToggle
		ToggleCount_InputA_Bits += InputAToggle
		ToggleCount_InputB_Bits += InputBToggle
		#totalCycles += (2*dim+a.size(2)-2)*math.ceil(a.size(1)/dim)*math.ceil(b.size(1)/dim)
		totalCycles_tmp += (cycles-3)
		#print("Total Cycles: ")
		#print(totalCycles)
		print("Simulator Cycles: ")
		print(totalCycles_tmp)
		diff = (ref - dut).abs().max()
		print("diff={}, util={}".format(diff.item(), (util / cycles).item()))
		print(ref)
		print(dut)
		if diff > 1e-4:
			print("fuck")
			exit()
		#print(total_zeros_first_a+total_zeros_first_b)
		csvFilesMaker(dim,0,all_util,Accumulator_TOT,InputA_TOT,InputB_TOT,ToggleCount_MultiplierBits,ToggleCount_Accumulator_Bits,ToggleCount_InputA_Bits,ToggleCount_InputB_Bits)

		return
	
if __name__ == '__main__':
	  main()	
		

		
