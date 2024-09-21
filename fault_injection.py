import math

import numpy as np
import torch
total_tiles = 0
SA_DIM=128

def bitflip(value, bit):
    return int(value) ^ (1 << bit)
    #return value

def bitflips_distribution_per_tiles(output_tmp, bitflips_num, regions_working, SA_region_A_B_rows, SA_region_C_D_rows, SA_region_A_C_cols, SA_region_B_D_cols):
    total_rows_working = SA_region_A_B_rows
    total_cols_working = SA_region_A_C_cols
    if ("D" in regions_working):
        total_cols_working += SA_region_B_D_cols
        total_rows_working += SA_region_C_D_rows
    elif ("B" in regions_working):
        total_cols_working += SA_region_B_D_cols
    elif ("C" in regions_working):
        total_rows_working += SA_region_C_D_rows

    random_indices = np.random.randint(0, total_rows_working*total_cols_working, size=bitflips_num)
    for i in range(0, bitflips_num):
        bit_in_acc_to_flip = np.random.randint(0, 14)
        i_index = int(random_indices[i]/total_cols_working)
        j_index = int(random_indices[i]%total_cols_working)
        output_tmp[i_index][j_index] = bitflip(output_tmp[i_index][j_index], bit_in_acc_to_flip)
        #globals_.flipping_map_accumulator[int(random_indices[i]/total_cols_working)][int(random_indices[i]%total_cols_working)][bit_in_acc_to_flip] += 1
    return output_tmp

################################################# TILING AND MATRIX MULTIPLICATION PYTHON #######################################
############# MUST CHECK DIMENSIONS ################## MUST CHECK DIMENSTIONS ############## MUST CHECK DIMENSIONS ##############
def matmul_FI(x_unfolded_expanded, w_unfolded_expanded, bitflip_per_mul):
    global total_tiles
    SA_region_A_B_rows = range(0, 68)
    SA_region_C_D_rows = range(68, 128)
    SA_region_A_C_cols = range(0, 64)
    SA_region_B_D_cols = range(64, 128)
    x_num_of_tiles = math.ceil(x_unfolded_expanded.size(0)/SA_DIM)
    w_num_of_tiles = math.ceil(w_unfolded_expanded.size(1)/SA_DIM)
    number_of_tiles_multiplications = x_num_of_tiles * w_num_of_tiles
    bitflip_per_tile_mul = int(bitflip_per_mul/number_of_tiles_multiplications)
    output_mat_final = torch.zeros([x_num_of_tiles*SA_DIM, w_num_of_tiles*SA_DIM])
    regions_working= ["A","B","C","D"]
    total_tiles += number_of_tiles_multiplications
    #num_integers_np = 20
    #print("Multiple random integers (NumPy):", random_integers_np)
    for i in range(0, x_num_of_tiles):
        flag_x_padded = False
        flag_w_padded = False
        for j in range(0, w_num_of_tiles):
            x_current_tile = x_unfolded_expanded[(i)*SA_DIM:((i+1)*SA_DIM), :]
            
            if (j == w_num_of_tiles-1) and w_unfolded_expanded.size(1)%SA_DIM != 0:
                zero_padded_mat_w = torch.zeros([w_unfolded_expanded.size(0), SA_DIM])
                zero_padded_mat_w[:, 0:w_unfolded_expanded.size(1)-(j*SA_DIM)] = w_unfolded_expanded[:, SA_DIM*(j):]
                w_current_tile = zero_padded_mat_w
                flag_w_padded = True
            else:
                w_current_tile = w_unfolded_expanded[:, (j)*SA_DIM:((j+1)*SA_DIM)]
            #globals_.hist_x += hist_x_tiles(x_current_tile,hist_x)
            #globals_.hist_w += hist_w_tiles(w_current_tile,hist_w)
            #plt.hist(x_current_tile)
            #plt.savefig('foo.png', bbox_inches='tight')
            #plt.close()
            if (i == x_num_of_tiles-1) and x_unfolded_expanded.size(0)%SA_DIM != 0:
                zero_padded_mat_x = torch.zeros([SA_DIM, x_unfolded_expanded.size(1)])
                zero_padded_mat_x[0:x_unfolded_expanded.size(0)-(i*SA_DIM), :] = x_unfolded_expanded[SA_DIM*i:, :]
                x_current_tile = zero_padded_mat_x
                flag_x_padded = True

            output_tmp = torch.matmul(x_current_tile, w_current_tile)
            if (flag_x_padded==True and flag_w_padded==True):
                regions_working = ["A"]
            elif (flag_x_padded==True and flag_w_padded==False):
                regions_working = ["A","B"]
            elif (flag_x_padded==False and flag_w_padded==True):
                regions_working = ["A","C"]
            else:
                regions_working= ["A","B","C","D"]

            # flipping_map_accumulator = np.zeros(shape=(SA_DIM,SA_DIM,32))
            ########################### FAULT INJECTION PART ##############################
            output_tmp = bitflips_distribution_per_tiles(output_tmp, bitflip_per_tile_mul, regions_working, len(SA_region_A_B_rows), len(SA_region_C_D_rows), len(SA_region_A_C_cols), len(SA_region_B_D_cols))



            output_mat_final[(i)*SA_DIM:(i+1)*SA_DIM, (j)*SA_DIM:(j+1)*SA_DIM] = output_tmp

    output_mat_final = output_mat_final[0:x_unfolded_expanded.size(0), 0:w_unfolded_expanded.size(1)]
    return output_mat_final

if __name__ == "__main__":
    X = torch.randint(0, 2**6, (197, 64)).cpu()
    W = torch.randint(0, 2**6, (64, 197)).cpu()
    bitFlips_number=500000
    bitflip_per_layer = bitFlips_number/24
    bitflip_per_mul = bitflip_per_layer/384
    out = matmul_FI(X, W, bitflip_per_mul)
    exp_out = np.matmul(X.cpu().numpy(), W.cpu().numpy())
    print(torch.allclose(out, torch.from_numpy(exp_out).float()))
    for i in range(out.size(0)):
        for j in range(out.size(1)):
            if out[i][j] != exp_out[i][j]:# and abs(out[i][j] - exp_out[i][j]) != 2**math.floor(math.log2(abs(out[i][j] - exp_out[i][j]))):
                print("Values at position ({}, {}) are not equal or differentiated by a power of 2".format(i, j))
                print("Expected: ", exp_out[i][j])
                print("Actual: ", out[i][j])