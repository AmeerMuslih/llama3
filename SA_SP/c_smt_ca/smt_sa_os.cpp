#include <torch/extension.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include <cmath>
#include "grid_os.cpp"

using namespace std;
using namespace torch::indexing;

struct tile_idx {
    uint32_t d1;
    uint32_t d2;
    uint32_t d3;
};

template <typename T>
class smt_sa_os {
private:
    uint16_t _dim;
    uint8_t _threads;
    uint16_t _max_depth;
    torch::Tensor _a;
    torch::Tensor _b;

    void _subtile_dict(vector<uint16_t> &subtile_start, vector<uint16_t> &subtile_end);
    void _subtile_range(uint8_t thread, uint16_t &thread_tile_start, uint16_t &thread_tile_end);

public:
    grid<T> sa_grid;
    uint64_t cycles;

    smt_sa_os (uint16_t dim, uint8_t threads, uint16_t max_depth=4096);

    void set_inputs(torch::Tensor a, torch::Tensor b);
    void get_tile(vector<torch::Tensor> &tile_a, vector<torch::Tensor> &tile_b, tile_idx t_idx);
    std::vector<torch::Tensor> go();
    std::vector<torch::Tensor> go(vector<tile_idx> &tile_vec);
};

template <typename T>
smt_sa_os<T>::smt_sa_os (uint16_t dim, uint8_t threads, uint16_t max_depth) : _dim(dim), _threads(threads), _max_depth(max_depth), sa_grid(dim, threads, max_depth), cycles(0) {}

template <typename T>
void smt_sa_os<T>::set_inputs(torch::Tensor a, torch::Tensor b) {
    assert(a.dim() == 3);
    assert(b.dim() == 2);
	//cout << a.size(2) << "      " << b.size(0) << endl;
    if(a.size(2) != b.size(0)){
        cout << "ERROR: WRONG DIMENSIONS" << endl;
        exit(0);
        }
    _a = a;
    _b = b; 
}

template <typename T>
void smt_sa_os<T>::get_tile(vector<torch::Tensor> &tile_a, vector<torch::Tensor> &tile_b, tile_idx t_idx) {
    uint16_t a_tile_H = _dim;
    uint16_t b_tile_W = _dim;

    vector<uint16_t> subtile_start;
    vector<uint16_t> subtile_end;
    _subtile_dict(subtile_start, subtile_end);

    for (uint8_t t=0; t<_threads; t++) {
        torch::Tensor tile = _a.index({(int)t_idx.d1,
                                       Slice((int)(t_idx.d2 * a_tile_H), (int)((t_idx.d2+1)*a_tile_H)),
                                       Slice((int)subtile_start[t], (int)subtile_end[t])});
        //torch::Tensor tile = _a[t_idx.d1];
		//tile = tile.narrow(0, t_idx.d2 * a_tile_H, (t_idx.d2+1)*a_tile_H);
		//tile = tile.narrow(1, subtile_start[t], subtile_end[t]);
		//xt::view(_a, t_idx.d1, xt::range(t_idx.d2 * a_tile_H, (t_idx.d2+1)*a_tile_H), xt::range(subtile_start[t], subtile_end[t]));
        tile_a.push_back(tile);

        if (_dim > tile_a[t].size(0)) {
            //tile_a[t] = xt::pad(tile_a[t], {{0, _dim - tile_a[t].shape()[0]}, {0, 0}});
            tile_a[t] = torch::nn::functional::pad(tile_a[t],
                   torch::nn::functional::PadFuncOptions({0, 0, 0, _dim - tile_a[t].size(0)}));
        }
    }

    for (uint8_t t=0; t<_threads; t++) {
        torch::Tensor tile = _b.index({Slice((int)subtile_start[t], (int)subtile_end[t]),
                                       Slice((int)(t_idx.d3*b_tile_W), (int)((t_idx.d3+1)*b_tile_W))});
		//torch::Tensor tile = _b.narrow(0, subtile_start[t], subtile_end[t]);
		//_b = _b.narrow(1, t_idx.d3*b_tile_W, (t_idx.d3+1)*b_tile_W);
        //xt::xarray<T> tile = xt::view(_b, xt::range(subtile_start[t], subtile_end[t]), xt::range(t_idx.d3*b_tile_W, (t_idx.d3+1)*b_tile_W));
        tile_b.push_back(tile);

        if (_dim > tile_b[t].size(1)) {
            //tile_b[t] = xt::pad(tile_b[t], {{0, 0}, {0, _dim - tile_b[t].shape()[1]}});
            tile_b[t] = torch::nn::functional::pad(tile_b[t],
                    torch::nn::functional::PadFuncOptions({0, _dim - tile_b[t].size(1)}));
        }
    }
}

template <typename T>
void smt_sa_os<T>::_subtile_dict(vector<uint16_t> &subtile_start, vector<uint16_t> &subtile_end) {
    for (uint8_t t=0; t<_threads; t++) {
        uint16_t start, end;
        _subtile_range(t, start, end);
        subtile_start.push_back(start);
        subtile_end.push_back(end);
    }
}

template <typename T>
void smt_sa_os<T>::_subtile_range(uint8_t thread, uint16_t &thread_tile_start, uint16_t &thread_tile_end) {
    uint16_t a_tile_W = _a.size(2);
    uint16_t b_tile_H = _b.size(0);
    assert(a_tile_W == b_tile_H);

    uint16_t subtile_size = floor(float(b_tile_H) / _threads);

    if (thread < _threads - 1) {
        thread_tile_start = thread * subtile_size;
        thread_tile_end = (thread+1) * subtile_size;
    }
    else {
        thread_tile_start = thread * subtile_size;
        thread_tile_end = b_tile_H;
    }
}

template <typename T>
std::vector<torch::Tensor> smt_sa_os<T>::go(vector<tile_idx> &tile_vec) {
    assert(tile_vec.size() > 0);
    uint16_t a_tiles = ceil(float(_a.size(1)) / _dim);
	uint16_t b_tiles = ceil(float(_b.size(1)) / _dim);

    torch::Tensor PUs_access_count = torch::zeros({_dim, _dim, ((sizeof(int16_t) * CHAR_BIT)+1)},torch::kInt32);
    torch::Tensor Accumulator_bits_count = torch::zeros({_dim, _dim, ((sizeof(int32_t) * CHAR_BIT))},torch::kInt32);
    torch::Tensor InputA_Bits_count = torch::zeros({_dim, _dim, ((sizeof(int8_t) * CHAR_BIT))},torch::kInt32);
    torch::Tensor InputB_Bits_count = torch::zeros({_dim, _dim, ((sizeof(int8_t) * CHAR_BIT))},torch::kInt32);
    torch::Tensor MultiplierToggleCount = torch::zeros({_dim, _dim, ((sizeof(int16_t) * CHAR_BIT)),2},torch::kInt32);
    torch::Tensor AccumulatorToggleCount = torch::zeros({_dim, _dim, ((sizeof(int32_t) * CHAR_BIT)),2},torch::kInt32);
    torch::Tensor InputAToggleCount = torch::zeros({_dim, _dim, ((sizeof(int8_t) * CHAR_BIT)),2},torch::kInt32);
    torch::Tensor InputBToggleCount = torch::zeros({_dim, _dim, ((sizeof(int8_t) * CHAR_BIT)),2},torch::kInt32);
    //auto PUs_access_count_ = PUs_access_count.accessor<T,2>();

    // Assuming tile_vec is ordered (batch, height, width), i.e., rows->columns->depth!
    int32_t counter_max = (tile_vec[0].d1 * a_tiles * b_tiles) + (tile_vec[0].d2 * b_tiles) + (tile_vec[0].d3);
    torch::Tensor array_ctrl = torch::full({_dim, _dim}, counter_max, torch::dtype(torch::kInt32));
	auto array_ctrl_ = array_ctrl.accessor<int32_t, 2>();
    uint32_t global_tile_idx = 1;
    vector<torch::Tensor> tile_a, tile_b;
	get_tile(tile_a, tile_b, tile_vec[0]);
    for (uint8_t t=0; t<_threads; t++)
        sa_grid.push(tile_a[t], tile_b[t], t, true);
    torch::Tensor result = torch::zeros({_a.size(0), _a.size(1), _b.size(1)},torch::kInt32);


	auto result_ = result.accessor<int, 3>();  
    vector<uint16_t> subtile_start, subtile_end;
    _subtile_dict(subtile_start, subtile_end);

    float util_rate = 0;
    uint32_t computed = 0;
    uint32_t while_end = tile_vec.size() * _dim * _dim;
    // cout << "while_end: " << while_end << endl;
    //cout << "Tile Vec Size: " << tile_vec.size() << endl;
	//cout << while_end << endl;
    while (computed < while_end) {
        sa_grid.cycle(PUs_access_count,Accumulator_bits_count,InputA_Bits_count,InputB_Bits_count,MultiplierToggleCount,AccumulatorToggleCount,InputAToggleCount,InputBToggleCount);
        cycles++;
		//cout<< "cycles: " << cycles << endl;
        for (uint16_t i=0; i<_dim; i++) {
            for (uint16_t j=0; j<_dim; j++) {
                uint8_t halt_count = 0;
				
                for (uint8_t t=0; t<_threads; t++) {
                    if (sa_grid.nodes[i][j].is_halt(t))
                        halt_count++;
                    uint32_t acc_t = sa_grid.nodes[i][j].get_acc_t(t);
                    assert(subtile_end[t] - subtile_start[t] >= 0);

                    if ((acc_t == uint32_t(subtile_end[t] - subtile_start[t])) && !sa_grid.nodes[i][j].is_halt(t))
                        sa_grid.nodes[i][j].halt(t);
                }

                if (halt_count == _threads) {
                    uint32_t batch = floor(float(array_ctrl_[i][j]) / (a_tiles * b_tiles));
                    uint32_t i_result = int(i + int((array_ctrl_[i][j] % (a_tiles * b_tiles)) / b_tiles) * _dim);
                    uint32_t j_result = int(j + ((array_ctrl_[i][j] % (a_tiles * b_tiles)) % b_tiles) * _dim);

                    if (i_result < result.size(1) && j_result < result.size(2))
                        result_[batch][i_result][j_result] = sa_grid.nodes[i][j].get_acc();
                    
                    array_ctrl[i][j] = array_ctrl[i][j] + 1;
                    sa_grid.nodes[i][j].reset_acc();
                    sa_grid.nodes[i][j].reset_acc_t();
                    computed++;

                    sa_grid.nodes[i][j].release();
                }
            }
        }

        if (sa_grid.mem_a[0]._buf[0].size() < 128) {
            if (global_tile_idx < tile_vec.size()) {
                tile_a.clear();
                tile_b.clear();

                get_tile(tile_a, tile_b, tile_vec[global_tile_idx]);

                for (uint8_t t=0; t<_threads; t++)
                    sa_grid.push(tile_a[t], tile_b[t], t);
				//cout <<"Tile A: " << tile_a[0].sizes() <<" " << "Tile B: " << tile_b[0].sizes()<<endl;
                global_tile_idx++;
				//cout << "cycles: " << cycles << endl;
            }
        }

        util_rate += sa_grid.get_util_rate();
    }
    //return {result, torch::full({1}, util_rate), torch::full({1}, (float)cycles),PUs_access_count};
	//cout << "Mult   " << PUs_access_count.sizes()<<"   Accumulator_bits_count: " << Accumulator_bits_count.sizes() << endl;
	//cout << "input A   " << InputA_Bits_count.sizes() << "Inputb:   " <<   InputB_Bits_count.sizes() << endl;
   return {result.to(torch::kInt32), torch::full({1}, util_rate), torch::full({1}, (float)cycles),PUs_access_count,Accumulator_bits_count,InputA_Bits_count,InputB_Bits_count,MultiplierToggleCount,AccumulatorToggleCount,InputAToggleCount,InputBToggleCount};

}

template <typename T>
std::vector<torch::Tensor> smt_sa_os<T>::go() {
    uint16_t batch = _a.size(0);
    uint16_t a_tiles = ceil(float(_a.size(1)) / _dim);
    uint16_t b_tiles = ceil(float(_b.size(1)) / _dim);

    vector<tile_idx> tile_vec;
    for (uint16_t b=0; b<batch; b++) {
        for (uint16_t i=0; i<a_tiles; i++) {
            for (uint16_t j=0; j<b_tiles; j++) {
                tile_idx t_idx;
                t_idx.d1 = b;
                t_idx.d2 = i;
                t_idx.d3 = j;
                tile_vec.push_back(t_idx);
            }
        }
    }
    return go(tile_vec);
}

