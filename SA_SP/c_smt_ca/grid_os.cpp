#include <torch/extension.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include "node_pu_os.cpp"
#include "node_mem.cpp"
#include <fstream>
#include <cstdint>
using namespace std;

template <typename T>
class grid {
private:
    uint16_t _dim;
    uint8_t _threads;
    uint16_t _max_depth;
public:
    vector<node_mem<T>> mem_a;
    vector<node_mem<T>> mem_b;
    vector<vector<node_pu<T>>> nodes;

    grid (uint16_t dim, uint8_t threads, uint16_t max_depth=4096);
    void push(torch::Tensor a, torch::Tensor b, uint8_t thread, bool pad=false);
    void cycle(torch::Tensor PUs_access_count, torch::Tensor Accumulator_bits_count, torch::Tensor InputA_Bits_count, torch::Tensor InputB_Bits_count, torch::Tensor MultiplierToggleCount, torch::Tensor AccumulatorToggleCount, torch::Tensor InputAToggleCount, torch::Tensor InputBToggleCount);
    float get_util_rate();
};

template <typename T>
grid<T>::grid (uint16_t dim, uint8_t threads, uint16_t max_depth) : _dim(dim), _threads(threads), _max_depth(max_depth) {
    mem_a.reserve(dim);
    mem_b.reserve(dim);
    nodes.reserve(dim);

    for (uint8_t i=0; i<dim; i++) {
        mem_a.push_back(node_mem<T>("mem_a_" + to_string(i), threads));
        mem_b.push_back(node_mem<T>("mem_b_" + to_string(i), threads));
        
        nodes.push_back(vector<node_pu<T>>());
        nodes[i].reserve(dim);

        for (uint8_t j=0; j<dim; j++) {
            nodes[i].push_back(node_pu<T>("node_" + to_string(i) + "_" + to_string(j), threads, max_depth));
        }
    }

    // Connect PUs
    for (uint16_t i=0; i<dim; i++) {
        for (uint16_t j=0; j<dim; j++) {
            if (j+1 < dim)
                nodes[i][j].out_a = &nodes[i][j+1];
            if (i+1 < dim)
                nodes[i][j].out_b = &nodes[i+1][j];
        }
    }

    // Connect memory units
    for (uint16_t i=0; i<dim; i++) {
        mem_a[i].out = &nodes[i][0];
        mem_a[i].out_buf_idx = 0;

        mem_b[i].out = &nodes[0][i];
        mem_b[i].out_buf_idx = 1;
    }
}

template <typename T>
void grid<T>::push(torch::Tensor a, torch::Tensor b, uint8_t thread, bool pad) {
    auto a_ = a.accessor<T, 2>();
    auto b_ = b.accessor<T, 2>();
    uint32_t a_H = a_.size(0);
    uint32_t a_W = a_.size(1);
    uint32_t b_H = b_.size(0);
    uint32_t b_W = b_.size(1);
    assert(a_W == b_H);
    for (uint32_t i=0; i<a_H; i++) {
        if (pad) {
            for (uint32_t k=0; k<i; k++) {
                mem_entry<T> me;
                me.valid = false;
                me.data = 0;
                mem_a[i]._buf[thread]._phy_fifo.push(me);
                mem_a[i]._buf[thread]._arch_fifo.push(me);
            }
        }

        for (uint32_t j=0; j<a_W; j++) {
            mem_entry<T> me;
            me.valid = true;
            me.data = a_[i][j];
            mem_a[i]._buf[thread]._phy_fifo.push(me);
            mem_a[i]._buf[thread]._arch_fifo.push(me);
        }
    }

    for (uint32_t j=0; j<b_W; j++) {
        if (pad) {
            for (uint32_t k=0; k<j; k++) {
                mem_entry<T> me;
                me.valid = false;
                me.data = 0;
                mem_b[j]._buf[thread]._phy_fifo.push(me);
                mem_b[j]._buf[thread]._arch_fifo.push(me);
            }
        }

        for (uint32_t i=0; i<b_H; i++) {
            mem_entry<T> me;
            me.valid = true;
            me.data = b_[i][j];
            mem_b[j]._buf[thread]._phy_fifo.push(me);
            mem_b[j]._buf[thread]._arch_fifo.push(me);
        }
    }
}

template <typename T>
void grid<T>::cycle(torch::Tensor PUs_access_count, torch::Tensor Accumulator_bits_count, torch::Tensor InputA_Bits_count, torch::Tensor InputB_Bits_count, torch::Tensor MultiplierToggleCount, torch::Tensor AccumulatorToggleCount, torch::Tensor InputAToggleCount, torch::Tensor InputBToggleCount) {

   auto PUs_access_count_ = PUs_access_count.accessor<int, 3>();
   auto Accumulator_bits_count_ = Accumulator_bits_count.accessor<int,3>();
   auto InputA_Bits_count_ = InputA_Bits_count.accessor<int,3>();
   auto InputB_Bits_count_ = InputB_Bits_count.accessor<int,3>();
   auto MultiplierToggleCount_ = MultiplierToggleCount.accessor<int,4>();
   auto AccumulatorToggleCount_ = AccumulatorToggleCount.accessor<int,4>();
   auto InputAToggleCount_ = InputAToggleCount.accessor<int,4>();
   auto InputBToggleCount_ = InputBToggleCount.accessor<int,4>();
   int stuck_bit = 0;
    for (uint16_t i=0; i<_dim; i++) {
        for (uint16_t j=0; j<_dim; j++){
			int stuck_bit = 0;
            T MultiplierBits[sizeof(int16_t) * CHAR_BIT] = {0}; // the 16/32bits of each MAC's multiplier output 
            T AccumulatorBits[sizeof(int) * CHAR_BIT] = {0}; // the 32bits of each MAC's Accumulator output
            T InputABits[sizeof(int8_t) * CHAR_BIT] = {0}; // the 32bits of each MAC's InputA output / Or 8 bit for quantization mode
            T InputBBits[sizeof(int8_t) * CHAR_BIT] = {0}; // the 32bits of each MAC's InputB output / Or 8 bit for quantization mode
            //cout << MultiplierBits.sizes() <<"  " << AccumulatorBits.sizes() <<"        " <<InputABits.sizes() << endl;
			PUs_access_count_[i][j][0] += nodes[i][j].go(MultiplierBits,AccumulatorBits,InputABits,InputBBits);
              // std::cout<<i<<"     "<<j<<endl;
			for(uint16_t k=1; k < ((sizeof(int) * CHAR_BIT)+1);k++){
            if(k<17){
			PUs_access_count_[i][j][k] += MultiplierBits[(sizeof(int16_t) * CHAR_BIT)-k]; // counting for everybit how many times it was 1 in the multiplier's
            MultiplierToggleCount_[i][j][k-1][0] += (MultiplierToggleCount_[i][j][k-1][1]!=MultiplierBits[(sizeof(int16_t) * CHAR_BIT)-k]);
            MultiplierToggleCount_[i][j][k-1][1] = MultiplierBits[(sizeof(int16_t) * CHAR_BIT)-k]; //updating the previous ones
            }
			Accumulator_bits_count_[i][j][k-1] += AccumulatorBits[(sizeof(int32_t) * CHAR_BIT)-k]; // counting for everybit how many times it was 1 in the accumulator's
            AccumulatorToggleCount_[i][j][k-1][0] += (AccumulatorToggleCount_[i][j][k-1][1]!=AccumulatorBits[(sizeof(int32_t) * CHAR_BIT)-k]);
            AccumulatorToggleCount_[i][j][k-1][1] = AccumulatorBits[(sizeof(int32_t) * CHAR_BIT)-k]; //updating the previous ones
			if(k<9){
            InputA_Bits_count_[i][j][k-1] += InputABits[(sizeof(int8_t) * CHAR_BIT)-k];
            InputAToggleCount_[i][j][k-1][0] += (InputAToggleCount_[i][j][k-1][1]!=InputABits[(sizeof(int8_t) * CHAR_BIT)-k]);
            InputAToggleCount_[i][j][k-1][1] = InputABits[(sizeof(int8_t) * CHAR_BIT)-k];
            InputB_Bits_count_[i][j][k-1] += InputBBits[(sizeof(int8_t) * CHAR_BIT)-k];
            InputBToggleCount_[i][j][k-1][0] += (InputBToggleCount_[i][j][k-1][1]!=InputBBits[(sizeof(int8_t) * CHAR_BIT)-k]);
            InputBToggleCount_[i][j][k-1][1] = InputBBits[(sizeof(int8_t) * CHAR_BIT)-k];
			}
            }
            for (uint8_t t=0; t<_threads; t++) {
                nodes[i][j]._buf_a[t].cycle();
                nodes[i][j]._buf_b[t].cycle();
            }
        }
    }

    for (uint16_t i=0; i<_dim; i++) {
        mem_a[i].go();
        mem_b[i].go();

        for (uint8_t t=0; t<_threads; t++) {
            mem_a[i]._buf[t].cycle();
            mem_b[i]._buf[t].cycle();
        }
    }
}

template <typename T>
float grid<T>::get_util_rate() {
    uint32_t util_sum = 0;

    for (uint16_t i=0; i<_dim; i++) {
        for (uint16_t j=0; j<_dim; j++) {
            util_sum += nodes[i][j].is_util();
        }
    }

    return (float)util_sum / (_dim * _dim);
}

