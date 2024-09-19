#ifndef NODEPU_CPP
#define NODEPU_CPP
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include "fifo.cpp"
#include <bitset>
#include <climits>

using namespace std;

template <typename T>
class node_pu
{
private:
    string _name;
    uint8_t _threads;
    uint16_t _max_depth;
    int _acc = 0;
    vector<uint32_t> _acc_t;
    vector<bool> _halt;
    bool _is_util;
    uint8_t _rr_start;

public:
    vector<fifo<T>> _buf_a, _buf_b;
    node_pu *out_a, *out_b;

    node_pu(string name, uint8_t threads, uint16_t max_depth);

    T pop(bool buf, uint8_t thread, bool &is_empty);
    T get_acc() { return _acc; };

    uint32_t get_acc_t(uint8_t thread) { return _acc_t[thread]; };
    uint32_t get_max_buf_util();

    string get_name() { return _name; };

    void push(T x, bool buf, uint8_t thread);
    void reset_acc() { _acc = 0; };
    void reset_acc_t();
    void reset_acc_t(uint8_t thread) { _acc_t[thread] = 0; };
    void halt(uint8_t thread) { _halt[thread] = true; };
    void release();
    void release(uint8_t thread) { _acc_t[thread] = false; };
    void cycle();
    // int go(T* MultiplierBits, T* AccumulatorBits, T* InputABits, T* InputBBits,int stuck_bit);

    int go(T *MultiplierBits, T *AccumulatorBits, T *InputABits, T *InputBBits);
    bool is_valid(uint8_t thread);
    bool is_ready(bool buf, uint8_t thread);
    bool is_ready_out(uint8_t thread);
    bool is_halt(uint8_t thread) { return _halt[thread]; };
    bool is_util() { return _is_util; };
};

template <typename T>
node_pu<T>::node_pu(string name, uint8_t threads, uint16_t max_depth) : _name(name), _threads(threads),
                                                                        _max_depth(max_depth), _rr_start(0), out_a(0), out_b(0)
{
    _buf_a.reserve(threads);
    _buf_b.reserve(threads);
    _acc_t.reserve(threads);
    _halt.reserve(threads);

    for (uint8_t i = 0; i < threads; i++)
    {
        _buf_a.push_back(fifo<T>("fifo_a_" + name));
        _buf_b.push_back(fifo<T>("fifo_b_" + name));
        _acc_t.push_back(0);
        _halt.push_back(false);
    }
}

template <typename T>
void node_pu<T>::push(T x, bool buf, uint8_t thread)
{
    assert(thread < _threads);

    if (buf == 0)
        _buf_a[thread].push(x);
    else
        _buf_b[thread].push(x);
}

template <typename T>
T node_pu<T>::pop(bool buf, uint8_t thread, bool &is_empty)
{
    assert(thread < _threads);

    if (buf == 0)
        return _buf_a[thread].pop(is_empty);
    else
        return _buf_b[thread].pop(is_empty);
}

template <typename T>
bool node_pu<T>::is_valid(uint8_t thread)
{
    if (_buf_a[thread].size() == 0 || _buf_b[thread].size() == 0)
        return false;
    else
        return true;
}

template <typename T>
bool node_pu<T>::is_ready(bool buf, uint8_t thread)
{
    if ((buf == 0 && _buf_a[thread].size() < _max_depth) || (buf == 1 && _buf_b[thread].size() < _max_depth))
        return true;
    else
        return false;
}

template <typename T>
bool node_pu<T>::is_ready_out(uint8_t thread)
{
    bool a_out_ready, b_out_ready;

    if (out_a == 0)
        a_out_ready = true;
    else
        a_out_ready = out_a->is_ready(0, thread);

    if (out_b == 0)
        b_out_ready = true;
    else
        b_out_ready = out_b->is_ready(1, thread);

    return a_out_ready && b_out_ready;
}

template <typename T>
int node_pu<T>::go(T MultiplierBits[sizeof(int16_t) * CHAR_BIT], T AccumulatorBits[sizeof(int) * CHAR_BIT], T InputABits[sizeof(T) * CHAR_BIT], T InputBBits[sizeof(T) * CHAR_BIT])
{
    _is_util = false;
    int accessed = 0;
    int tmp = 0;
    union
    {
        T input;
        int32_t output;
    } BinaryData;
    union
    {
        int32_t input1;
        int32_t output1;
    } BinaryData1;
    for (uint8_t i = 0; i < _threads; i++)
    {
        uint8_t t = (_rr_start + i) % _threads;

        if (is_valid(t) && !is_halt(t) && is_ready_out(t))
        {
            bool is_a_empty, is_b_empty;
            T a = pop(0, t, is_a_empty);
            T b = pop(1, t, is_b_empty);
            // cout<< _name << " Inputs: " << a << "   " << b << endl;
            assert(!is_a_empty && !is_b_empty);

            _acc_t[t]++;

            // if(stuck_bit > 0){
            //	a = a&(~(1<<(stuck_bit-1)));
            //	}
            BinaryData.input = a;
            std::bitset<sizeof(uint8_t) * CHAR_BIT> InputABits_tmp(BinaryData.output); // bit number 8 is the sign bit (MSB)
                                                                                       // if(a>127)
            // cout << "A Decimal:  " << a << "    Binary:    " << InputABits_tmp <<endl;
            BinaryData.input = b;
            std::bitset<sizeof(int8_t) * CHAR_BIT> InputBBits_tmp(BinaryData.output); // bit number 8 is the sign bit (MSB)
            // if(a>127 || a<-128)
            // cout << "B Decimal: " << b << "     Binary:    " << InputBBits_tmp <<endl;
            // std::cout<< _name << endl;
            BinaryData1.input1 = a * b;
            std::bitset<sizeof(int16_t) * CHAR_BIT> MultiplierBits_tmp(BinaryData1.output1); // bit number 16 is the sign bit (MSB)
                                                                                             // cout << "A*B Decimal: " << a*b << "     Binary:    " << MultiplierBits_tmp <<endl;
            BinaryData1.input1 = _acc + (a * b);
            std::bitset<sizeof(int32_t) * CHAR_BIT> AccumulatorBits_tmp(BinaryData1.output1); // bit number 32 is the sign bit (MSB)
            // std::cout<< _acc << endl;
            // cout<< (int) a << "          " <<  (int)b << "           " << a*b<<"                  casted:" << (int)a*b <<endl;
            for (uint16_t bits = 0; bits < (sizeof(T) * CHAR_BIT); bits++)
            {
                if (bits < 8)
                {
                    InputABits[bits] = InputABits_tmp[bits];
                    InputBBits[bits] = InputBBits_tmp[bits];
                }
                if (bits < 16)
                    MultiplierBits[bits] = MultiplierBits_tmp[bits];

                AccumulatorBits[bits] = AccumulatorBits_tmp[bits];
            }
            if (out_a != 0)
                out_a->push(a, 0, t);
            if (out_b != 0)
                out_b->push(b, 1, t);
            if (a != 0 and b != 0)
            {
                accessed = 1;
                tmp = a * b;
                _acc += tmp;

                // std::cout << _name << "      "<< tmp << "   " << MultiplierBits[31]<<endl;
                _is_util = true;
                _rr_start = (t + 1) % _threads;
                break;
            }
        }
    }
    return accessed;
}

template <typename T>
void node_pu<T>::reset_acc_t()
{
    for (uint8_t t = 0; t < _threads; t++)
        _acc_t[t] = 0;
}

template <typename T>
void node_pu<T>::release()
{
    for (uint8_t t = 0; t < _threads; t++)
        _halt[t] = false;
}

template <typename T>
void node_pu<T>::cycle()
{
    for (uint8_t t = 0; t < _threads; t++)
    {
        _buf_a[t].cycle();
        _buf_b[t].cycle();
    }
}

#endif
