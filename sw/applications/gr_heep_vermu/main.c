#include <stdio.h>
#include <stdint.h>

#define ARRAY_LEN 4 // Vector length in elements

void test_vector_scalar_add() {
    
    uint8_t data[ARRAY_LEN];
    for (int i = 0; i < ARRAY_LEN; i++) {
        data[i] = i; 
    }    
    uint8_t out[ARRAY_LEN];
    uint8_t out_expected[ARRAY_LEN];
    for (int i = 0; i < ARRAY_LEN; i++) {
        out_expected[i] = data[i] + 1;
    }
    // Configure vector length (4 elements) and element width (8 bits)
    printf("Configure vector length and element width.\n");
    asm volatile ("vsetivli t0, 4, e8, m1, ta, ma");
    // Load 4 elements from memory into vector register v0
    printf("Load values to the vector register.\n");
    asm volatile("vle8.v v0, (%0)" :: "r"(data));
    // Increment each element in vector register v0 by 1 and store the result in v1
    printf("Increment elements in the vector register.\n");
    asm volatile("vadd.vi v1, v0, 1");
    // Store the result from vector register v1 back to memory
    printf("Store the result back to memory.\n");
    asm volatile("vse8.v v1, (%0)" :: "r"(out));
    for (int i = 0; i < ARRAY_LEN; i++) {
        printf("out[%d] = %u\n", i, out[i]);
        if (out[i] != out_expected[i]) {
            printf("Test failed at index %d: expected %u but got %u\n", i, out_expected[i], out[i]);
            return;
        }
    }
    printf("Test passed.\n");
}

void test_vector_vector_add() {

    uint8_t data0[ARRAY_LEN];
    uint8_t data1[ARRAY_LEN];
    uint8_t out_expected[ARRAY_LEN];
    uint8_t out[ARRAY_LEN];
    for (int i = 0; i < ARRAY_LEN; i++) {
        data0[i] = i;
        data1[i] = 2*i;
        out_expected[i] = data0[i] + data1[i];
    }
    // Configure vector length (4 elements) and element width (8 bits)
    printf("Configure vector length and element width.\n");
    asm volatile ("vsetivli t0, 4, e8, m1, ta, ma");
    // Load 4 elements from memory into vector register v0
    printf("Load values to the vector register 0.\n");
    asm volatile("vle8.v v0, (%0)" :: "r"(data0));
    printf("Load values to the vector register 1.\n");
    asm volatile("vle8.v v1, (%0)" :: "r"(data1));
    // Add elements of vector register v0 and v1, store the result in v2
    printf("Add vectors.\n");
    asm volatile("vadd.vv v2, v0, v1");
    // Store the result from vector register v2 back to memory
    printf("Store the result back to memory.\n");
    asm volatile("vse8.v v2, (%0)" :: "r"(out));
    for (int i = 0; i < ARRAY_LEN; i++) {
        printf("out[%d] = %u\n", i, out[i]);
        if (out[i] != out_expected[i]) {
            printf("Test failed at index %d: expected %u but got %u\n", i, out_expected[i], out[i]);
            return;
        }
    }
    printf("Test passed.\n");    

}

int main(void) {
    
    printf("Test started.\n");
    printf("Test adding scalar to vector elements.\n");
    test_vector_scalar_add();
    printf("Test ended.\n");

    return 0;
}