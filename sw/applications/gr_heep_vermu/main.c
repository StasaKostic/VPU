#include <stdio.h>
#include <stdint.h>

void test_vector_scalar_add() {
    uint8_t data[4] = {1,2,3,4};
    uint8_t out[4];
    uint8_t out_expected[4] = {2,3,4,5};
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
    for (int i = 0; i < 4; i++) {
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