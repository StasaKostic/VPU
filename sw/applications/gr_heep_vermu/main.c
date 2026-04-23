#include <stdio.h>
#include <stdint.h>

int main(void) {
    
    uint8_t data[4] = {1,2,3,4};
    uint8_t out[4];
    uint8_t out_expected[4] = {2,3,4,5};
    // Load 4 elements from memory into vector register v0
    asm volatile("vle8.v v0, (%0)" :: "r"(data));
    // Configure vector length (4 elements) and element width (8 bits)
    asm volatile ("vsetivli t0, 4, e8, m1, ta, ma");
    // Increment each element in vector register v0 by 1 and store the result in v1
    asm volatile("vadd.vi v1, v0, 1");
    // Store the result from vector register v1 back to memory
    asm volatile("vse8.v v1, (%0)" :: "r"(out));
    for (int i = 0; i < 4; i++) {
        if (out[i] != out_expected[i]) {
            printf("Test failed at index %d: expected %u, got %u\n", i, out_expected[i], out[i]);
        }
    }
    printf("Test ended.\n");


    return 0;
}