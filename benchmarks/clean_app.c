#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define ITERATIONS 20
#define ALLOC_SIZE 1024

int main() {
    printf("[CLEAN] Starting clean memory allocation workload...\n");

    for (int i = 0; i < ITERATIONS; i++) {
        char *ptr = (char *)malloc(ALLOC_SIZE);
        if (ptr == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }
        ptr[0] = 'A';
        usleep(100000);
        free(ptr);
        printf("[CLEAN] Iteration %d/%d: Allocated and freed %d bytes\n", i + 1, ITERATIONS, ALLOC_SIZE);
    }

    printf("[CLEAN] Completed cleanly. All allocations freed.\n");
    return 0;
}
