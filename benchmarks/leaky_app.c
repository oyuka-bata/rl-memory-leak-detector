#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define ITERATIONS 30
#define ALLOC_SIZE 2048

int main() {
    printf("[LEAKY] Starting leaky memory allocation workload...\n");

    for (int i = 0; i < ITERATIONS; i++) {
        char *ptr = (char *)malloc(ALLOC_SIZE);
        if (ptr == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }
        ptr[0] = 'X';
        printf("[LEAKY] Iteration %d/%d: Leaked %d bytes at address %p\n", i + 1, ITERATIONS, ALLOC_SIZE, (void *)ptr);
        usleep(150000);
    }

    printf("[LEAKY] Finished allocation loop. Exiting (leaked %d bytes total).\n", ITERATIONS * ALLOC_SIZE);
    return 0;
}
