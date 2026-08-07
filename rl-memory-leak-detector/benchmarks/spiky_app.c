#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define BATCH_SIZE 5
#define SPIKE_SIZE 102400 // 100 KB
#define CYCLES 3

int main() {
    printf("[SPIKY] Starting spiky workload (large temporary allocations)...\n");

    for (int cycle = 0; cycle < CYCLES; cycle++) {
        void *ptrs[BATCH_SIZE];

        printf("\n[SPIKY] Cycle %d: Burst allocating %d large blocks...\n", cycle + 1, BATCH_SIZE);
        for (int i = 0; i < BATCH_SIZE; i++) {
            ptrs[i] = malloc(SPIKE_SIZE);
            if (ptrs[i] == NULL) {
                fprintf(stderr, "Allocation failed\n");
                return 1;
            }
            usleep(50000); // 50ms interval between burst allocations
        }

        printf("[SPIKY] Cycle %d: Holding memory spike in active use...\n", cycle + 1);
        sleep(1); // Held for 1 second (static heuristics often falsely flag this!)

        printf("[SPIKY] Cycle %d: Processing done. Freeing burst allocation...\n", cycle + 1);
        for (int i = 0; i < BATCH_SIZE; i++) {
            free(ptrs[i]);
        }

        usleep(200000); // 200ms rest between spikes
    }

    printf("\n[SPIKY] Workload finished. All temporary spikes released.\n");
    return 0;
}