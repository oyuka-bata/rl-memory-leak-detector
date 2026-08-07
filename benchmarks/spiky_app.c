#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define BATCH_SIZE 5
#define SPIKE_SIZE 102400
#define CYCLES 3

int main() {
    printf("[SPIKY] Starting spiky workload...\n");

    for (int cycle = 0; cycle < CYCLES; cycle++) {
        void *ptrs[BATCH_SIZE];

        printf("[SPIKY] Cycle %d: Burst allocating %d large blocks...\n", cycle + 1, BATCH_SIZE);
        for (int i = 0; i < BATCH_SIZE; i++) {
            ptrs[i] = malloc(SPIKE_SIZE);
            if (ptrs[i] == NULL) return 1;
            usleep(50000);
        }

        printf("[SPIKY] Cycle %d: Holding memory spike...\n", cycle + 1);
        sleep(1);

        printf("[SPIKY] Cycle %d: Freeing burst allocation...\n", cycle + 1);
        for (int i = 0; i < BATCH_SIZE; i++) {
            free(ptrs[i]);
        }

        usleep(200000);
    }

    printf("[SPIKY] Workload finished. All temporary spikes released.\n");
    return 0;
}
