// Loaded with DYLD_INSERT_LIBRARIES: every 200 ms, writes the page faults and CPU times of this
// process to $FAULTS_DIR/<pid>.tsv.
#include <mach/mach.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

static FILE *out;
static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static void sample(double t0) {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  task_events_info_data_t ev;
  mach_msg_type_number_t n = TASK_EVENTS_INFO_COUNT;
  task_info(mach_task_self(), TASK_EVENTS_INFO, (task_info_t)&ev, &n);
  task_vm_info_data_t vm;
  n = TASK_VM_INFO_COUNT;
  task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm, &n);
  fprintf(out, "%.3f\t%ld\t%ld\t%d\t%d\t%d\t%.3f\t%.3f\t%llu\t%llu\n", now() - t0, ru.ru_minflt,
          ru.ru_majflt, ev.faults, ev.pageins, ev.cow_faults,
          ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6,
          ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6,
          (unsigned long long)vm.phys_footprint, (unsigned long long)vm.resident_size);
  fflush(out);
}
static void *loop(void *arg) {
  double t0 = *(double *)arg;
  for (;;) {
    sample(t0);
    usleep(200000);
  }
  return 0;
}
static double start;
__attribute__((constructor)) static void begin(void) {
  const char *dir = getenv("FAULTS_DIR");
  if (!dir)
    return;
  char path[1024];
  snprintf(path, sizeof path, "%s/%d.tsv", dir, getpid());
  out = fopen(path, "w");
  if (!out)
    return;
  fprintf(out, "t\tminflt\tmajflt\tfaults\tpageins\tcow\tuser\tsys\tfootprint\tresident\n");
  start = now();
  pthread_t th;
  pthread_create(&th, 0, loop, &start);
  pthread_detach(th);
}
__attribute__((destructor)) static void end(void) {
  if (out)
    sample(start);
}
