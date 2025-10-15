#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "simulation.h"
#include <time.h>
#include <sys/stat.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdarg.h> 
#include <pthread.h>
//#include <direct.h>
//#define DEBUG
#define ALL_SIMULATION_COUNT 20
#define SAVE_LINES 5

typedef struct {
    float gain;
    float gbw;
    float phase_margin;
    float idc;
    float noise;
    float cmrr;
    float sr;
    float psrr;
    bool error;
    float mos[50][39];
    float R[50][18];
    float C[50][18];
    float V[1][18];
    float I[1][18];
} Performance;

typedef struct {
    char cmd[512];
    char test[512];
    bool op_aly;
} Task;

typedef struct {
    float dev_parm[20];
    int sim_times;
    int id;
} Param;

Performance all_performances[ALL_SIMULATION_COUNT];
Performance final_perf;
int simulation_count = 0;
pthread_mutex_t result_mutex = PTHREAD_MUTEX_INITIALIZER;


char* Append_Folder(char* folder, int size, int sim_times, int id){
    int len = strlen(folder);
    char temp[1000];
    if(folder[len-1] != '/'){
        folder[len] = '/';
        folder[len+1] = '\0';
        len++;
    }

    snprintf(folder+len, size-len, "%d_%c", sim_times, id);
    int result = mkdir(folder, 0755);

    snprintf(temp, sizeof(temp), "%s/TT", folder);
    result += mkdir(temp, 0755);
    snprintf(temp, sizeof(temp), "%s/SS", folder);
    result += mkdir(temp, 0755);
    snprintf(temp, sizeof(temp), "%s/SF", folder);
    result += mkdir(temp, 0755);
    snprintf(temp, sizeof(temp), "%s/FS", folder);
    result += mkdir(temp, 0755);
    snprintf(temp, sizeof(temp), "%s/FF", folder);
    result += mkdir(temp, 0755);
#ifdef DEBUG
    if(result != 0){
        printf("Error at mkdir\n");
    }
#endif
    return folder;
}

bool Write_File(char* folder, char* file_path, float* parms, int corner_id, const char* file){
    FILE *fp;
    char filepath[1000];
    char temp[50000];

    snprintf(filepath, sizeof(filepath), "%s/%s", folder, file_path);
    fp = fopen(filepath,"w");
    if (!fp) {
        perror("No such file.");
        return false;
    }

    if(file == AMP_AC){
        if(corner_id == 0){
            snprintf(temp, sizeof(temp), file, corners_id[corner_id],OP_str);
        }else{
            snprintf(temp, sizeof(temp), file, corners_id[corner_id],"");
        }
    }
    else if(parms != NULL){
        snprintf(temp, sizeof(temp), file,  parms[0],
                                            parms[1],
                                            parms[2],
                                            parms[3],
                                            parms[4],
                                            parms[5],
                                            parms[6],
                                            parms[7],
                                            parms[8],
                                            parms[9],
                                            parms[10],
                                            parms[11],
                                            parms[12],
                                            parms[13],
                                            parms[14],
                                            parms[15],
                                            parms[16],
                                            parms[17],
                                            parms[18],
                                            parms[19]);
    }else if(corner_id >= 0 && corner_id < ALL_SIMULATION_COUNT){
        snprintf(temp, sizeof(temp), file, corners_id[corner_id]);
    }else{
        snprintf(temp, sizeof(temp), "%s", file);
    }

    fprintf(fp, "%s", temp);
    fclose(fp);

    return true;
}

bool Write_All(char* folder, float* parms){
    Write_File(folder, "/TT/AMP_AC.cir", NULL, 0, AMP_AC);
    Write_File(folder, "/TT/AMP_IDC.cir", NULL, 0, AMP_IDC);
    Write_File(folder, "/TT/AMP_SR.cir", NULL, 0, AMP_SR);
    Write_File(folder, "/FF/AMP_AC.cir", NULL, 1, AMP_AC);
    Write_File(folder, "/FF/AMP_IDC.cir", NULL, 1, AMP_IDC);
    Write_File(folder, "/FF/AMP_SR.cir", NULL, 1, AMP_SR);
    Write_File(folder, "/FS/AMP_AC.cir", NULL, 2, AMP_AC);
    Write_File(folder, "/FS/AMP_IDC.cir", NULL, 2, AMP_IDC);
    Write_File(folder, "/FS/AMP_SR.cir", NULL, 2, AMP_SR);
    Write_File(folder, "/SF/AMP_AC.cir", NULL, 3, AMP_AC);
    Write_File(folder, "/SF/AMP_IDC.cir", NULL, 3, AMP_IDC);
    Write_File(folder, "/SF/AMP_SR.cir", NULL, 3, AMP_SR);
    Write_File(folder, "/SS/AMP_AC.cir", NULL, 4, AMP_AC);
    Write_File(folder, "/SS/AMP_IDC.cir", NULL, 4, AMP_IDC);
    Write_File(folder, "/SS/AMP_SR.cir", NULL, 4, AMP_SR);
    Write_File(folder, "/param", parms, 5, param);
    Write_File(folder, "/AMP.cir", NULL, 5, AMP1);
    Write_File(folder, "/AMP_CMRR.cir", NULL, 5, AMP_CMRR);
    Write_File(folder, "/AMP_noise.cir", NULL, 5, AMP_noise);
    Write_File(folder, "/AMP_PSRR.cir", NULL, 5, AMP_PSRR);
}

void* Do_Simulation_Analyse(void* arg){
    Task* task = (Task*)arg;
    char lines[SAVE_LINES][300];
    char buf[300];
    Performance local_perf = {
        .gain = 1e20,
        .gbw = 1e20,
        .phase_margin = 1e20,
        .idc = 0.0,
        .noise = 0.0,
        .cmrr = 1e20, 
        .sr = 1e20,
        .psrr = 1e20
    };//local performance save

    char full_cmd[512];
    snprintf(full_cmd, sizeof(full_cmd), "%s %s", task->cmd, task->test);
#ifdef DEBUG
    printf("command: %s\n",full_cmd);
#endif
    //do simulation
    FILE *fp;
    fp = popen(full_cmd,"r");


    int line_count = 0;
    if(task->op_aly){
        char op_buf[1000][300];
        int device = 0;
        int count = 0;
        while(fgets(buf, sizeof(buf)-1, fp) != NULL){
            buf[strcspn(buf, "\n\r")] = 0;
            strcpy(op_buf[line_count], buf);
            line_count++;
        }
        for(int i = 0; i < line_count; i++){
            char name[128];
            float value;
            if(sscanf(op_buf[i], "%s = %e", name, &value) == 2){
#ifdef DEBUG
                printf("get output val %s: %f\n", name, value);

#endif
                if(device < 10){
                    final_perf.mos[device][count] = value;
#ifdef DEBUG
                    printf("device: %d count: %d value: %f\n",device, count, final_perf.mos[device][count]);
#endif
                }else{
                    final_perf.C[device][count] = value;
                }
                count++;
                if(count >= 39 && device < 10){
                    count = 0;
                    device++;
                }
            }
        }// this 'for' part could be combined in 'while' reading process, further work 

#ifdef DEBUG
        printf("finish op analyze\n");
#endif
        pclose(fp);
        pthread_mutex_lock(&result_mutex);
        simulation_count++;
        pthread_mutex_unlock(&result_mutex);

        pthread_exit(NULL);
    }else{//add last SAVE_LINES to "lines" buffer 
        while(fgets(buf, sizeof(buf)-1, fp) != NULL){
#ifdef DEBUG
            printf("reveal output: %s", buf);
#endif
            buf[strcspn(buf, "\n\r")] = 0;
            strcpy(lines[line_count % SAVE_LINES], buf);
            line_count++;
        }
#ifdef DEBUG
            printf("review str in lines:\n");
            printf(lines[0]);
            printf("\n");
            printf(lines[1]);
            printf("\n");
            printf(lines[2]);
            printf("\n");
            printf(lines[3]);
            printf("\n");
            printf(lines[4]);
            printf("\n");
#endif
        int total = (line_count > SAVE_LINES) ? SAVE_LINES : line_count;
        int start = (line_count - 1) % SAVE_LINES;

        int found = 0;
        bool has_gain = false, has_gbw = false, has_pm = false;
        bool has_idc = false, has_sr = false;
        bool has_cmrr = false, has_psrr = false, has_noise = false;

        for (int i = 0; i < total && i < SAVE_LINES; i++) {
            int idx = (start - i + SAVE_LINES) % SAVE_LINES;
            char* line = lines[idx];

            float value = 1.0;
            char name[128];

            if (sscanf(line, "%s = %e", name, &value) == 2 ||
                sscanf(line, "%s = %f", name, &value) == 2) {
                if (strcmp(name, "gain_max") == 0) {
                    local_perf.gain = value;
                    has_gain = true;
                }
                else if (strcmp(name, "gbw") == 0) {
                    local_perf.gbw = value;
                    has_gbw = true;
                }
                else if (strcmp(name, "phase_margin") == 0) {
                    local_perf.phase_margin = value;
                    has_pm = true;
                }
                else if (strcmp(name, "i(vmeas)") == 0) {
                    local_perf.idc = value;
                    has_idc = true;
                }
                else if (strcmp(name, "sr") == 0) {
                    local_perf.sr = value;
                    has_sr = true;
                }
                else if (strcmp(name, "cm_gain") == 0) {
                    local_perf.cmrr = - value;
                    has_cmrr = true;
                }
                else if (strcmp(name, "noise_1k") == 0) {
                    local_perf.noise = value;
                    has_noise = true;
                }
                else if (strcmp(name, "vdd_gain") == 0) {
                    local_perf.psrr = -value;
                    has_psrr = true;
                }else{
                    //simulation error
                    local_perf.error =true;
                }

                if ((has_gbw && has_pm && has_gain) || has_idc || has_sr || has_cmrr || has_noise || has_psrr || local_perf.error) break;
            }
        }
#ifdef DEBUG
        printf("finish analyze\n");
#endif
        pclose(fp);
        pthread_mutex_lock(&result_mutex);
        if(local_perf.error){
            Performance worst = {
                .gain = 0,
                .gbw = 0,
                .phase_margin = 0,
                .idc = 1e20,
                .noise = 1e20,
                .cmrr = 0, 
                .sr = 0,
                .psrr = 0
            };
            all_performances[simulation_count] = worst;
        }else{
            all_performances[simulation_count] = local_perf;
        }
        simulation_count++;
        pthread_mutex_unlock(&result_mutex);

        pthread_exit(NULL);
    }
}

Performance find_worst_case_performance() {
    Performance current = {
        .gain = 1e20,
        .gbw = 1e20,
        .phase_margin = 1e20,
        .idc = 0.0,
        .noise = 0.0,
        .cmrr = 1e20, 
        .sr = 1e20,
        .psrr = 1e20
    };

    for (int i = 0; i < ALL_SIMULATION_COUNT; i++) {
        Performance p = all_performances[i];

        if (p.gain < current.gain)        current.gain = p.gain;
        if (p.gbw < current.gbw)          current.gbw = p.gbw;
        if (p.phase_margin < current.phase_margin) current.phase_margin = p.phase_margin;
        if (p.sr < current.sr)            current.sr = p.sr;
        if (p.cmrr < current.cmrr)        current.cmrr = p.cmrr;
        if (p.psrr < current.psrr)        current.psrr = p.psrr;
        if (p.idc > current.idc)          current.idc = p.idc;
        if (p.noise > current.noise)          current.noise = p.noise;
    }
    return current;
}

void Initial_All_Performance(){
    for(int i =0; i < ALL_SIMULATION_COUNT; i++){
        all_performances[i] = (Performance){
        .gain = 1e20,
        .gbw = 1e20,
        .phase_margin = 1e20,
        .idc = 0.0,
        .noise = 0.0,
        .cmrr = 1e20, 
        .sr = 1e20,
        .psrr = 1e20
        };
    }
}

void print_performance(Performance p){
    printf("\nprint_perf: %f, %f, %f, %f, %f, %f, %.20f, %f", p.gain, p.gbw, p.phase_margin, p.idc, p.sr, p.cmrr, p.noise, p.psrr);
    for(int i=0;i<39;i++){
        printf("val: %.20f",p.mos[0][i]);
    }
}

int Simulate_Op(char* folder){
    char prefix_cmd[1024];
    Task task;
    snprintf(prefix_cmd, sizeof(prefix_cmd),
             "%s/../../Circuit/ngspice-43/release/src/ngspice -b ",
             folder);
    snprintf(task.cmd, sizeof(task.cmd), "%s", 
                     prefix_cmd);
    snprintf(task.test, sizeof(task.test), "%s/%s%s", 
                     folder, "TT", "/AMP_AC.cir");
    task.op_aly = true;
    pthread_t thread;
    pthread_create(&thread, NULL, Do_Simulation_Analyse, &task);
    pthread_join(thread, NULL);
}

int Simulate_All(char* folder){
    char prefix_cmd[1024];
    snprintf(prefix_cmd, sizeof(prefix_cmd),
             "%s/../../Circuit/ngspice-43/release/src/ngspice -b ",
             folder);

    const char* corners[] = {"TT", "SS", "SF", "FS", "FF"};
    const char* tests_ac[] = {"/AMP_AC.cir", "/AMP_IDC.cir", "/AMP_SR.cir"};
    const char* tests_single[] = {"/AMP_CMRR.cir", "/AMP_noise.cir", "/AMP_PSRR.cir"};

    int total_tasks = 0;
    Task tasks[100];

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 3; j++) {
            snprintf(tasks[total_tasks].cmd, sizeof(tasks[total_tasks].cmd), "%s", 
                     prefix_cmd);
            snprintf(tasks[total_tasks].test, sizeof(tasks[total_tasks].test), "%s/%s%s", 
                     folder, corners[i], tests_ac[j]);
            if(i == 0 && j == 0){
                tasks[total_tasks].op_aly = true;
            }else{
                tasks[total_tasks].op_aly = false;
            }
            total_tasks++;
        }
    }

    for (int i = 0; i < 3; i++) {
        snprintf(tasks[total_tasks].cmd, sizeof(tasks[total_tasks].cmd), "%s", 
                 prefix_cmd);
        snprintf(tasks[total_tasks].test, sizeof(tasks[total_tasks].test), "%s%s", folder, tests_single[i]);
        total_tasks++;
    }

    pthread_t threads[total_tasks];
    for (int i = 0; i < total_tasks; i++) {
        pthread_create(&threads[i], NULL, Do_Simulation_Analyse, &tasks[i]);
    }

#ifdef DEBUG
    printf("\t\t thread created\n");
#endif
    for (int i = 0; i < total_tasks; i++) {
        pthread_join(threads[i], NULL);
    }

    Performance worst = find_worst_case_performance();
    final_perf.gain = worst.gain;
    final_perf.gbw = worst.gbw;
    final_perf.phase_margin = worst.phase_margin;
    final_perf.idc = worst.idc;
    final_perf.noise = worst.noise;
    final_perf.cmrr = worst.cmrr;
    final_perf.sr = worst.sr;
    final_perf.psrr = worst.psrr;

    return 0;
}

void Get_Op_Range(Param P){
    char folder[1500];

    if (getcwd(folder, sizeof(folder)) == NULL) {
        perror("Getting path failed.");//printf("in: %s\n", folder);
    }
    Append_Folder(folder, sizeof(folder), P.sim_times, P.id);
    Write_File(folder, "/TT/AMP_AC.cir", NULL, 0, AMP_AC);
    Write_File(folder, "/AMP.cir", NULL, 5, AMP1);
    Write_File(folder, "/param", P.dev_parm, 5, param);
    Simulate_Op(folder);
}

void Simulate(Param p){
    char folder[1500];
    Initial_All_Performance();
    if (getcwd(folder, sizeof(folder)) == NULL) {
        perror("Getting path failed.");//printf("in: %s\n", folder);
    }
    Append_Folder(folder, sizeof(folder), p.sim_times, p.id);
    Write_All(folder, p.dev_parm);
    Simulate_All(folder);
}

int main(){
    char folder[1500];
    float parms[] = {1, 25, 1, 25, 1, 25, 1, 25, 1, 25, 1, 10, 1, 20, 1, 20, 1, 20, 5, 62300};
    Param x = {
        .dev_parm = {1, 25, 1, 25, 1, 25, 1, 25, 1, 25, 1, 10, 1, 20, 1, 20, 1, 20, 5, 62300},
        .sim_times = 0,
        .id = 'a',
    };
    Get_Op_Range(x);
    return 0;
    Initial_All_Performance();
    if (getcwd(folder, sizeof(folder)) == NULL) {
        perror("Getting path failed.");//printf("in: %s\n", folder);
    }
    Append_Folder(folder, sizeof(folder), 2, 'C');
    Write_All(folder, parms);
    Simulate_All(folder);
#ifdef DEBUG
    print_performance(final_perf);
#endif
    return 0;
}